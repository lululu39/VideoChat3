#!/usr/bin/env python3
"""Teacher-forced sweep of the learned v4 LM-facing FW feature residual."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_videochat3_base_visual_token_perturbations as feature_tools
import eval_videochat3_lact_teacher_forced_ablation as common


ALPHAS = (-2.0, 0.0, 0.5, 1.0, 2.0, 4.0)


def alpha_label(alpha: float) -> str:
    rendered = f"{abs(alpha):g}".replace(".", "p")
    return f"alpha_{'m' if alpha < 0 else ''}{rendered}"


CONDITIONS = tuple(alpha_label(alpha) for alpha in ALPHAS)
ALPHA_BY_CONDITION = dict(zip(CONDITIONS, ALPHAS, strict=True))
COMPARISONS = {
    "alpha_0p5_minus_alpha_1": (alpha_label(0.5), alpha_label(1.0)),
    "alpha_0_minus_alpha_1": (alpha_label(0.0), alpha_label(1.0)),
    "alpha_2_minus_alpha_1": (alpha_label(2.0), alpha_label(1.0)),
    "alpha_4_minus_alpha_1": (alpha_label(4.0), alpha_label(1.0)),
    "alpha_2_minus_alpha_m2": (alpha_label(2.0), alpha_label(-2.0)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--annotation-xlsx", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=96)
    parser.add_argument("--max-questions-per-video", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--min-pixels", type=int, default=28 * 28)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--total-pixels", type=int, default=80_000 * 2 * 4 * 14 * 14)
    return parser.parse_args()


def interpolate_fw_residual(
    base_features: torch.Tensor,
    v4_features: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if base_features.shape != v4_features.shape:
        raise ValueError(
            f"Base/v4 feature shape mismatch: {base_features.shape} vs {v4_features.shape}"
        )
    if alpha == 0.0:
        return base_features
    if alpha == 1.0:
        return v4_features
    output = torch.empty_like(base_features)
    for section in feature_tools.chunk_slices(base_features.shape[0]):
        base = base_features[section].float()
        residual = v4_features[section].float() - base
        output[section] = (base + alpha * residual).to(base_features.dtype)
    return output


def residual_metrics(
    base_features: torch.Tensor,
    v4_features: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    metrics = feature_tools.feature_metrics(base_features, candidate)
    residual_sq = 0.0
    candidate_delta_dot_residual = 0.0
    for section in feature_tools.chunk_slices(base_features.shape[0]):
        base = base_features[section].float()
        residual = v4_features[section].float() - base
        candidate_delta = candidate[section].float() - base
        residual_sq += residual.square().sum().item()
        candidate_delta_dot_residual += (candidate_delta * residual).sum().item()
    if residual_sq <= 0:
        raise ValueError("v4 projected FW residual has zero norm")
    metrics["realized_alpha_projection"] = candidate_delta_dot_residual / residual_sq
    return metrics


def summarize(output_dir: Path, bootstrap_samples: int, seed: int, protocol: dict):
    rows = []
    for path in sorted(output_dir.glob("rank-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not rows:
        raise RuntimeError("No sweep rows were produced")
    expected = protocol["selected_questions"]
    unique = {(row["video"], row["index"]) for row in rows}
    if len(rows) != expected or len(unique) != expected:
        raise RuntimeError(
            f"Expected {expected} unique rows, got rows={len(rows)}, unique={len(unique)}"
        )

    summary = {"protocol": protocol, "conditions": {}, "deltas": {}}
    for condition in CONDITIONS:
        values = [row["conditions"][condition]["nll"] for row in rows]
        video_feature_metrics = {
            row["video"]: row["feature_metrics"][condition] for row in rows
        }
        summary["conditions"][condition] = {
            "alpha": ALPHA_BY_CONDITION[condition],
            "mean_nll": float(np.mean(values)),
            "median_nll": float(np.median(values)),
            "mean_answer_probability": float(
                np.mean(
                    [row["conditions"][condition]["answer_probability"] for row in rows]
                )
            ),
            "mean_feature_relative_l2_delta": float(
                np.mean(
                    [value["relative_l2_delta"] for value in video_feature_metrics.values()]
                )
            ),
            "mean_feature_cosine_similarity": float(
                np.mean(
                    [value["cosine_similarity"] for value in video_feature_metrics.values()]
                )
            ),
            "mean_feature_norm_ratio": float(
                np.mean(
                    [
                        value["candidate_to_reference_norm"]
                        for value in video_feature_metrics.values()
                    ]
                )
            ),
            "mean_realized_alpha_projection": float(
                np.mean(
                    [
                        value["realized_alpha_projection"]
                        for value in video_feature_metrics.values()
                    ]
                )
            ),
        }

    for comparison_index, (name, (left, right)) in enumerate(COMPARISONS.items()):
        question_deltas = [
            row["conditions"][left]["nll"] - row["conditions"][right]["nll"]
            for row in rows
        ]
        grouped = {}
        for video in sorted({row["video"] for row in rows}):
            grouped[video] = float(
                np.mean(
                    [
                        row["conditions"][left]["nll"]
                        - row["conditions"][right]["nll"]
                        for row in rows
                        if row["video"] == video
                    ]
                )
            )
        summary["deltas"][name] = {
            "left": left,
            "right": right,
            "mean_nll_delta": float(np.mean(question_deltas)),
            "median_nll_delta": float(np.median(question_deltas)),
            "positive_fraction": float(np.mean(np.asarray(question_deltas) > 0)),
            "video_cluster_bootstrap_95_ci": common.bootstrap_ci(
                grouped,
                bootstrap_samples,
                seed + comparison_index,
            ),
        }

    nll_matrix = np.asarray(
        [[row["conditions"][condition]["nll"] for condition in CONDITIONS] for row in rows]
    )
    winner_indices = np.argmin(nll_matrix, axis=1)
    summary["question_level_best_alpha_fraction"] = {
        condition: float(np.mean(winner_indices == index))
        for index, condition in enumerate(CONDITIONS)
    }
    summary["best_mean_nll_condition"] = min(
        CONDITIONS,
        key=lambda condition: summary["conditions"][condition]["mean_nll"],
    )
    common.atomic_json_dump(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    rank, world_size, device = common.distributed_context()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_excel(args.annotation_xlsx)
    required_columns = {"index", "video", "question", "candidates", "answer"}
    if not required_columns.issubset(annotations.columns):
        raise ValueError(f"Missing columns: {required_columns - set(annotations.columns)}")
    grouped = {
        video: frame.copy() for video, frame in annotations.groupby("video", sort=True)
    }
    videos = sorted(grouped)
    rng = random.Random(args.seed)
    rng.shuffle(videos)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    for video in videos:
        grouped[video] = (
            grouped[video].sort_values("index").head(args.max_questions_per_video)
        )
    selected_questions = sum(len(grouped[video]) for video in videos)
    protocol = {
        "model_path": str(args.model_path.resolve()),
        "annotation_xlsx": str(args.annotation_xlsx.resolve()),
        "frame_root": str(args.frame_root.resolve()),
        "video_root": str(args.video_root.resolve()),
        "seed": args.seed,
        "selected_videos": len(videos),
        "selected_questions": selected_questions,
        "world_size": world_size,
        "alphas": list(ALPHAS),
        "intervention_point": "projected visual tokens inserted into the language model",
        "definition": "h(alpha) = h_base + alpha * (h_v4 - h_base)",
        "h_base": "v4 checkpoint with the complete FW memory scan bypassed",
        "h_v4": "normal sequential v4 checkpoint",
        "nll": "mean teacher-forced NLL over correct answer-letter continuation tokens",
        "bootstrap": f"{args.bootstrap_samples} resamples clustered by video",
    }
    if rank == 0:
        common.atomic_json_dump(protocol, args.output_dir / "protocol.json")

    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map={"": str(device)},
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).eval()

    output_path = args.output_dir / f"rank-{rank:02d}.jsonl"
    completed = common.load_completed_keys(output_path)
    assigned_videos = videos[rank::world_size]
    for video_number, video in enumerate(assigned_videos, start=1):
        rows = grouped[video]
        missing_rows = [
            row
            for _, row in rows.iterrows()
            if (video, int(row["index"])) not in completed
        ]
        if not missing_rows:
            continue
        frame_dir = args.frame_root / video
        frame_paths = sorted(frame_dir.glob("*.jpg"), key=common.frame_index)
        if not frame_paths:
            raise FileNotFoundError(f"No cached frames for {video}: {frame_dir}")
        media_path = args.video_root / f"{video}.mp4"
        duration = common.video_duration(media_path)
        sample_fps = len(frame_paths) / duration
        video_item = common.video_content(
            frame_paths,
            sample_fps,
            args.min_pixels,
            args.max_pixels,
            args.total_pixels,
        )
        vision_messages = common.build_messages(missing_rows[0], video_item)
        _, videos_and_metadata, video_kwargs = process_vision_info(
            vision_messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_tensor, metadata = videos_and_metadata[0]
        if isinstance(metadata, dict):
            video_module = importlib.import_module(
                processor.video_processor.__class__.__module__
            )
            metadata = video_module.VideoChat3VideoMetadata(**metadata)
        video_inputs = processor.video_processor(
            videos=[video_tensor],
            video_metadata=[metadata],
            do_resize=False,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        pixel_values = video_inputs["pixel_values_videos"].to(
            device=device,
            dtype=model.dtype,
        )
        grid_thw = video_inputs["video_grid_thw"].to(device)
        h_base = common.projected_video_features(
            model,
            pixel_values,
            grid_thw,
            "gate_zero",
        )
        h_v4 = common.projected_video_features(
            model,
            pixel_values,
            grid_thw,
            "sequential",
        )
        features = {
            alpha_label(alpha): interpolate_fw_residual(h_base, h_v4, alpha)
            for alpha in ALPHAS
        }
        if features[alpha_label(0.0)] is not h_base:
            raise AssertionError("alpha=0 must reuse h_base exactly")
        if features[alpha_label(1.0)] is not h_v4:
            raise AssertionError("alpha=1 must reuse h_v4 exactly")
        metrics = {
            condition: residual_metrics(h_base, h_v4, values)
            for condition, values in features.items()
        }
        for row in missing_rows:
            answer = str(row["answer"]).strip()
            messages = common.build_messages(row, video_item)
            tokenized, prompt_length, answer_ids = common.tokenize_question(
                processor,
                messages,
                answer,
                video_inputs["video_grid_thw"][0],
                metadata,
            )
            condition_results = {
                condition: common.answer_nll(
                    model,
                    tokenized,
                    prompt_length,
                    answer_ids,
                    features[condition],
                    device,
                )
                for condition in CONDITIONS
            }
            payload = {
                "index": int(row["index"]),
                "video": video,
                "answer": answer,
                "frames": len(frame_paths),
                "grid_thw": [int(value) for value in grid_thw[0].tolist()],
                "feature_metrics": metrics,
                "conditions": condition_results,
            }
            common.append_jsonl(output_path, payload)
            completed.add((video, int(row["index"])))
        print(
            f"[rank {rank}] video {video_number}/{len(assigned_videos)} "
            f"{video}: {len(missing_rows)} questions",
            flush=True,
        )
        del features, h_base, h_v4, pixel_values, grid_thw, video_inputs
        torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        summary = summarize(
            args.output_dir,
            args.bootstrap_samples,
            args.seed,
            protocol,
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
