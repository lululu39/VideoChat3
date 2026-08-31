#!/usr/bin/env python3
"""Teacher-forced Base-model sensitivity to LM-facing visual-token changes."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
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

import eval_videochat3_lact_teacher_forced_ablation as common


CONDITIONS = ("base", "random_r015", "sinusoidal_r015")
COMPARISONS = {
    "random_minus_base": ("random_r015", "base"),
    "sinusoidal_minus_base": ("sinusoidal_r015", "base"),
    "sinusoidal_minus_random": ("sinusoidal_r015", "random_r015"),
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
    parser.add_argument("--relative-l2", type=float, default=0.15)
    parser.add_argument("--frame-group-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--min-pixels", type=int, default=28 * 28)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--total-pixels", type=int, default=80_000 * 2 * 4 * 14 * 14)
    return parser.parse_args()


def chunk_slices(rows: int, chunk_rows: int = 1024):
    for start in range(0, rows, chunk_rows):
        yield slice(start, min(start + chunk_rows, rows))


def squared_l2(tensor: torch.Tensor) -> float:
    total = 0.0
    for section in chunk_slices(tensor.shape[0]):
        values = tensor[section].float()
        total += values.square().sum().item()
    return total


def feature_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Feature shape mismatch: {reference.shape} vs {candidate.shape}")
    reference_sq = 0.0
    candidate_sq = 0.0
    delta_sq = 0.0
    dot = 0.0
    for section in chunk_slices(reference.shape[0]):
        left = reference[section].float()
        right = candidate[section].float()
        delta = right - left
        reference_sq += left.square().sum().item()
        candidate_sq += right.square().sum().item()
        delta_sq += delta.square().sum().item()
        dot += (left * right).sum().item()
    if reference_sq <= 0 or candidate_sq <= 0:
        raise ValueError("Visual features must have nonzero norm")
    return {
        "relative_l2_delta": math.sqrt(delta_sq / reference_sq),
        "cosine_similarity": dot / math.sqrt(reference_sq * candidate_sq),
        "candidate_to_reference_norm": math.sqrt(candidate_sq / reference_sq),
    }


def random_perturbation(
    reference: torch.Tensor,
    *,
    relative_l2: float,
    seed: int,
) -> torch.Tensor:
    if relative_l2 < 0:
        raise ValueError("relative_l2 must be nonnegative")
    if relative_l2 == 0:
        return reference.clone()
    reference_norm = math.sqrt(squared_l2(reference))
    generator = torch.Generator(device=reference.device).manual_seed(seed)
    noise_sq = 0.0
    for section in chunk_slices(reference.shape[0]):
        noise = torch.randn(
            reference[section].shape,
            generator=generator,
            device=reference.device,
            dtype=torch.float32,
        )
        noise_sq += noise.square().sum().item()
    scale = relative_l2 * reference_norm / math.sqrt(noise_sq)

    generator.manual_seed(seed)
    candidate = torch.empty_like(reference)
    for section in chunk_slices(reference.shape[0]):
        noise = torch.randn(
            reference[section].shape,
            generator=generator,
            device=reference.device,
            dtype=torch.float32,
        )
        candidate[section] = (
            reference[section].float() + noise * scale
        ).to(reference.dtype)
    return candidate


def sinusoidal_group_encoding(
    groups: int,
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if groups < 2:
        raise ValueError("Centered temporal encoding requires at least two groups")
    if width < 1:
        raise ValueError("Feature width must be positive")
    positions = torch.arange(groups, device=device, dtype=torch.float32).unsqueeze(1)
    even_dimensions = torch.arange(0, width, 2, device=device, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10_000.0) * even_dimensions / width)
    encoding = torch.empty((groups, width), device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    if width > 1:
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    # Remove the feature-wise constant component so the intervention carries
    # temporal variation rather than a shared bias on every visual token.
    return encoding - encoding.mean(dim=0, keepdim=True)


def sinusoidal_temporal_perturbation(
    reference: torch.Tensor,
    *,
    grid_time: int,
    frame_group_size: int,
    relative_l2: float,
) -> tuple[torch.Tensor, int, int]:
    if frame_group_size <= 0:
        raise ValueError("frame_group_size must be positive")
    if relative_l2 < 0:
        raise ValueError("relative_l2 must be nonnegative")
    groups = math.ceil(grid_time / frame_group_size)
    if reference.shape[0] % groups:
        raise ValueError(
            f"Projected tokens {reference.shape[0]} are not divisible by {groups} temporal groups"
        )
    tokens_per_group = reference.shape[0] // groups
    encoding = sinusoidal_group_encoding(
        groups,
        reference.shape[1],
        device=reference.device,
    )
    reference_norm = math.sqrt(squared_l2(reference))
    raw_norm = math.sqrt(encoding.square().sum().item() * tokens_per_group)
    scale = relative_l2 * reference_norm / raw_norm if relative_l2 else 0.0
    candidate = torch.empty_like(reference)
    for group in range(groups):
        start = group * tokens_per_group
        end = start + tokens_per_group
        candidate[start:end] = (
            reference[start:end].float() + encoding[group] * scale
        ).to(reference.dtype)
    return candidate, groups, tokens_per_group


def summarize(output_dir: Path, bootstrap_samples: int, seed: int, protocol: dict):
    rows = []
    for path in sorted(output_dir.glob("rank-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not rows:
        raise RuntimeError("No perturbation rows were produced")
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
            "mean_nll_delta": float(np.mean(question_deltas)),
            "median_nll_delta": float(np.median(question_deltas)),
            "positive_fraction": float(np.mean(np.asarray(question_deltas) > 0)),
            "video_cluster_bootstrap_95_ci": common.bootstrap_ci(
                grouped,
                bootstrap_samples,
                seed + comparison_index,
            ),
        }
    common.atomic_json_dump(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    if not 0 <= args.relative_l2 <= 1:
        raise ValueError("relative_l2 must be between zero and one")
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
        "intervention_point": "projected visual tokens inserted into the language model",
        "target_relative_l2": args.relative_l2,
        "frame_group_size": args.frame_group_size,
        "conditions": {
            "base": "unaltered Base-model projected visual tokens",
            "random_r015": (
                "deterministic per-video iid Gaussian additive perturbation, globally "
                "scaled to the target relative L2"
            ),
            "sinusoidal_r015": (
                "one centered standard sinusoidal vector per non-overlapping four-frame "
                "group, shared by that group's projected tokens and globally scaled to "
                "the target relative L2"
            ),
        },
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
        base_features = common.projected_video_features(
            model,
            pixel_values,
            grid_thw,
            "sequential",
        )
        random_features = random_perturbation(
            base_features,
            relative_l2=args.relative_l2,
            seed=common.deterministic_video_seed(args.seed, video),
        )
        sinusoidal_features, temporal_groups, tokens_per_group = (
            sinusoidal_temporal_perturbation(
                base_features,
                grid_time=int(grid_thw[0, 0]),
                frame_group_size=args.frame_group_size,
                relative_l2=args.relative_l2,
            )
        )
        features = {
            "base": base_features,
            "random_r015": random_features,
            "sinusoidal_r015": sinusoidal_features,
        }
        metrics = {
            condition: feature_metrics(base_features, values)
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
                "temporal_groups": temporal_groups,
                "tokens_per_group": tokens_per_group,
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
        del features, base_features, random_features, sinusoidal_features
        del pixel_values, grid_thw, video_inputs
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
