#!/usr/bin/env python3
"""Teacher-forced Base VideoChat3 macro temporal compression sweep."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from eval_videochat3_lact_teacher_forced_ablation import (
    answer_nll,
    append_jsonl,
    atomic_json_dump,
    bootstrap_ci,
    distributed_context,
    frame_index,
    load_completed_keys,
    projected_video_features,
    tokenize_question,
    video_content,
    video_duration,
)
from xtuner.v1.model.compose.videochat3.macro_temporal import (
    compress_chunk_outputs,
    compress_timestamps,
    macro_video_token_count,
    video_clip_counts,
)


FACTORS = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--annotation-xlsx", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-format",
        choices=("videomme", "mvbench"),
        default="videomme",
    )
    parser.add_argument("--nframes", type=int, default=64)
    parser.add_argument("--max-videos", type=int, default=96)
    parser.add_argument("--max-questions-per-video", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--min-pixels", type=int, default=28 * 28)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--total-pixels", type=int, default=80_000 * 2 * 4 * 14 * 14)
    return parser.parse_args()


def build_messages(row: pd.Series, video_item: dict, dataset_format: str) -> list[dict]:
    candidates = ast.literal_eval(row["candidates"])
    if dataset_format == "mvbench":
        candidates = [f"{chr(ord('A') + index)}. {candidate}" for index, candidate in enumerate(candidates)]
    question = str(row["question"]) + "\n" + "\n".join(candidates)
    prompt = f"Question: {question}\nAnswer: "
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ""},
                video_item,
                {
                    "type": "text",
                    "text": (
                        "\nThese are the frames of a video. Select the best answer to the following "
                        "multiple-choice question based on the video. Respond with only the letter "
                        "(A, B, C, or D) of the correct option.\n"
                    ),
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def correct_answer(row: pd.Series, dataset_format: str) -> str:
    answer = str(row["answer"]).strip()
    if dataset_format == "videomme":
        return answer
    candidates = ast.literal_eval(row["candidates"])
    try:
        answer_index = candidates.index(answer)
    except ValueError as error:
        raise ValueError(f"MVBench answer {answer!r} is absent from candidates={candidates}") from error
    return chr(ord("A") + answer_index)


def expand_macro_video_placeholder(
    processor,
    text: str,
    grid_thw: torch.Tensor,
    metadata,
    factor: int,
) -> str:
    timestamps = processor._calculate_timestamps(
        metadata,
        processor.video_processor.temporal_merge_size,
    )
    timestamps = compress_timestamps(timestamps, factor)
    merge_length = processor.video_processor.merge_size**2
    frame_sequence_length = int(grid_thw[1:].prod().item() // merge_length)
    placeholder = ""
    for timestamp in timestamps:
        placeholder += f"<{timestamp:.1f} seconds>"
        placeholder += (
            processor.vision_start_token
            + "<|placeholder|>" * frame_sequence_length
            + processor.vision_end_token
        )
    wrapped_video_token = (
        processor.vision_start_token
        + processor.video_token
        + processor.vision_end_token
    )
    if wrapped_video_token not in text:
        raise ValueError("Chat template does not contain the wrapped video token")
    return text.replace(wrapped_video_token, placeholder, 1).replace(
        "<|placeholder|>", processor.video_token
    )


def tokenize_macro_question(
    processor,
    messages: list[dict],
    answer: str,
    grid_thw: torch.Tensor,
    metadata,
    factor: int,
):
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    expanded_prompt = expand_macro_video_placeholder(
        processor,
        prompt,
        grid_thw,
        metadata,
        factor,
    )
    expanded_full = expand_macro_video_placeholder(
        processor,
        prompt + answer,
        grid_thw,
        metadata,
        factor,
    )
    prompt_tokens = processor.tokenizer(expanded_prompt, return_tensors="pt")
    full_tokens = processor.tokenizer(expanded_full, return_tensors="pt")
    prompt_ids = prompt_tokens["input_ids"][0]
    full_ids = full_tokens["input_ids"][0]
    if full_ids.shape[0] <= prompt_ids.shape[0]:
        raise ValueError(f"Answer produced no continuation tokens: {answer!r}")
    if not torch.equal(full_ids[: prompt_ids.shape[0]], prompt_ids):
        raise ValueError(f"Answer changed prompt tokenization boundary: {answer!r}")
    answer_ids = full_ids[prompt_ids.shape[0] :]
    decoded = processor.tokenizer.decode(
        answer_ids,
        clean_up_tokenization_spaces=False,
    )
    if decoded != answer:
        raise ValueError(
            f"Answer suffix mismatch: expected={answer!r}, decoded={decoded!r}"
        )
    return full_tokens, prompt_ids.shape[0], answer_ids


@torch.inference_mode()
def projected_macro_features(model, pixel_values, grid_thw):
    chunk_outputs = model.model.vision_tower(
        pixel_values=pixel_values,
        grid_thws=grid_thw,
    )
    counts = video_clip_counts(
        grid_thw,
        model.config.vision_config.temporal_merge_size,
    )
    features = {}
    for factor in FACTORS:
        compressed = compress_chunk_outputs(
            chunk_outputs,
            counts,
            factor,
            mode="mean",
        )
        features[factor] = model.model.multi_modal_projector(compressed)
    return chunk_outputs, features


def _clustered_deltas(rows, factor: int, field: str) -> tuple[list[float], dict[str, float]]:
    question_deltas = [
        row["conditions"][f"r{factor}"][field]
        - row["conditions"]["r1"][field]
        for row in rows
    ]
    grouped = {}
    for video in sorted({row["video"] for row in rows}):
        grouped[video] = float(
            np.mean(
                [
                    row["conditions"][f"r{factor}"][field]
                    - row["conditions"]["r1"][field]
                    for row in rows
                    if row["video"] == video
                ]
            )
        )
    return question_deltas, grouped


def summarize(output_dir: Path, bootstrap_samples: int, seed: int, protocol: dict):
    rows = []
    for path in sorted(output_dir.glob("rank-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    expected = protocol["selected_questions"]
    unique = {(row["video"], row["index"]) for row in rows}
    if len(rows) != expected or len(unique) != expected:
        raise RuntimeError(
            f"Expected {expected} unique rows, got rows={len(rows)}, unique={len(unique)}"
        )

    summary = {
        "protocol": protocol,
        "parity": {},
        "conditions": {},
        "paired_deltas_vs_r1": {},
    }
    parity_nll = np.asarray(
        [row["parity"]["macro_r1_nll_minus_legacy_nll"] for row in rows],
        dtype=np.float64,
    )
    parity_probability = np.asarray(
        [
            row["parity"]["macro_r1_probability_minus_legacy_probability"]
            for row in rows
        ],
        dtype=np.float64,
    )
    summary["parity"] = {
        "all_projected_features_bitwise_equal": all(
            row["parity"]["projected_features_bitwise_equal"] for row in rows
        ),
        "all_input_ids_equal": all(row["parity"]["input_ids_equal"] for row in rows),
        "max_abs_nll_delta": float(np.abs(parity_nll).max()),
        "max_abs_answer_probability_delta": float(np.abs(parity_probability).max()),
    }

    for factor in FACTORS:
        key = f"r{factor}"
        nll = np.asarray([row["conditions"][key]["nll"] for row in rows])
        probability = np.asarray(
            [row["conditions"][key]["answer_probability"] for row in rows]
        )
        visual_tokens = np.asarray(
            [row["conditions"][key]["video_tokens"] for row in rows],
            dtype=np.float64,
        )
        r1_tokens = np.asarray(
            [row["conditions"]["r1"]["video_tokens"] for row in rows],
            dtype=np.float64,
        )
        summary["conditions"][key] = {
            "mean_nll": float(nll.mean()),
            "median_nll": float(np.median(nll)),
            "mean_answer_probability": float(probability.mean()),
            "median_answer_probability": float(np.median(probability)),
            "mean_visual_tokens": float(visual_tokens.mean()),
            "median_visual_tokens": float(np.median(visual_tokens)),
            "aggregate_actual_compression_ratio": float(
                r1_tokens.sum() / visual_tokens.sum()
            ),
            "mean_per_video_actual_compression_ratio": float(
                np.mean(r1_tokens / visual_tokens)
            ),
        }
        if factor == 1:
            continue
        nll_deltas, nll_by_video = _clustered_deltas(rows, factor, "nll")
        probability_deltas, probability_by_video = _clustered_deltas(
            rows,
            factor,
            "answer_probability",
        )
        summary["paired_deltas_vs_r1"][key] = {
            "mean_nll_delta": float(np.mean(nll_deltas)),
            "median_nll_delta": float(np.median(nll_deltas)),
            "nll_video_cluster_bootstrap_95_ci": bootstrap_ci(
                nll_by_video,
                bootstrap_samples,
                seed + factor,
            ),
            "mean_answer_probability_delta": float(np.mean(probability_deltas)),
            "median_answer_probability_delta": float(
                np.median(probability_deltas)
            ),
            "answer_probability_video_cluster_bootstrap_95_ci": bootstrap_ci(
                probability_by_video,
                bootstrap_samples,
                seed + 100 + factor,
            ),
        }
    atomic_json_dump(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    rank, world_size, device = distributed_context()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_excel(args.annotation_xlsx)
    required_columns = {"index", "video", "question", "candidates", "answer"}
    if args.dataset_format == "mvbench":
        required_columns.add("prefix")
    if not required_columns.issubset(annotations.columns):
        raise ValueError(f"Missing columns: {required_columns - set(annotations.columns)}")
    if args.dataset_format == "mvbench":
        annotations = annotations.copy()
        annotations["video_key"] = annotations["prefix"].str.rstrip("/") + "/" + annotations["video"]
        group_column = "video_key"
    else:
        group_column = "video"
    grouped = {
        video: frame.copy()
        for video, frame in annotations.groupby(group_column, sort=True)
    }
    videos = sorted(grouped)
    rng = random.Random(args.seed)
    rng.shuffle(videos)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    for video in videos:
        grouped[video] = grouped[video].sort_values("index").head(
            args.max_questions_per_video
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
        "dataset_format": args.dataset_format,
        "nframes": args.nframes if args.dataset_format == "mvbench" else None,
        "factors": list(FACTORS),
        "base_compression": (
            "mean identical spatial positions over R consecutive post-final-layernorm, "
            "post-four-frame-patch-merger chunk outputs"
        ),
        "tail": "mean over the actual remaining chunks within each video",
        "timestamp": "mean of the original four-frame chunk timestamps in each macro group",
        "nll": "mean teacher-forced NLL over correct answer-letter continuation tokens",
        "bootstrap": f"{args.bootstrap_samples} resamples clustered by video",
        "placeholder_check": "asserted for legacy and every compression factor on every question",
    }
    if rank == 0:
        atomic_json_dump(protocol, args.output_dir / "protocol.json")

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
    completed = load_completed_keys(output_path)
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
        if args.dataset_format == "videomme":
            frame_dir = args.frame_root / video
            frame_paths = sorted(frame_dir.glob("*.jpg"), key=frame_index)
            if not frame_paths:
                raise FileNotFoundError(f"No cached frames for {video}: {frame_dir}")
            media_path = args.video_root / f"{video}.mp4"
            duration = video_duration(media_path)
            sample_fps = len(frame_paths) / duration
            video_item = video_content(
                frame_paths,
                sample_fps,
                args.min_pixels,
                args.max_pixels,
                args.total_pixels,
            )
        else:
            media_path = args.video_root / video
            if not media_path.is_file():
                raise FileNotFoundError(f"Missing MVBench media: {media_path}")
            video_item = {
                "type": "video",
                "video": f"file://{media_path}",
                "nframes": args.nframes,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
                "total_pixels": args.total_pixels,
            }
        vision_messages = build_messages(
            missing_rows[0],
            video_item,
            args.dataset_format,
        )
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

        # This invokes the unchanged checkpoint path independently of the new
        # post-encoder implementation and is the R=1 parity reference.
        legacy_features = projected_video_features(
            model,
            pixel_values,
            grid_thw,
            "sequential",
        )
        _, features = projected_macro_features(model, pixel_values, grid_thw)
        feature_parity = torch.equal(legacy_features, features[1])
        if not feature_parity:
            max_difference = (legacy_features.float() - features[1].float()).abs().max()
            raise RuntimeError(f"Base R=1 feature parity failed: max_abs_diff={max_difference.item()}")

        grid_cpu = video_inputs["video_grid_thw"][0]
        expected_tokens = {}
        for factor in FACTORS:
            expected_tokens[factor] = macro_video_token_count(
                [int(value) for value in grid_cpu.tolist()],
                temporal_merge_size=processor.video_processor.temporal_merge_size,
                spatial_merge_size=processor.video_processor.merge_size,
                factor=factor,
            )
            if features[factor].shape[0] != expected_tokens[factor]:
                raise RuntimeError(
                    f"R={factor} projected token mismatch: "
                    f"projected={features[factor].shape[0]}, expected={expected_tokens[factor]}"
                )

        for row in missing_rows:
            answer = correct_answer(row, args.dataset_format)
            messages = build_messages(row, video_item, args.dataset_format)
            legacy_tokenized, legacy_prompt_length, legacy_answer_ids = tokenize_question(
                processor,
                messages,
                answer,
                grid_cpu,
                metadata,
            )
            legacy_result = answer_nll(
                model,
                legacy_tokenized,
                legacy_prompt_length,
                legacy_answer_ids,
                legacy_features,
                device,
            )
            condition_results = {}
            r1_input_ids_equal = False
            for factor in FACTORS:
                tokenized, prompt_length, answer_ids = tokenize_macro_question(
                    processor,
                    messages,
                    answer,
                    grid_cpu,
                    metadata,
                    factor,
                )
                placeholders = int(
                    (tokenized["input_ids"] == model.config.video_token_id).sum()
                )
                if placeholders != expected_tokens[factor]:
                    raise RuntimeError(
                        f"R={factor} placeholder mismatch: placeholders={placeholders}, "
                        f"projected={expected_tokens[factor]}"
                    )
                if factor == 1:
                    r1_input_ids_equal = torch.equal(
                        legacy_tokenized["input_ids"],
                        tokenized["input_ids"],
                    )
                    if not r1_input_ids_equal:
                        raise RuntimeError("Base R=1 prompt/token parity failed")
                condition_results[f"r{factor}"] = answer_nll(
                    model,
                    tokenized,
                    prompt_length,
                    answer_ids,
                    features[factor],
                    device,
                )
            r1_result = condition_results["r1"]
            payload = {
                "index": int(row["index"]),
                "video": video,
                "answer": answer,
                "frames": int(grid_thw[0, 0].item()),
                "grid_thw": [int(value) for value in grid_thw[0].tolist()],
                "parity": {
                    "projected_features_bitwise_equal": feature_parity,
                    "input_ids_equal": r1_input_ids_equal,
                    "macro_r1_nll_minus_legacy_nll": r1_result["nll"]
                    - legacy_result["nll"],
                    "macro_r1_probability_minus_legacy_probability": r1_result[
                        "answer_probability"
                    ]
                    - legacy_result["answer_probability"],
                },
                "conditions": condition_results,
            }
            append_jsonl(output_path, payload)
            completed.add((video, int(row["index"])))
        print(
            f"[rank {rank}] video {video_number}/{len(assigned_videos)} "
            f"{video}: {len(missing_rows)} questions, "
            f"tokens={{{', '.join(f'R{factor}: {features[factor].shape[0]}' for factor in FACTORS)}}}",
            flush=True,
        )
        del legacy_features, features, pixel_values, grid_thw, video_inputs
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
