#!/usr/bin/env python3
"""Teacher-forced Video-MME NLL ablations for a VideoChat3-LACT checkpoint."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F


CONDITIONS = ("sequential", "shuffle_chunks", "reset_state", "gate_zero")
FRAMES_TEMPLATE = """
These are the frames of a video. Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
"""


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


def distributed_context() -> tuple[int, int, torch.device]:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    return rank, world_size, device


def atomic_json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def frame_index(path: Path) -> int:
    fields = path.stem.split("-")
    if len(fields) < 4 or fields[0] != "frame":
        raise ValueError(f"Unexpected cached frame name: {path}")
    return int(fields[1])


def video_duration(path: Path) -> float:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        text=True,
    ).strip()
    duration = float(output)
    if duration <= 0:
        raise ValueError(f"Invalid duration for {path}: {duration}")
    return duration


def video_content(
    frame_paths: list[Path],
    sample_fps: float,
    min_pixels: int,
    max_pixels: int,
    total_pixels: int,
) -> dict:
    return {
        "type": "video",
        "video": [f"file://{path}" for path in frame_paths],
        "sample_fps": sample_fps,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "total_pixels": total_pixels,
    }


def build_messages(row: pd.Series, video_item: dict) -> list[dict]:
    candidates = ast.literal_eval(row["candidates"])
    question = str(row["question"]) + "\n" + "\n".join(candidates)
    prompt = f"Question: {question}\nAnswer: "
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ""},
                video_item,
                {"type": "text", "text": FRAMES_TEMPLATE},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def expand_video_placeholder(processor, text: str, grid_thw: torch.Tensor, metadata) -> str:
    merge_length = processor.video_processor.merge_size**2
    timestamps = processor._calculate_timestamps(
        metadata,
        processor.video_processor.temporal_merge_size,
    )
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
    text = text.replace(wrapped_video_token, placeholder, 1)
    return text.replace("<|placeholder|>", processor.video_token)


def tokenize_question(processor, messages: list[dict], answer: str, grid_thw, metadata):
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    expanded_prompt = expand_video_placeholder(processor, prompt, grid_thw, metadata)
    expanded_full = expand_video_placeholder(
        processor,
        prompt + answer,
        grid_thw,
        metadata,
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
    decoded = processor.tokenizer.decode(answer_ids, clean_up_tokenization_spaces=False)
    if decoded != answer:
        raise ValueError(f"Answer suffix mismatch: expected={answer!r}, decoded={decoded!r}")
    return full_tokens, prompt_ids.shape[0], answer_ids


def shuffle_four_frame_chunks(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, list[int]]:
    if grid_thw.shape != (1, 3):
        raise ValueError(f"Expected one video grid, got {tuple(grid_thw.shape)}")
    time, height, width = (int(value) for value in grid_thw[0].tolist())
    spatial_tokens = height * width
    if pixel_values.shape[0] != time * spatial_tokens:
        raise ValueError("Pixel tensor and video grid disagree")
    chunks = []
    frames = pixel_values.view(time, spatial_tokens, pixel_values.shape[-1])
    for start in range(0, time, 4):
        chunks.append(frames[start : start + 4])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(chunks), generator=generator).tolist()
    shuffled = torch.cat([chunks[index] for index in order], dim=0)
    return shuffled.reshape_as(pixel_values), order


def reset_state_scan(self, hidden_states, clip_slices, video_clip_counts):
    outputs = []
    clip_index = 0
    for clip_count in video_clip_counts:
        for _ in range(clip_count):
            fast_weights, _ = self.init_fast_weights(batch_size=1)
            start, end = clip_slices[clip_index]
            clip_hidden = hidden_states[start:end].unsqueeze(0)
            clip_hidden, _, _, _ = self.apply_memory(clip_hidden, fast_weights)
            outputs.append(clip_hidden.squeeze(0))
            clip_index += 1
    if clip_index != len(clip_slices):
        raise ValueError("Reset scan did not consume every clip")
    return torch.cat(outputs, dim=0)


def gate_zero_scan(self, hidden_states, clip_slices, video_clip_counts):
    if sum(video_clip_counts) != len(clip_slices):
        raise ValueError("Gate-zero layout mismatch")
    return hidden_states


@contextmanager
def memory_scan_mode(vision_tower, mode: str):
    if mode == "sequential":
        yield
        return
    replacement = {
        "reset_state": reset_state_scan,
        "gate_zero": gate_zero_scan,
    }[mode]
    originals = []
    for block in vision_tower.encoder.blocks:
        originals.append(block.forward_memory_scan)
        block.forward_memory_scan = types.MethodType(replacement, block)
    try:
        yield
    finally:
        for block, original in zip(vision_tower.encoder.blocks, originals):
            block.forward_memory_scan = original


@torch.inference_mode()
def projected_video_features(model, pixel_values, grid_thw, mode: str):
    with memory_scan_mode(model.model.vision_tower, mode):
        vision = model.model.vision_tower(
            pixel_values=pixel_values,
            grid_thws=grid_thw,
        )
        return model.model.multi_modal_projector(vision)


@torch.inference_mode()
def answer_nll(model, tokenized, prompt_length: int, answer_ids, video_features, device):
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    embeddings = model.model.get_input_embeddings()(input_ids)
    video_mask = input_ids == model.config.video_token_id
    if int(video_mask.sum()) != video_features.shape[0]:
        raise ValueError(
            f"Video token mismatch: placeholders={int(video_mask.sum())}, "
            f"features={video_features.shape[0]}"
        )
    embeddings[video_mask] = video_features.to(embeddings.dtype)
    outputs = model.model.language_model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        use_cache=False,
    )
    hidden = outputs[0]
    targets = answer_ids.to(device)
    target_positions = torch.arange(
        prompt_length,
        prompt_length + targets.shape[0],
        device=device,
    )
    logits = model.lm_head(hidden[:, target_positions - 1, :]).float().squeeze(0)
    token_nll = F.cross_entropy(logits, targets, reduction="none")
    return {
        "nll": token_nll.mean().item(),
        "nll_sum": token_nll.sum().item(),
        "answer_tokens": targets.shape[0],
        "answer_probability": math.exp(-token_nll.sum().item()),
        "sequence_tokens": input_ids.shape[1],
        "video_tokens": video_features.shape[0],
    }


def deterministic_video_seed(global_seed: int, video: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{video}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def load_completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.is_file():
        return set()
    completed = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        completed.add((row["video"], int(row["index"])))
    return completed


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def bootstrap_ci(values_by_video: dict[str, float], samples: int, seed: int):
    keys = sorted(values_by_video)
    values = np.asarray([values_by_video[key] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(output_dir: Path, bootstrap_samples: int, seed: int, protocol: dict):
    rows = []
    for path in sorted(output_dir.glob("rank-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not rows:
        raise RuntimeError("No ablation rows were produced")
    expected = protocol["selected_questions"]
    unique = {(row["video"], row["index"]) for row in rows}
    if len(rows) != expected or len(unique) != expected:
        raise RuntimeError(
            f"Expected {expected} unique rows, got rows={len(rows)}, unique={len(unique)}"
        )
    summary = {"protocol": protocol, "conditions": {}, "deltas": {}}
    for condition in CONDITIONS:
        values = [row["conditions"][condition]["nll"] for row in rows]
        summary["conditions"][condition] = {
            "mean_nll": float(np.mean(values)),
            "median_nll": float(np.median(values)),
            "mean_answer_probability": float(
                np.mean([row["conditions"][condition]["answer_probability"] for row in rows])
            ),
        }
    comparisons = {
        "shuffle_minus_sequential": ("shuffle_chunks", "sequential"),
        "reset_minus_sequential": ("reset_state", "sequential"),
        "gate_zero_minus_sequential": ("gate_zero", "sequential"),
    }
    for name, (left, right) in comparisons.items():
        question_deltas = [
            row["conditions"][left]["nll"] - row["conditions"][right]["nll"]
            for row in rows
        ]
        grouped = {}
        for video in sorted({row["video"] for row in rows}):
            grouped[video] = float(
                np.mean(
                    [
                        row["conditions"][left]["nll"] - row["conditions"][right]["nll"]
                        for row in rows
                        if row["video"] == video
                    ]
                )
            )
        summary["deltas"][name] = {
            "mean_nll_delta": float(np.mean(question_deltas)),
            "median_nll_delta": float(np.median(question_deltas)),
            "positive_fraction": float(np.mean(np.asarray(question_deltas) > 0)),
            "video_cluster_bootstrap_95_ci": bootstrap_ci(
                grouped,
                bootstrap_samples,
                seed + len(summary["deltas"]),
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
    if not required_columns.issubset(annotations.columns):
        raise ValueError(f"Missing columns: {required_columns - set(annotations.columns)}")
    grouped = {video: frame.copy() for video, frame in annotations.groupby("video", sort=True)}
    videos = sorted(grouped)
    rng = random.Random(args.seed)
    rng.shuffle(videos)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    for video in videos:
        grouped[video] = grouped[video].sort_values("index").head(args.max_questions_per_video)
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
        "conditions": {
            "sequential": "normal cached frames and normal recurrent FW scan",
            "shuffle_chunks": "deterministically shuffle non-overlapping four-frame chunks; timestamps stay ordered",
            "reset_state": "normal frames; reinitialize FW state independently for every four-frame chunk",
            "gate_zero": "normal frames; bypass the entire FW memory scan, algebraically equivalent to zero gates",
        },
        "nll": "mean teacher-forced NLL over correct answer-letter continuation tokens",
        "bootstrap": f"{args.bootstrap_samples} resamples clustered by video",
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
            row for _, row in rows.iterrows() if (video, int(row["index"])) not in completed
        ]
        if not missing_rows:
            continue
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
        vision_messages = build_messages(missing_rows[0], video_item)
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
        shuffled_pixels, shuffle_order = shuffle_four_frame_chunks(
            pixel_values,
            grid_thw,
            seed=deterministic_video_seed(args.seed, video),
        )
        features = {
            "sequential": projected_video_features(
                model, pixel_values, grid_thw, "sequential"
            ),
            "shuffle_chunks": projected_video_features(
                model, shuffled_pixels, grid_thw, "sequential"
            ),
            "reset_state": projected_video_features(
                model, pixel_values, grid_thw, "reset_state"
            ),
            "gate_zero": projected_video_features(
                model, pixel_values, grid_thw, "gate_zero"
            ),
        }
        for row in missing_rows:
            answer = str(row["answer"]).strip()
            messages = build_messages(row, video_item)
            tokenized, prompt_length, answer_ids = tokenize_question(
                processor,
                messages,
                answer,
                video_inputs["video_grid_thw"][0],
                metadata,
            )
            condition_results = {
                condition: answer_nll(
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
                "shuffle_order_sha256": hashlib.sha256(
                    json.dumps(shuffle_order).encode()
                ).hexdigest(),
                "conditions": condition_results,
            }
            append_jsonl(output_path, payload)
            completed.add((video, int(row["index"])))
        print(
            f"[rank {rank}] video {video_number}/{len(assigned_videos)} "
            f"{video}: {len(missing_rows)} questions",
            flush=True,
        )
        del features, pixel_values, shuffled_pixels, grid_thw, video_inputs
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
