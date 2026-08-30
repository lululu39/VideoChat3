#!/usr/bin/env python3
"""Compare VideoChat3 visual tokens across Base and LACT checkpoints.

The script has three deliberately separate stages so expensive preprocessing is
performed once and multiple checkpoints can be evaluated in parallel:

1. ``prepare`` converts fixed frame directories into the exact BF16 patch input.
2. ``reference`` records Base vision-tower and projected visual tokens.
3. ``compare`` measures a checkpoint against those Base tensors.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--processor-path", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--sample",
        action="append",
        required=True,
        metavar="NAME=CATEGORY:FRAME_DIR:SAMPLE_FPS",
    )
    prepare.add_argument("--min-pixels", type=int, default=28 * 28)
    prepare.add_argument("--max-pixels", type=int, default=448 * 448)
    prepare.add_argument("--total-pixels", type=int, default=80_000 * 2 * 4 * 14 * 14)

    reference = subparsers.add_parser("reference")
    add_extract_args(reference)
    reference.add_argument("--reference-dir", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    add_extract_args(compare)
    compare.add_argument("--reference-dir", type=Path, required=True)
    compare.add_argument("--output-json", type=Path, required=True)

    return parser.parse_args()


def add_extract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="flash_attention_2")


def atomic_json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    if not value:
        raise ValueError("Sample name became empty after sanitization")
    return value


def frame_index(path: Path) -> int:
    match = re.search(r"frame-(\d+)-of-\d+", path.stem)
    if match is not None:
        return int(match.group(1))
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"Cannot infer frame index from {path}")
    return int(numbers[-1])


def parse_sample(specification: str) -> dict:
    if "=" not in specification:
        raise ValueError(f"Invalid --sample {specification!r}")
    name, payload = specification.split("=", 1)
    category, frame_dir, sample_fps = payload.split(":", 2)
    directory = Path(frame_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    frames = sorted(directory.glob("*.jpg"), key=frame_index)
    if not frames:
        raise FileNotFoundError(f"No JPG frames under {directory}")
    return {
        "name": safe_name(name),
        "category": category,
        "frame_dir": str(directory),
        "sample_fps": float(sample_fps),
        "frames": frames,
    }


def prepare_inputs(args: argparse.Namespace) -> None:
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        args.processor_path, trust_remote_code=True
    )
    samples = []
    for specification in args.sample:
        sample = parse_sample(specification)
        content = [
            {
                "type": "video",
                "video": [f"file://{path}" for path in sample["frames"]],
                "sample_fps": sample["sample_fps"],
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
                "total_pixels": args.total_pixels,
            },
            {"type": "text", "text": "Describe the video."},
        ]
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        videos, video_metadatas = zip(*videos)
        inputs = processor(
            text=text,
            images=images,
            videos=list(videos),
            video_metadata=list(video_metadatas),
            do_resize=False,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        pixel_values = inputs["pixel_values_videos"].to(torch.bfloat16)
        grid_thw = inputs["video_grid_thw"].to(torch.long)
        output_path = args.output_dir / f"{sample['name']}.pt"
        torch.save(
            {
                "pixel_values_videos": pixel_values,
                "video_grid_thw": grid_thw,
            },
            output_path,
        )
        time, height, width = (int(value) for value in grid_thw[0].tolist())
        record = {
            "name": sample["name"],
            "category": sample["category"],
            "frame_dir": sample["frame_dir"],
            "sample_fps": sample["sample_fps"],
            "frames": len(sample["frames"]),
            "grid_thw": [time, height, width],
            "input_path": str(output_path),
            "input_elements": pixel_values.numel(),
        }
        samples.append(record)
        print(json.dumps(record), flush=True)
        del inputs, pixel_values, grid_thw, images, videos, video_metadatas
        gc.collect()
    atomic_json_dump(
        {
            "processor_path": str(args.processor_path.resolve()),
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "total_pixels": args.total_pixels,
            "samples": samples,
        },
        args.output_dir / "manifest.json",
    )


def load_model(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_manifest(input_dir: Path) -> dict:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return json.loads(manifest_path.read_text())


@torch.inference_mode()
def extract_features(model, input_path: Path, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.load(input_path, map_location="cpu", weights_only=True, mmap=True)
    pixel_values = inputs["pixel_values_videos"].to(
        device=device, dtype=model.dtype, non_blocking=True
    )
    grid_thw = inputs["video_grid_thw"].to(device=device, non_blocking=True)
    vision_output = model.model.vision_tower(
        pixel_values=pixel_values, grid_thws=grid_thw
    )
    projected_tokens = model.model.multi_modal_projector(vision_output)
    vision_tokens = (
        torch.cat(vision_output, dim=0)
        if isinstance(vision_output, list)
        else vision_output
    )
    # patch_merger keeps the 2x2 neighborhood as [token, 4, hidden];
    # flatten it to the actual pre-projector visual-token representation.
    if vision_tokens.ndim > 2:
        vision_tokens = vision_tokens.flatten(start_dim=1)
    return vision_tokens, projected_tokens


def save_references(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.input_dir)
    args.reference_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args)
    records = []
    for sample in manifest["samples"]:
        vision, projected = extract_features(
            model, Path(sample["input_path"]), args.device
        )
        output_path = args.reference_dir / f"{sample['name']}.pt"
        torch.save(
            {
                "vision_tokens": vision.detach().to("cpu", dtype=torch.bfloat16),
                "projected_tokens": projected.detach().to(
                    "cpu", dtype=torch.bfloat16
                ),
            },
            output_path,
        )
        record = {
            "name": sample["name"],
            "reference_path": str(output_path),
            "vision_shape": list(vision.shape),
            "projected_shape": list(projected.shape),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        del vision, projected
        torch.cuda.empty_cache()
        gc.collect()
    atomic_json_dump(
        {
            "model_name": args.model_name,
            "model_path": str(args.model_path.resolve()),
            "samples": records,
        },
        args.reference_dir / "manifest.json",
    )


def tensor_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    if candidate.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: candidate={candidate.shape}, reference={reference.shape}"
        )
    candidate = candidate.float()
    reference = reference.float()
    difference = candidate - reference
    cosine = F.cosine_similarity(candidate, reference, dim=-1)
    reference_token_norm = torch.linalg.vector_norm(reference, dim=-1)
    difference_token_norm = torch.linalg.vector_norm(difference, dim=-1)
    token_relative_l2 = difference_token_norm / reference_token_norm.clamp_min(1e-12)
    return {
        "tokens": candidate.shape[0],
        "width": candidate.shape[-1],
        "cosine_mean": cosine.mean().item(),
        "cosine_median": cosine.median().item(),
        "cosine_p01": torch.quantile(cosine, 0.01).item(),
        "cosine_min": cosine.min().item(),
        "relative_l2": (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        ).item(),
        "token_relative_l2_mean": token_relative_l2.mean().item(),
        "token_relative_l2_p95": torch.quantile(token_relative_l2, 0.95).item(),
        "norm_ratio": (
            torch.linalg.vector_norm(candidate)
            / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        ).item(),
        "max_abs_difference": difference.abs().max().item(),
        "exact_element_fraction": (candidate == reference).float().mean().item(),
    }


def temporal_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    grid_thw: list[int],
    temporal_merge_size: int,
    spatial_merge_size: tuple[int, int],
) -> dict:
    time, height, width = grid_thw
    clips = math.ceil(time / temporal_merge_size)
    merge_height, merge_width = spatial_merge_size
    tokens_per_clip = height * width // (merge_height * merge_width)
    if clips * tokens_per_clip != candidate.shape[0]:
        raise ValueError(
            f"Unexpected merged token count: clips={clips}, tokens_per_clip="
            f"{tokens_per_clip}, tensor={candidate.shape}"
        )
    segments = {
        "first_clip": (0, 1),
        "first_quarter": (0, max(1, math.ceil(clips / 4))),
        "last_quarter": (max(0, clips - math.ceil(clips / 4)), clips),
        "last_clip": (clips - 1, clips),
    }
    output = {"clips": clips, "tokens_per_clip": tokens_per_clip}
    for name, (start_clip, end_clip) in segments.items():
        start = start_clip * tokens_per_clip
        end = end_clip * tokens_per_clip
        output[name] = tensor_metrics(candidate[start:end], reference[start:end])
    return output


def compare_checkpoint(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.input_dir)
    reference_manifest = json.loads(
        (args.reference_dir / "manifest.json").read_text()
    )
    reference_by_name = {
        item["name"]: item for item in reference_manifest["samples"]
    }
    model = load_model(args)
    temporal_merge_size = int(model.config.vision_config.temporal_merge_size)
    merge_kernel_size = model.config.vision_config.merge_kernel_size
    if isinstance(merge_kernel_size, int):
        spatial_merge_size = (merge_kernel_size, merge_kernel_size)
    else:
        spatial_merge_size = tuple(int(value) for value in merge_kernel_size)
    if len(spatial_merge_size) != 2:
        raise ValueError(f"Unexpected merge kernel: {spatial_merge_size}")
    result = {
        "model_name": args.model_name,
        "model_path": str(args.model_path.resolve()),
        "reference_model_name": reference_manifest["model_name"],
        "reference_model_path": reference_manifest["model_path"],
        "temporal_merge_size": temporal_merge_size,
        "spatial_merge_size": list(spatial_merge_size),
        "samples": [],
    }
    for sample in manifest["samples"]:
        reference_path = Path(reference_by_name[sample["name"]]["reference_path"])
        reference = torch.load(
            reference_path, map_location="cpu", weights_only=True, mmap=True
        )
        vision, projected = extract_features(
            model, Path(sample["input_path"]), args.device
        )
        reference_vision = reference["vision_tokens"].to(
            device=args.device, non_blocking=True
        )
        reference_projected = reference["projected_tokens"].to(
            device=args.device, non_blocking=True
        )
        record = {
            "name": sample["name"],
            "category": sample["category"],
            "frames": sample["frames"],
            "grid_thw": sample["grid_thw"],
            "vision_tokens": tensor_metrics(vision, reference_vision),
            "projected_tokens": tensor_metrics(projected, reference_projected),
            "vision_temporal": temporal_metrics(
                vision,
                reference_vision,
                sample["grid_thw"],
                temporal_merge_size,
                spatial_merge_size,
            ),
            "projected_temporal": temporal_metrics(
                projected,
                reference_projected,
                sample["grid_thw"],
                temporal_merge_size,
                spatial_merge_size,
            ),
        }
        result["samples"].append(record)
        atomic_json_dump(result, args.output_json)
        print(
            json.dumps(
                {
                    "name": record["name"],
                    "vision_cosine": record["vision_tokens"]["cosine_mean"],
                    "vision_relative_l2": record["vision_tokens"]["relative_l2"],
                    "projected_cosine": record["projected_tokens"]["cosine_mean"],
                    "projected_relative_l2": record["projected_tokens"]["relative_l2"],
                }
            ),
            flush=True,
        )
        del vision, projected, reference_vision, reference_projected, reference
        torch.cuda.empty_cache()
        gc.collect()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_inputs(args)
    elif args.command == "reference":
        save_references(args)
    elif args.command == "compare":
        compare_checkpoint(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
