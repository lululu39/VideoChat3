#!/usr/bin/env python3
import argparse
import gc
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark original and LACT VideoChat3 checkpoints."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--lact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--grid-height", type=int, default=16)
    parser.add_argument("--grid-width", type=int, default=16)
    parser.add_argument("--text-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--frames", type=int, nargs="+", default=[4, 8])
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def summarize_times(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    return {
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def benchmark_callable(
    function,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_memory = torch.cuda.memory_allocated(device)

    samples_ms = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        function()
        synchronize(device)
        samples_ms.append((time.perf_counter() - start) * 1000)

    result = summarize_times(samples_ms)
    result["peak_working_memory_gib"] = (
        torch.cuda.max_memory_allocated(device) - baseline_memory
    ) / 2**30
    return result


def make_inputs(
    model,
    *,
    frames: int,
    grid_height: int,
    grid_width: int,
    text_tokens: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    patch_dim = 3 * model.config.vision_config.patch_size**2
    pixel_values = torch.randn(
        frames * grid_height * grid_width,
        patch_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    grid_thw = torch.tensor(
        [[frames, grid_height, grid_width]],
        device=device,
        dtype=torch.int32,
    )
    temporal_merge_size = model.config.vision_config.temporal_merge_size
    clip_count = (frames + temporal_merge_size - 1) // temporal_merge_size
    visual_tokens = clip_count * (grid_height // 2) * (grid_width // 2)
    ordinary_token = model.config.text_config.bos_token_id
    input_ids = torch.tensor(
        [
            [ordinary_token]
            + [model.config.video_token_id] * visual_tokens
            + [ordinary_token] * text_tokens
        ],
        device=device,
        dtype=torch.long,
    )
    return {
        "pixel_values": pixel_values,
        "grid_thw": grid_thw,
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "visual_tokens": visual_tokens,
    }


def benchmark_model(
    checkpoint: Path,
    *,
    device: torch.device,
    frame_counts: list[int],
    grid_height: int,
    grid_width: int,
    text_tokens: int,
    warmup: int,
    repeats: int,
) -> dict:
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": device.index},
        attn_implementation="flash_attention_2",
    ).eval()
    model_result = {
        "checkpoint": str(checkpoint),
        "model_class": type(model).__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "cases": {},
    }

    with torch.inference_mode():
        for frames in frame_counts:
            inputs = make_inputs(
                model,
                frames=frames,
                grid_height=grid_height,
                grid_width=grid_width,
                text_tokens=text_tokens,
                device=device,
            )

            def vision_forward():
                return model.model.vision_tower(
                    inputs["pixel_values"], inputs["grid_thw"]
                )

            def full_forward():
                return model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values_videos=inputs["pixel_values"],
                    video_grid_thw=inputs["grid_thw"],
                    logits_to_keep=1,
                    use_cache=False,
                )

            case = {
                "frames": frames,
                "grid_thw": [frames, grid_height, grid_width],
                "vision_patch_tokens": frames * grid_height * grid_width,
                "llm_visual_tokens": inputs["visual_tokens"],
                "llm_sequence_tokens": inputs["input_ids"].shape[1],
                "vision_only": benchmark_callable(
                    vision_forward,
                    device=device,
                    warmup=warmup,
                    repeats=repeats,
                ),
                "full_vlm": benchmark_callable(
                    full_forward,
                    device=device,
                    warmup=warmup,
                    repeats=repeats,
                ),
            }
            model_result["cases"][str(frames)] = case
            inputs = None

    model = None
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)
    return model_result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.grid_height % 2 or args.grid_width % 2:
        raise ValueError("grid height and width must be divisible by 2")
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "settings": {
            "dtype": "bfloat16",
            "attention": "flash_attention_2",
            "inference_mode": True,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "grid_height": args.grid_height,
            "grid_width": args.grid_width,
            "text_tokens": args.text_tokens,
            "frames": args.frames,
        },
        "baseline": benchmark_model(
            args.baseline,
            device=device,
            frame_counts=args.frames,
            grid_height=args.grid_height,
            grid_width=args.grid_width,
            text_tokens=args.text_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "lact": benchmark_model(
            args.lact,
            device=device,
            frame_counts=args.frames,
            grid_height=args.grid_height,
            grid_width=args.grid_width,
            text_tokens=args.text_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    }
    report["ratios"] = {}
    for frames in args.frames:
        frame_key = str(frames)
        report["ratios"][frame_key] = {}
        for scope in ("vision_only", "full_vlm"):
            baseline_ms = report["baseline"]["cases"][frame_key][scope]["median_ms"]
            lact_ms = report["lact"]["cases"][frame_key][scope]["median_ms"]
            report["ratios"][frame_key][scope] = {
                "lact_over_baseline": lact_ms / baseline_ms,
                "slowdown_percent": (lact_ms / baseline_ms - 1) * 100,
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
