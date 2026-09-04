#!/usr/bin/env python3

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


def load_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    return json.loads(index_path.read_text())["weight_map"]


def parameter_group(name: str) -> str:
    if name.endswith("chunk_query"):
        return "chunk_query"
    if ".memory_gate" in name:
        return "memory_gate"
    if any(f".memory.{weight}" in name for weight in ("w0", "w1", "w2")):
        return "fw_base_weights"
    if any(
        projection in name
        for projection in (
            ".memory.apply_proj.",
            ".memory.update_proj.",
            ".memory.output_proj.",
        )
    ):
        return "fw_private_projections"
    if ".value_proj." in name:
        return "fw_value_projection"
    if ".lr_proj." in name:
        return "fw_lr_projection"
    if ".memory_norm." in name:
        return "fw_memory_norm"
    if name.startswith("model.vision_tower"):
        if any(part in name for part in (".wqkv.", ".wo.", ".norm0.")):
            return "original_attention"
        if any(part in name for part in (".mlp.", ".norm1.")):
            return "original_mlp"
        return "original_vision_other"
    if name.startswith("model.multi_modal_projector"):
        return "projector"
    if name.startswith("model.language_model") or name.startswith("lm_head"):
        return "language_model"
    return "other"


def empty_stats() -> dict[str, float | int]:
    return {
        "parameters": 0,
        "tensors": 0,
        "changed_tensors": 0,
        "base_sq": 0.0,
        "delta_sq": 0.0,
        "delta_abs_max": 0.0,
    }


def finalize(stats: dict[str, float | int]) -> dict[str, float | int | bool | None]:
    parameters = int(stats["parameters"])
    base_sq = float(stats["base_sq"])
    delta_sq = float(stats["delta_sq"])
    return {
        "parameters": parameters,
        "tensors": int(stats["tensors"]),
        "changed_tensors": int(stats["changed_tensors"]),
        "bitwise_unchanged": int(stats["changed_tensors"]) == 0,
        "base_rms": math.sqrt(base_sq / parameters) if parameters else None,
        "delta_rms": math.sqrt(delta_sq / parameters) if parameters else None,
        "relative_l2_delta": math.sqrt(delta_sq / base_sq) if base_sq else None,
        "delta_abs_max": float(stats["delta_abs_max"]),
    }


def compare_checkpoints(init_dir: Path, trained_dir: Path) -> dict:
    init_index = load_index(init_dir)
    trained_index = load_index(trained_dir)
    if init_index.keys() != trained_index.keys():
        missing = sorted(init_index.keys() - trained_index.keys())
        extra = sorted(trained_index.keys() - init_index.keys())
        raise ValueError(f"Checkpoint keys differ: missing={missing[:5]}, extra={extra[:5]}")

    grouped = defaultdict(empty_stats)
    gate_values = []
    init_handles = {}
    trained_handles = {}
    try:
        for name in sorted(init_index):
            init_file = init_index[name]
            trained_file = trained_index[name]
            if init_file not in init_handles:
                init_handles[init_file] = safe_open(
                    init_dir / init_file,
                    framework="pt",
                    device="cpu",
                )
            if trained_file not in trained_handles:
                trained_handles[trained_file] = safe_open(
                    trained_dir / trained_file,
                    framework="pt",
                    device="cpu",
                )

            initial = init_handles[init_file].get_tensor(name)
            trained = trained_handles[trained_file].get_tensor(name)
            if initial.shape != trained.shape:
                raise ValueError(f"Shape mismatch for {name}: {initial.shape} != {trained.shape}")

            initial_fp32 = initial.float()
            trained_fp32 = trained.float()
            delta = trained_fp32 - initial_fp32
            group = grouped[parameter_group(name)]
            group["parameters"] += initial.numel()
            group["tensors"] += 1
            group["changed_tensors"] += int(not torch.equal(initial, trained))
            group["base_sq"] += torch.sum(initial_fp32.square()).item()
            group["delta_sq"] += torch.sum(delta.square()).item()
            group["delta_abs_max"] = max(
                float(group["delta_abs_max"]),
                torch.max(torch.abs(delta)).item(),
            )
            if ".memory_gate" in name:
                gate_values.append(trained_fp32.flatten())
    finally:
        init_handles.clear()
        trained_handles.clear()

    result = {
        "initial_checkpoint": str(init_dir.resolve()),
        "trained_checkpoint": str(trained_dir.resolve()),
        "groups": {name: finalize(stats) for name, stats in sorted(grouped.items())},
        "memory_gate": None,
    }
    if gate_values:
        gates = torch.cat(gate_values)
        gate_abs = gates.abs()
        quantiles = torch.quantile(
            gate_abs,
            torch.tensor([0.5, 0.9, 0.99], dtype=gate_abs.dtype),
        )
        result["memory_gate"] = {
            "parameters": gates.numel(),
            "rms": torch.sqrt(torch.mean(gates.square())).item(),
            "mean_abs": torch.mean(gate_abs).item(),
            "max_abs": torch.max(gate_abs).item(),
            "median_abs": quantiles[0].item(),
            "p90_abs": quantiles[1].item(),
            "p99_abs": quantiles[2].item(),
        }
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    result = compare_checkpoints(args.init, args.trained)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
