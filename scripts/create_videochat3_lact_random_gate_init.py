#!/usr/bin/env python3
"""Clone a VideoChat3-LACT checkpoint and randomize only memory gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_safetensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    with safe_open(path, framework="pt", device="cpu") as stream:
        tensors = {key: stream.get_tensor(key) for key in stream.keys()}
        metadata = stream.metadata()
    return tensors, metadata


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    target = args.target.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(target)
    if source.parent != target.parent:
        raise ValueError("Source and target must share a parent for atomic creation")

    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    gate_keys = sorted(key for key in weight_map if key.endswith("memory_gate"))
    if len(gate_keys) != 27:
        raise ValueError(f"Expected 27 memory gates, found {len(gate_keys)}")
    gate_shards = sorted({weight_map[key] for key in gate_keys})

    temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)

    try:
        shutil.copytree(source, temporary, copy_function=os.link, symlinks=True)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        initialized = {}
        unchanged_tensors = 0
        for shard_name in gate_shards:
            source_shard = source / shard_name
            target_shard = temporary / shard_name
            tensors, metadata = load_safetensors(source_shard)
            for key in sorted(key for key in gate_keys if weight_map[key] == shard_name):
                source_gate = tensors[key]
                if source_gate.ndim != 1:
                    raise ValueError(f"Expected a 1D gate, got {key}: {tuple(source_gate.shape)}")
                if torch.count_nonzero(source_gate).item() != 0:
                    raise ValueError(f"Source gate is not zero initialized: {key}")
                hidden_size = source_gate.numel()
                bound = 1.0 / math.sqrt(hidden_size)
                random_gate = torch.empty_like(source_gate).uniform_(
                    -bound,
                    bound,
                    generator=generator,
                )
                tensors[key] = random_gate
                initialized[key] = random_gate.float()

            replacement = target_shard.with_suffix(target_shard.suffix + ".tmp")
            save_file(tensors, replacement, metadata=metadata)
            replacement.replace(target_shard)

            source_tensors, _ = load_safetensors(source_shard)
            target_tensors, _ = load_safetensors(target_shard)
            if source_tensors.keys() != target_tensors.keys():
                raise RuntimeError(f"Tensor keys changed in {shard_name}")
            for key in source_tensors:
                if key in initialized:
                    if torch.equal(source_tensors[key], target_tensors[key]):
                        raise RuntimeError(f"Gate did not change: {key}")
                else:
                    if not torch.equal(source_tensors[key], target_tensors[key]):
                        raise RuntimeError(f"Non-gate tensor changed: {key}")
                    unchanged_tensors += 1

        all_gate_values = torch.cat([initialized[key] for key in gate_keys])
        bound = 1.0 / math.sqrt(initialized[gate_keys[0]].numel())
        report = {
            "source": str(source),
            "target": str(target),
            "seed": args.seed,
            "initializer": "torch.nn.Linear fan-in uniform",
            "distribution": f"U(-1/sqrt(1152), 1/sqrt(1152))",
            "bound": bound,
            "gate_tensors": len(gate_keys),
            "gate_parameters": all_gate_values.numel(),
            "gate_mean": all_gate_values.mean().item(),
            "gate_rms": all_gate_values.square().mean().sqrt().item(),
            "gate_min": all_gate_values.min().item(),
            "gate_max": all_gate_values.max().item(),
            "modified_shards": gate_shards,
            "unchanged_tensors_in_modified_shards": unchanged_tensors,
            "source_index_sha256": sha256(index_path),
            "target_index_sha256": sha256(temporary / index_path.name),
        }
        if report["source_index_sha256"] != report["target_index_sha256"]:
            raise RuntimeError("Safetensors index changed")
        (temporary / "random_gate_initialization.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(target)
        print(json.dumps(report, indent=2, sort_keys=True))
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
