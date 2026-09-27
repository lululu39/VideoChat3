#!/usr/bin/env python3
"""TAS group deltas and frozen-backbone integrity, reusing shard comparison."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
import inspect_videochat3_lact_checkpoint as common


original_group = common.parameter_group


def group(name):
    if ".tas." in name:
        suffix = name.split(".tas.", 1)[1]
        if suffix.startswith("readers."):
            return "tas_readers"
        if suffix.startswith("time_encoding."):
            return "tas_time_encoding"
        if suffix.startswith("gate_"):
            return "tas_ema_gate"
        if suffix in ("initial_state", "slot_embedding"):
            return "tas_" + suffix
        return "tas_writer"
    return original_group(name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--trained", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    common.parameter_group = group
    result = common.compare_checkpoints(args.init, args.trained)
    result.pop("memory_gate", None)
    assert result["groups"]["language_model"]["bitwise_unchanged"], "Frozen LM changed"
    index = common.load_index(args.trained)
    prefix = "model.vision_tower.tas."
    gate = {}
    for name in ("gate_weight", "gate_bias"):
        with safe_open(args.trained/index[prefix+name], framework="pt") as f:
            v = f.get_tensor(prefix+name).float()
            gate[name] = dict(rms=v.square().mean().sqrt().item(), min=v.min().item(), max=v.max().item())
    result["ema_gate_parameters"] = gate
    result["note"] = "EMA is input-dependent; parameter statistics are not realized gate statistics."
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
