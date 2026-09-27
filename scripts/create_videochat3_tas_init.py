#!/usr/bin/env python3
"""Export a new Base-initialized TAS checkpoint without modifying the source."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from safetensors import safe_open
from xtuner.v1.model.compose.videochat3.hf_tas_export import prepare_tas_assets, refresh_tas_code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("/mnt/localssd/VideoChat3/VideoChat3-4B"))
    p.add_argument("--target", type=Path, default=Path("/mnt/localssd/VideoChat3/VideoChat3-4B-TAS-init"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.target.exists():
        raise FileExistsError(args.target)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    prepare_tas_assets(args.source, args.target)
    config = AutoConfig.from_pretrained(args.target, trust_remote_code=True)
    cls = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], args.target)
    model, loading = cls.from_pretrained(args.source, config=config, dtype=torch.bfloat16,
                                        attn_implementation="eager", output_loading_info=True)
    assert not loading["unexpected_keys"] and not loading["mismatched_keys"], loading
    assert loading["missing_keys"] and all(k.startswith("model.vision_tower.tas.") for k in loading["missing_keys"]), loading
    model.model.vision_tower.tas.reset_parameters()
    # A from_pretrained missing-parameter pass may initialize standalone params.
    memory = model.model.vision_tower.tas
    with torch.no_grad():
        memory.gate_weight.zero_()
        memory.gate_bias.fill_(torch.logit(torch.tensor(config.vision_config.tas_ema_init)).item())
    config.vision_config.attn_impl = "flash_attention_2"
    model.save_pretrained(args.target, max_shard_size="4GB")
    refresh_tas_code(args.target)
    processor = AutoProcessor.from_pretrained(args.target, trust_remote_code=True)
    index = json.loads((args.source/"model.safetensors.index.json").read_text())["weight_map"]
    state = model.state_dict()
    checked = 0
    for shard in set(index.values()):
        with safe_open(args.source/shard, framework="pt") as f:
            for key in f.keys():
                assert torch.equal(f.get_tensor(key), state[key]), key
                checked += 1
    # state_dict includes a tied lm_head alias omitted by safetensors. Count
    # only actually exported new tensors, not that frozen embedding alias.
    exported_index = json.loads((args.target/"model.safetensors.index.json").read_text())["weight_map"]
    added = {k: state[k] for k in exported_index if k not in index}
    assert all(k.startswith("model.vision_tower.tas.") for k in added)
    assert all(torch.isfinite(v).all() for v in added.values())
    assert torch.equal(memory.initial_state, memory.slot_embedding)
    report = dict(source=str(args.source), source_revision="37fa901ec5913f84bc31108ebc1e60ad1903634c",
                  seed=args.seed, original_tensors_bitwise_unchanged=checked,
                  added_tensors=len(added), added_parameters=sum(v.numel() for v in added.values()),
                  processor_class=type(processor).__name__, vision_config=config.vision_config.to_dict())
    (args.target/"initialization_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
