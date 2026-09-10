#!/usr/bin/env python3
"""Create and validate a fresh no-query Linear16+Delta training checkpoint."""

import argparse
import gc
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from xtuner.v1.model.compose.videochat3.videochat3_config import (
    VideoChat3LACTDense4BConfig,
    VideoChat3LACTVisionConfig,
)

BASE_REVISION = "37fa901ec5913f84bc31108ebc1e60ad1903634c"


def indexed_tensors(root, stack):
    mapping = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {
        shard: stack.enter_context(safe_open(root / shard, framework="pt", device="cpu"))
        for shard in sorted(set(mapping.values()))
    }
    actual = {key: shard for shard, handle in handles.items() for key in handle.keys()}
    assert actual == mapping, "Shard contents do not match the index"
    return {key: handles[shard].get_tensor(key) for key, shard in mapping.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/mnt/localssd/VideoChat3/VideoChat3-4B"))
    parser.add_argument("--target", type=Path, default=Path("/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    if target.exists():
        raise FileExistsError(target)
    if target.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Checkpoint must be outside the repository")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    remote = HfApi().model_info("MCG-NJU/VideoChat3-4B", revision=BASE_REVISION, files_metadata=True)
    assert len(remote.siblings) == 28
    for item in remote.siblings:
        path = source / item.rfilename
        assert path.is_file() and path.stat().st_size == item.size, path
    AutoConfig.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    AutoProcessor.from_pretrained(source, trust_remote_code=True, local_files_only=True)

    vision = VideoChat3LACTVisionConfig(
        attn_impl="flash_attention_2", memory_type="linear", fw_num_heads=16,
        inner_optim="delta", fw_order="parallel", fw_share_init=True,
        fw_share_proj=False, fw_base_lr=0.01, fw_update_layer_group_size=1,
        lact_3d_rope=True, lact_gate="linear", lact_gate_init=0,
        lact_chunk_query=False, macro_temporal_compression_factor=1,
        macro_temporal_compression_mode="auto", clip_ns_grad_ratio=False,
        clip_state_grad_ratio=False,
    )
    config = VideoChat3LACTDense4BConfig(
        vision_config=vision, freeze_vision=False, freeze_projector=False,
        freeze_language=True, train_lact_only=False,
    )
    with torch.device("meta"):
        model = config.build().to(torch.bfloat16)
    model.from_hf(source, strict=True)
    model.save_hf(target, save_dtype=torch.bfloat16)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    with ExitStack() as stack:
        base = indexed_tensors(source, stack)
        lact = indexed_tensors(target, stack)
        assert len(base) == 734
        assert set(base) <= set(lact)
        changed = [key for key in base if not torch.equal(base[key], lact[key])]
        assert not changed, changed
        added = sorted(set(lact) - set(base))
        assert all(key.startswith("model.vision_tower.") for key in added)
        assert all(torch.isfinite(lact[key]).all() for key in added)
        checks = []
        for index in range(27):
            prefix = f"model.vision_tower.encoder.blocks.{index}."
            q, k, v = base[prefix + "wqkv.weight"].chunk(3)
            pairs = {"memory.apply_proj.0.weight": q,
                     "memory.update_proj.0.weight": k,
                     "value_proj.weight": v,
                     "memory.output_proj.weight": base[prefix + "wo.weight"]}
            assert all(torch.equal(lact[prefix + name], tensor) for name, tensor in pairs.items())
            assert torch.count_nonzero(lact[prefix + "memory_gate"]) == 0
            assert torch.all(lact[prefix + "memory_norm.weight"] == 1)
            checks.append({"layer": index, "qkvo_share_init": True, "gate_zero": True, "memory_norm_unit": True})
        report = {"source": str(source), "source_revision": BASE_REVISION,
                  "target": str(target), "seed": args.seed,
                  "official_files_verified": len(remote.siblings),
                  "original_tensors_bitwise_unchanged": len(base),
                  "added_tensors": len(added),
                  "added_parameters": sum(lact[key].numel() for key in added),
                  "vision_config": vision.model_dump(mode="json"), "layers": checks}
        del base, lact, pairs, q, k, v

    processor = AutoProcessor.from_pretrained(target, trust_remote_code=True, local_files_only=True)
    report["processor_class"] = type(processor).__name__
    hf, loading = AutoModelForCausalLM.from_pretrained(
        target, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="flash_attention_2", output_loading_info=True,
    )
    assert not any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), loading
    hf.eval()
    original = AutoModelForCausalLM.from_pretrained(
        source, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="flash_attention_2",
    ).eval()
    parity = []
    with torch.inference_mode():
        for grids in ([[8, 16, 16]], [[9, 8, 8], [4, 8, 8]], [[1, 16, 16]]):
            grid = torch.tensor(grids, dtype=torch.int32, device="cuda")
            pixels = torch.randn(sum(t*h*w for t, h, w in grids), 3*14**2, dtype=torch.bfloat16, device="cuda")
            a = original.model.vision_tower(pixels, grid)
            b = hf.model.vision_tower(pixels, grid)
            assert len(a) == len(b)
            assert all(torch.equal(x, y) for x, y in zip(a, b)), grids
            parity.append({"grids": grids, "vision_bitwise_equal": True})
        tokens = 2 * 8 * 8
        ids = torch.tensor([[original.config.text_config.bos_token_id] + [original.config.video_token_id]*tokens + [42]], device="cuda")
        grid = torch.tensor([[8, 16, 16]], dtype=torch.int32, device="cuda")
        pixels = torch.randn(8*16*16, 3*14**2, dtype=torch.bfloat16, device="cuda")
        inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                      pixel_values_videos=pixels, video_grid_thw=grid,
                      use_cache=False, logits_to_keep=1)
        assert torch.equal(original(**inputs).logits, hf(**inputs).logits)
    report.update(hf_loading=loading, zero_gate_parity=parity, full_vlm_logits_bitwise_equal=True)
    (target / "initialization_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
