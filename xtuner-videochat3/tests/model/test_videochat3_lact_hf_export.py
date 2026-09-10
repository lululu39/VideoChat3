import json
import os
import shutil
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from xtuner.v1.model.compose.videochat3.videochat3_config import (
    VideoChat3Dense4BConfig,
    VideoChat3LACTDense4BConfig,
    VideoChat3LACTVisionConfig,
    VideoChat3ProjectorConfig,
    VideoChat3VisionConfig,
)
from xtuner.v1.model.dense.qwen3 import Qwen3Dense4BConfig
from xtuner.v1.module.attention import MHAConfig


OFFICIAL_CHECKPOINT = Path(
    os.environ.get(
        "VIDEOCHAT3_DENSE_PATH",
        "/mnt/localssd/VideoChat3/VideoChat3-4B",
    )
)


@pytest.mark.parametrize("factor,mode", [(16, "select_last"), (128, "video_last")])
@pytest.mark.skipif(not (OFFICIAL_CHECKPOINT / "modeling_videochat3.py").is_file(),
                    reason="Official VideoChat3 remote code is unavailable")
def test_curriculum_stage_export_is_native_and_keeps_parameters(tmp_path, factor, mode):
    from xtuner.v1.datasets.videochat3_curriculum import ChunkStage
    from xtuner.v1.train.videochat3_curriculum import VideoChat3CurriculumTrainer
    cfg = _tiny_model_config()
    cfg.vision_config.memory_type = "linear"
    cfg.vision_config.inner_optim = "delta"
    cfg.vision_config.fw_num_heads = 4
    cfg.vision_config.lact_chunk_query = False
    cfg.vision_config.clip_state_grad_ratio = False
    model = cfg.build()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    fake = SimpleNamespace(_trainer_cfg=SimpleNamespace(model_cfg=cfg),
        _engine=SimpleNamespace(model=model), _active_video_stage=None, rank=0,
        logger=SimpleNamespace(info=lambda message: None), exp_dir=tmp_path, total_step=16)
    VideoChat3CurriculumTrainer._activate_video_stage(fake, ChunkStage(8, 14, 16, factor, mode))
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)
    base_path, save_path = tmp_path / "base", tmp_path / "saved"
    base_path.mkdir()
    for filename in ("configuration_videochat3.py", "modeling_videochat3.py"):
        shutil.copy2(OFFICIAL_CHECKPOINT / filename, base_path / filename)
    _write_tiny_base_config(base_path, cfg)
    model.set_hf(base_path)
    model.save_hf(save_path, save_dtype=torch.bfloat16)
    model.to(torch.bfloat16).float()
    processor = json.loads((save_path / "processor_config.json").read_text())
    assert processor["macro_temporal_compression_factor"] == factor
    assert processor["macro_temporal_compression_mode"] == mode
    hf, info = AutoModelForCausalLM.from_pretrained(save_path, trust_remote_code=True,
                                                   dtype=torch.float32, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    assert hf.config.vision_config.macro_temporal_compression_factor == factor
    assert hf.config.vision_config.macro_temporal_compression_mode == mode
    assert hf.model.vision_tower.chunk_query is None
    for block in hf.model.vision_tower.encoder.blocks:
        block.attn_impl = "sdpa"
    pixels = torch.randn(368, 12)
    grids = torch.tensor([[80, 2, 2], [12, 2, 2]], dtype=torch.int32)
    expected = model.vision_tower(pixels, grids)
    actual = hf.model.vision_tower(pixels, grids)
    assert len(actual) == (2 if mode == "video_last" else 3)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


def _tiny_model_config():
    vision_config = VideoChat3LACTVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        patch_size=2,
        merge_kernel_size=[2, 2],
        temporal_merge_size=4,
        init_pos_emb_height=2,
        init_pos_emb_width=2,
        attn_impl="eager_attention",
        fw_muon_update_steps=0,
        fw_order="parallel",
        lact_3d_rope=True,
        lact_chunk_query=True,
        lact_gate="tanh",
        lact_gate_init=0.5,
    )
    text_config = Qwen3Dense4BConfig(
        vocab_size=128,
        max_position_embeddings=128,
        eos_token_id=1,
        bos_token_id=0,
        num_hidden_layers=1,
        max_window_layers=1,
        hidden_size=32,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        hidden_act="silu",
        attention=MHAConfig(
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            qk_norm=True,
            sliding_window=None,
        ),
        tie_word_embeddings=False,
    )
    return VideoChat3LACTDense4BConfig(
        vision_config=vision_config,
        projector_config=VideoChat3ProjectorConfig(
            vision_hidden_size=16,
            text_hidden_size=32,
            merge_kernel_size=[2, 2],
        ),
        text_config=text_config,
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=102,
        vision_end_token_id=103,
    )


@pytest.mark.parametrize("source_memory", ["linear", "swiglu", "partial_linear"])
def test_linear_transfer_preserves_trained_branch_or_share_initializes(tmp_path, source_memory):
    cfg = _tiny_model_config()
    cfg.vision_config.memory_type = source_memory.removeprefix("partial_")
    cfg.vision_config.fw_num_heads = 4
    cfg.vision_config.inner_optim = "delta"
    cfg.vision_config.lact_chunk_query = False
    base_path, save_path = tmp_path / "base", tmp_path / "saved"
    base_path.mkdir()
    _write_tiny_base_config(base_path, cfg)
    model = cfg.build()
    with torch.no_grad():
        for name, param in model.vision_tower.named_parameters():
            if model.vision_tower._is_lact_state_key(name):
                param.add_(0.125)
    expected = {k: v.to(torch.bfloat16).float() for k, v in model.vision_tower.state_dict().items()}
    model.set_hf(base_path)
    model.save_hf(save_path)
    if source_memory == "partial_linear":
        from safetensors.torch import load_file, save_file
        index_path = save_path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        gate_key = next(k for k in index["weight_map"] if k.endswith("memory_gate"))
        shard = save_path / index["weight_map"].pop(gate_key)
        tensors = load_file(shard)
        del tensors[gate_key]
        save_file(tensors, shard)
        index_path.write_text(json.dumps(index))
    cfg.vision_config.memory_type = "linear"
    restored = cfg.build()
    if source_memory == "partial_linear":
        with pytest.raises(RuntimeError, match="Incomplete Linear LACT"):
            restored.from_hf(save_path, strict=True)
        return
    restored.from_hf(save_path, strict=True)
    if source_memory == "linear":
        for key, value in restored.vision_tower.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    else:
        for block in restored.vision_tower.encoder.blocks:
            assert torch.all(block.memory_gate == cfg.vision_config.lact_gate_init)
            torch.testing.assert_close(block.memory.apply_proj[0].weight, block.wqkv.weight.chunk(3)[0], rtol=0, atol=0)


def _tiny_base_query_model_config():
    vision_config = VideoChat3VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        patch_size=2,
        merge_kernel_size=[2, 2],
        temporal_merge_size=4,
        init_pos_emb_height=4,
        init_pos_emb_width=8,
        attn_impl="eager_attention",
        chunk_query=True,
        chunk_query_mode="spatial_quarter",
    )
    text_config = Qwen3Dense4BConfig(
        vocab_size=128,
        max_position_embeddings=128,
        eos_token_id=1,
        bos_token_id=0,
        num_hidden_layers=1,
        max_window_layers=1,
        hidden_size=32,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        hidden_act="silu",
        attention=MHAConfig(
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            qk_norm=True,
            sliding_window=None,
        ),
        tie_word_embeddings=False,
    )
    return VideoChat3Dense4BConfig(
        vision_config=vision_config,
        projector_config=VideoChat3ProjectorConfig(
            vision_hidden_size=16,
            text_hidden_size=32,
            merge_kernel_size=[2, 2],
        ),
        text_config=text_config,
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=102,
        vision_end_token_id=103,
    )


def _write_tiny_base_config(path: Path, model_config) -> None:
    vision_config = model_config.vision_config
    config = {
        "architectures": ["VideoChat3ForConditionalGeneration"],
        "auto_map": {
            "AutoConfig": "configuration_videochat3.VideoChat3Config",
            "AutoModel": ("modeling_videochat3.VideoChat3ForConditionalGeneration"),
            "AutoModelForCausalLM": ("modeling_videochat3.VideoChat3ForConditionalGeneration"),
        },
        "model_type": "videochat3",
        "vision_config": {
            "hidden_size": vision_config.hidden_size,
            "intermediate_size": vision_config.intermediate_size,
            "num_attention_heads": vision_config.num_attention_heads,
            "num_hidden_layers": vision_config.num_hidden_layers,
            "patch_size": vision_config.patch_size,
            "merge_kernel_size": vision_config.merge_kernel_size,
            "temporal_patch_size": vision_config.temporal_patch_size,
            "temporal_merge_size": vision_config.temporal_merge_size,
            "init_pos_emb_height": vision_config.init_pos_emb_height,
            "init_pos_emb_width": vision_config.init_pos_emb_width,
            "dtype": "float32",
            "attn_impl": "eager",
        },
        "text_config": model_config.text_config.hf_config.to_dict(),
        "image_token_id": model_config.image_token_id,
        "video_token_id": model_config.video_token_id,
        "vision_start_token_id": model_config.vision_start_token_id,
        "vision_end_token_id": model_config.vision_end_token_id,
    }
    (path / "config.json").write_text(json.dumps(config, indent=2))
    (path / "processor_config.json").write_text(
        json.dumps({"processor_class": "VideoChat3Processor"}, indent=2)
    )


@pytest.mark.parametrize("variant", ["lact", "macro"])
@pytest.mark.parametrize("mode", ["chunk_select_last", "chunk_select_uniform"])
@pytest.mark.skipif(not (OFFICIAL_CHECKPOINT / "modeling_videochat3.py").is_file(),
                    reason="Official VideoChat3 remote code is not available")
def test_chunk_selection_hf_roundtrip_and_processor_layout(tmp_path, monkeypatch, variant, mode):
    config = _tiny_model_config() if variant == "lact" else _tiny_base_query_model_config()
    query_flag = "lact_chunk_query" if variant == "lact" else "chunk_query"
    updates = {
        query_flag: False, f"{query_flag}_mode": "single",
        "macro_temporal_compression_factor": 4,
        "macro_temporal_compression_mode": mode,
        "init_pos_emb_height": 4, "init_pos_emb_width": 8,
    }
    if variant == "lact":
        updates.update(memory_type="linear", inner_optim="delta", fw_num_heads=4,
                       lact_gate="linear", lact_gate_init=0.1, clip_state_grad_ratio=False)
    config.vision_config = type(config.vision_config)(**(config.vision_config.model_dump() | updates))
    base_path, save_path = tmp_path / "base", tmp_path / "saved"
    base_path.mkdir()
    for source in OFFICIAL_CHECKPOINT.glob("*.py"):
        shutil.copy2(source, base_path / source.name)
    _write_tiny_base_config(base_path, config)
    model = config.build()
    model.set_hf(base_path)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if tensor.is_floating_point():
                tensor.copy_(tensor.to(torch.bfloat16).float())
    model.save_hf(save_path, save_dtype=torch.bfloat16)
    saved = AutoConfig.from_pretrained(save_path, trust_remote_code=True)
    assert saved.vision_config.macro_temporal_compression_mode == mode
    assert saved.vision_config.macro_temporal_compression_factor == 4
    assert not getattr(saved.vision_config, query_flag)
    processor_config = json.loads((save_path / "processor_config.json").read_text())
    assert processor_config["macro_temporal_compression_mode"] == mode
    assert processor_config["macro_temporal_compression_factor"] == 4
    hf_model, info = AutoModelForCausalLM.from_pretrained(
        save_path, trust_remote_code=True, dtype=torch.float32, output_loading_info=True,
    )
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    assert not any("chunk_query" in name for name, _ in hf_model.named_parameters())
    for block in hf_model.model.vision_tower.encoder.blocks:
        block.attn_impl = "sdpa"
    grids = torch.tensor([[9, 4, 8], [4, 2, 2]], dtype=torch.int32)
    pixels = torch.randn(304, 12)
    with torch.no_grad():
        expected = model.vision_tower(pixels, grids)
        actual = hf_model.model.vision_tower(pixels, grids)
    assert [output.shape[0] for output in actual] == [2, 2, 2, 1]
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)

    class_name = "VideoChat3LACTProcessor" if variant == "lact" else "VideoChat3MacroProcessor"
    processor_cls = get_class_from_dynamic_module(f"processing_videochat3_{variant}.{class_name}", save_path)
    processor = processor_cls.__new__(processor_cls)
    processor.macro_temporal_compression_mode = mode
    processor.macro_temporal_compression_factor = 4
    setattr(processor, query_flag, False)
    setattr(processor, f"{query_flag}_mode", "single")
    processor.video_token_id, processor.image_token_id = 101, 100
    processor.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1, padding_side="right")
    meta = SimpleNamespace(timestamps=list(range(9)))
    assert processor._calculate_timestamps(meta) == [1.5, 5.5, 8.0]
    ids = torch.tensor([[7] + [101] * 8 + [8] + [101] * 6 + [9] + [100] * 8 + [10]])
    def parent_call(self, *args, **kwargs):
        return {"input_ids": ids.clone(), "attention_mask": torch.ones_like(ids),
                "position_trace": torch.arange(ids.numel()).unsqueeze(0), "pixel_values": pixels}
    monkeypatch.setattr(processor_cls.__mro__[1], "__call__", parent_call)
    result = processor()
    assert result["input_ids"].tolist() == [[7, 101, 101, 8, 101, 9, 100, 100, 10]]
    expected_positions = (
        [0, 7, 8, 9, 15, 16, 23, 24, 25]
        if mode == "chunk_select_last"
        else [0, 1, 8, 9, 13, 16, 17, 24, 25]
    )
    assert result["position_trace"].tolist() == [expected_positions]
    assert result["attention_mask"].shape == result["input_ids"].shape
    assert result["pixel_values"] is pixels


def test_lact_only_config_freezes_every_original_vision_parameter():
    model_config = _tiny_model_config().model_copy(
        update={"train_lact_only": True},
    )
    model = model_config.build()
    trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable
    assert all(name.startswith("vision_tower.") for name in trainable)
    assert all(
        model.vision_tower._is_lact_state_key(
            name.removeprefix("vision_tower."),
        )
        for name in trainable
    )
    assert any(name.endswith("memory_gate") for name in trainable)
    assert not any(".wqkv." in name or ".mlp." in name for name in trainable)


def test_lact_only_config_can_freeze_memory_gates():
    model_config = _tiny_model_config().model_copy(
        update={
            "train_lact_only": True,
            "freeze_lact_memory_gate": True,
        },
    )
    model = model_config.build()
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    frozen_gates = {
        name
        for name, parameter in model.named_parameters()
        if name.endswith("memory_gate") and not parameter.requires_grad
    }

    assert trainable
    assert frozen_gates
    assert not any(name.endswith("memory_gate") for name in trainable)
    assert all(name.startswith("vision_tower.") for name in trainable)
    assert all(
        model.vision_tower._is_lact_state_key(
            name.removeprefix("vision_tower."),
        )
        for name in trainable
    )


@pytest.mark.skipif(
    not (OFFICIAL_CHECKPOINT / "modeling_videochat3.py").is_file(),
    reason="Official VideoChat3 remote code is not available",
)
def test_hf_interval_save_loads_independent_lact_model(tmp_path):
    model_config = _tiny_model_config()
    base_path = tmp_path / "base"
    save_path = tmp_path / "saved"
    base_path.mkdir()
    for file_name in ("configuration_videochat3.py", "modeling_videochat3.py"):
        shutil.copy2(OFFICIAL_CHECKPOINT / file_name, base_path / file_name)
    _write_tiny_base_config(base_path, model_config)

    model = model_config.build()
    assert type(model).__name__ == "VideoChat3LACTForConditionalGeneration"
    with torch.no_grad():
        for block in model.vision_tower.encoder.blocks:
            block.memory_gate.fill_(0.1)
    model.set_hf(base_path)
    model.save_hf(save_path, save_dtype=torch.bfloat16)

    saved_config = json.loads((save_path / "config.json").read_text())
    assert saved_config["model_type"] == "videochat3_lact"
    assert saved_config["vision_config"]["model_type"] == "videochat3_lact_vision"
    assert saved_config["vision_config"]["clip_ns_grad_ratio"] is False
    assert saved_config["vision_config"]["clip_state_grad_ratio"] is True
    assert saved_config["vision_config"]["macro_temporal_compression_mode"] == "auto"
    assert saved_config["vision_config"]["lact_3d_rope"] is True
    assert saved_config["vision_config"]["lact_chunk_query"] is True
    assert saved_config["vision_config"]["lact_chunk_query_mode"] == "single"
    assert saved_config["vision_config"]["fw_order"] == "parallel"
    assert saved_config["vision_config"]["lact_gate"] == "tanh"
    assert saved_config["vision_config"]["lact_gate_init"] == 0.5
    saved_processor = json.loads((save_path / "processor_config.json").read_text())
    assert saved_processor["lact_chunk_query"] is True
    assert saved_processor["lact_chunk_query_mode"] == "single"
    assert saved_config["architectures"] == ["VideoChat3LACTForConditionalGeneration"]
    assert saved_config["auto_map"]["AutoModelForCausalLM"] == (
        "modeling_videochat3_lact.VideoChat3LACTForConditionalGeneration"
    )
    assert saved_config["auto_map"]["AutoModelForImageTextToText"] == (
        "modeling_videochat3_lact.VideoChat3LACTForConditionalGeneration"
    )

    hf_config = AutoConfig.from_pretrained(save_path, trust_remote_code=True)
    assert type(hf_config).__name__ == "VideoChat3LACTConfig"
    assert type(hf_config.vision_config).__name__ == ("VideoChat3LACTVisionConfig")
    assert hf_config.vision_config.clip_ns_grad_ratio is False
    assert hf_config.vision_config.clip_state_grad_ratio is True
    assert hf_config.vision_config.macro_temporal_compression_mode == "auto"
    assert hf_config.vision_config.lact_3d_rope is True
    assert hf_config.vision_config.lact_chunk_query is True
    assert hf_config.vision_config.lact_chunk_query_mode == "single"
    assert hf_config.vision_config.fw_order == "parallel"
    assert hf_config.vision_config.lact_gate == "tanh"
    assert hf_config.vision_config.lact_gate_init == 0.5
    hf_model, loading_info = AutoModelForCausalLM.from_pretrained(
        save_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert not loading_info["mismatched_keys"]
    assert type(hf_model).__name__ == "VideoChat3LACTForConditionalGeneration"
    assert type(hf_model.model.vision_tower).__name__ == ("VideoChat3LACTVisionModel")
    assert hf_model.model.vision_tower.chunk_query.shape == (16,)
    image_text_model = AutoModelForImageTextToText.from_pretrained(
        save_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    assert type(image_text_model).__name__ == ("VideoChat3LACTForConditionalGeneration")

    xtuner_vision_state = model.vision_tower.state_dict()
    hf_vision_state = hf_model.model.vision_tower.state_dict()
    assert xtuner_vision_state.keys() == hf_vision_state.keys()
    assert any(".memory.w0" in key for key in hf_vision_state)
    for key, xtuner_tensor in xtuner_vision_state.items():
        torch.testing.assert_close(
            hf_vision_state[key],
            xtuner_tensor.to(torch.bfloat16),
            rtol=0,
            atol=0,
        )

    resumed_model = model_config.build()
    resumed_model.from_hf(save_path, strict=True)
    for key, original_tensor in model.vision_tower.state_dict().items():
        resumed_tensor = resumed_model.vision_tower.state_dict()[key]
        torch.testing.assert_close(
            resumed_tensor,
            original_tensor.to(torch.bfloat16).to(resumed_tensor.dtype),
            rtol=0,
            atol=0,
        )

    hf_model.float()
    for block in hf_model.model.vision_tower.encoder.blocks:
        # The pinned HF base exposes a correct packed SDPA fallback.
        block.attn_impl = "sdpa"
    torch.manual_seed(23)
    pixel_values = torch.randn(32, 12)
    grid_thws = torch.tensor([[8, 2, 2]], dtype=torch.int32)
    hf_outputs = hf_model.model.vision_tower(pixel_values, grid_thws)
    resumed_outputs = resumed_model.vision_tower(pixel_values, grid_thws)
    projected_outputs = resumed_model.multi_modal_projector(resumed_outputs)
    assert projected_outputs.shape == (2, 32)
    for hf_output, resumed_output in zip(hf_outputs, resumed_outputs, strict=True):
        torch.testing.assert_close(
            hf_output,
            resumed_output,
            rtol=2e-5,
            atol=2e-6,
        )


@pytest.mark.skipif(
    not (OFFICIAL_CHECKPOINT / "modeling_videochat3.py").is_file(),
    reason="Official VideoChat3 remote code is not available",
)
def test_hf_interval_save_loads_base_chunk_query_model(tmp_path):
    model_config = _tiny_base_query_model_config()
    base_path = tmp_path / "base"
    save_path = tmp_path / "saved"
    base_path.mkdir()
    for file_name in ("configuration_videochat3.py", "modeling_videochat3.py"):
        shutil.copy2(OFFICIAL_CHECKPOINT / file_name, base_path / file_name)
    _write_tiny_base_config(base_path, model_config)

    model = model_config.build()
    assert not any("memory" in name for name, _ in model.named_parameters())
    model.set_hf(base_path)
    model.save_hf(save_path, save_dtype=torch.bfloat16)

    saved_config = json.loads((save_path / "config.json").read_text())
    saved_processor = json.loads((save_path / "processor_config.json").read_text())
    assert saved_config["model_type"] == "videochat3_macro"
    assert saved_config["vision_config"]["chunk_query"] is True
    assert saved_config["vision_config"]["chunk_query_mode"] == "spatial_quarter"
    assert saved_processor["chunk_query"] is True
    assert saved_processor["chunk_query_mode"] == "spatial_quarter"

    hf_config = AutoConfig.from_pretrained(save_path, trust_remote_code=True)
    assert hf_config.vision_config.chunk_query is True
    assert hf_config.vision_config.chunk_query_mode == "spatial_quarter"
    hf_model, loading_info = AutoModelForCausalLM.from_pretrained(
        save_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert not loading_info["mismatched_keys"]
    assert hf_model.model.vision_tower.chunk_query.shape == (16, 16)
    assert not any("memory" in name for name, _ in hf_model.named_parameters())

    hf_model.float()
    for block in hf_model.model.vision_tower.encoder.blocks:
        block.attn_impl = "sdpa"
    outputs = hf_model.model.vision_tower(
        torch.randn(256, 12),
        torch.tensor([[8, 4, 8]], dtype=torch.int32),
    )
    assert len(outputs) == 4
    assert all(output.shape == (1, 4, 16) for output in outputs)
