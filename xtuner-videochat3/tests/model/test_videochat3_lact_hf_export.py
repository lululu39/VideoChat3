import json
import os
import shutil
from pathlib import Path

import pytest
import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from xtuner.v1.model.compose.videochat3.videochat3_config import (
    VideoChat3LACTDense4BConfig,
    VideoChat3LACTVisionConfig,
    VideoChat3ProjectorConfig,
)
from xtuner.v1.model.dense.qwen3 import Qwen3Dense4BConfig
from xtuner.v1.module.attention import MHAConfig


OFFICIAL_CHECKPOINT = Path(
    os.environ.get(
        "VIDEOCHAT3_DENSE_PATH",
        "/mnt/localssd/VideoChat3/VideoChat3-4B",
    )
)


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
    for hf_output, resumed_output in zip(hf_outputs, resumed_outputs, strict=True):
        torch.testing.assert_close(
            hf_output,
            resumed_output,
            rtol=2e-5,
            atol=2e-6,
        )
