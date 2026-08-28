import json
import shutil
from pathlib import Path
from typing import Any


LACT_CONFIG_NAME = "configuration_videochat3_lact.py"
LACT_MODELING_NAME = "modeling_videochat3_lact.py"


def export_lact_hf_artifacts(hf_dir: str | Path, model_config: Any) -> None:
    """Make an XTuner HF save self-describing as a VideoChat3-LACT model."""
    hf_dir = Path(hf_dir)
    config_path = hf_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Hugging Face config: {config_path}")

    config = json.loads(config_path.read_text())
    config["model_type"] = "videochat3_lact"
    config["architectures"] = ["VideoChat3LACTForConditionalGeneration"]
    auto_map = config.setdefault("auto_map", {})
    auto_map.update(
        {
            "AutoConfig": ("configuration_videochat3_lact.VideoChat3LACTConfig"),
            "AutoModel": "modeling_videochat3_lact.VideoChat3LACTForConditionalGeneration",
            "AutoModelForCausalLM": ("modeling_videochat3_lact.VideoChat3LACTForConditionalGeneration"),
            "AutoModelForImageTextToText": ("modeling_videochat3_lact.VideoChat3LACTForConditionalGeneration"),
            "AutoModelForVision2Seq": ("modeling_videochat3_lact.VideoChat3LACTForConditionalGeneration"),
        }
    )

    vision_config = model_config.vision_config
    exported_vision_fields = (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_hidden_layers",
        "patch_size",
        "merge_kernel_size",
        "temporal_patch_size",
        "temporal_merge_size",
        "init_pos_emb_height",
        "init_pos_emb_width",
        "fw_inter_multi",
        "fw_num_heads",
        "fw_base_lr",
        "fw_muon_update_steps",
        "fw_share_proj",
        "fw_share_init",
        "fw_norm_epsilon",
        "clip_ns_grad_ratio",
    )
    hf_vision_config = config.setdefault("vision_config", {})
    hf_vision_config["model_type"] = "videochat3_lact_vision"
    for field in exported_vision_fields:
        hf_vision_config[field] = getattr(vision_config, field)
    hf_vision_config["dtype"] = vision_config.torch_dtype
    hf_vision_config["attn_impl"] = {"eager_attention": "eager"}.get(vision_config.attn_impl, vision_config.attn_impl)

    text_hf_config = model_config.text_config.hf_config
    if text_hf_config is not None:
        config["text_config"] = text_hf_config.to_dict()
    for field in (
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ):
        config[field] = getattr(model_config, field)

    temporary_config_path = config_path.with_suffix(".json.tmp")
    temporary_config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    temporary_config_path.replace(config_path)

    source_dir = Path(__file__).parent
    for file_name in (LACT_CONFIG_NAME, LACT_MODELING_NAME):
        shutil.copy2(source_dir / file_name, hf_dir / file_name)


__all__ = ["export_lact_hf_artifacts"]
