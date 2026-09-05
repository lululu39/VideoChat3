import json
import shutil
from pathlib import Path
from typing import Any


MACRO_CONFIG_NAME = "configuration_videochat3_macro.py"
MACRO_MODELING_NAME = "modeling_videochat3_macro.py"
MACRO_PROCESSING_NAME = "processing_videochat3_macro.py"


def export_macro_hf_artifacts(hf_dir: str | Path, model_config: Any) -> None:
    """Make a macro-compressed Base save self-describing and loadable."""
    hf_dir = Path(hf_dir)
    config_path = hf_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Hugging Face config: {config_path}")

    config = json.loads(config_path.read_text())
    config["model_type"] = "videochat3_macro"
    config["architectures"] = ["VideoChat3MacroForConditionalGeneration"]
    config.setdefault("auto_map", {}).update(
        {
            "AutoProcessor": "processing_videochat3_macro.VideoChat3MacroProcessor",
            "AutoConfig": "configuration_videochat3_macro.VideoChat3MacroConfig",
            "AutoModel": "modeling_videochat3_macro.VideoChat3MacroForConditionalGeneration",
            "AutoModelForCausalLM": "modeling_videochat3_macro.VideoChat3MacroForConditionalGeneration",
            "AutoModelForImageTextToText": "modeling_videochat3_macro.VideoChat3MacroForConditionalGeneration",
            "AutoModelForVision2Seq": "modeling_videochat3_macro.VideoChat3MacroForConditionalGeneration",
        }
    )
    vision_config = config.setdefault("vision_config", {})
    vision_config["model_type"] = "videochat3_macro_vision"
    vision_config["macro_temporal_compression_factor"] = (
        model_config.vision_config.macro_temporal_compression_factor
    )
    vision_config["macro_temporal_compression_mode"] = (
        model_config.vision_config.macro_temporal_compression_mode
    )
    vision_config["chunk_query"] = model_config.vision_config.chunk_query
    vision_config["chunk_query_mode"] = model_config.vision_config.chunk_query_mode
    temporary_config_path = config_path.with_suffix(".json.tmp")
    temporary_config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )
    temporary_config_path.replace(config_path)

    processor_config_path = hf_dir / "processor_config.json"
    if not processor_config_path.is_file():
        raise FileNotFoundError(
            f"Missing Hugging Face processor config: {processor_config_path}"
        )
    processor_config = json.loads(processor_config_path.read_text())
    processor_config["processor_class"] = "VideoChat3MacroProcessor"
    processor_config["macro_temporal_compression_factor"] = (
        model_config.vision_config.macro_temporal_compression_factor
    )
    processor_config["macro_temporal_compression_mode"] = (
        model_config.vision_config.macro_temporal_compression_mode
    )
    processor_config["chunk_query"] = model_config.vision_config.chunk_query
    processor_config["chunk_query_mode"] = (
        model_config.vision_config.chunk_query_mode
    )
    processor_config.setdefault("auto_map", {})["AutoProcessor"] = (
        "processing_videochat3_macro.VideoChat3MacroProcessor"
    )
    temporary_processor_config_path = processor_config_path.with_suffix(".json.tmp")
    temporary_processor_config_path.write_text(
        json.dumps(processor_config, indent=2, ensure_ascii=False) + "\n"
    )
    temporary_processor_config_path.replace(processor_config_path)

    source_dir = Path(__file__).parent
    for file_name in (
        MACRO_CONFIG_NAME,
        MACRO_MODELING_NAME,
        MACRO_PROCESSING_NAME,
    ):
        shutil.copy2(source_dir / file_name, hf_dir / file_name)


__all__ = ["export_macro_hf_artifacts"]
