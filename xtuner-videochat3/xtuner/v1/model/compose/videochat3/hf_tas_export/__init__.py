"""Standalone HF assets for TAS; reuse original VideoChat3 assets by value."""
import json
import shutil
from pathlib import Path


def prepare_tas_assets(source, target, **vision_overrides):
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and (path.suffix == ".py" or path.name in (
            "config.json", "processor_config.json", "preprocessor_config.json",
            "video_preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "chat_template.json", "chat_template.jinja",
            "generation_config.json", "vocab.json", "merges.txt", "added_tokens.json")):
            if path.resolve() != (target/path.name).resolve():
                shutil.copy2(path, target/path.name)
    config = json.loads((target/"config.json").read_text())
    config["model_type"] = "videochat3_tas"
    config["architectures"] = ["VideoChat3TASForConditionalGeneration"]
    config["auto_map"] = {
        "AutoConfig": "configuration_videochat3_tas.VideoChat3TASConfig",
        "AutoProcessor": "processing_videochat3_tas.VideoChat3TASProcessor",
        **{key: "modeling_videochat3_tas.VideoChat3TASForConditionalGeneration" for key in
           ("AutoModel", "AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForVision2Seq")},
    }
    vision = config["vision_config"]
    vision.update(model_type="videochat3_tas_vision", tas_memory_tokens=0,
                  tas_max_memory_tokens=4096, tas_ema_init=0.1, tas_time_encoding=True)
    vision.update(vision_overrides)
    (target/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    processor = json.loads((target/"processor_config.json").read_text())
    processor.update(processor_class="VideoChat3TASProcessor",
                     tas_memory_tokens=vision["tas_memory_tokens"],
                     tas_max_memory_tokens=vision["tas_max_memory_tokens"],
                     auto_map={"AutoProcessor": config["auto_map"]["AutoProcessor"]})
    (target/"processor_config.json").write_text(json.dumps(processor, indent=2)+"\n")
    for path in Path(__file__).parent.glob("*.py"):
        if path.name != "__init__.py":
            shutil.copy2(path, target/path.name)


def refresh_tas_code(target):
    for path in Path(__file__).parent.glob("*.py"):
        if path.name != "__init__.py":
            shutil.copy2(path, Path(target)/path.name)
