from typing import Any

from transformers import AutoConfig

from .configuration_videochat3 import VideoChat3Config, VideoChat3VisionConfig


class VideoChat3MacroVisionConfig(VideoChat3VisionConfig):
    model_type = "videochat3_macro_vision"

    def __init__(self, macro_temporal_compression_factor: int = 1, **kwargs: Any):
        super().__init__(**kwargs)
        if macro_temporal_compression_factor not in (1, 2, 4, 8):
            raise ValueError(
                "macro_temporal_compression_factor must be one of (1, 2, 4, 8), "
                f"got {macro_temporal_compression_factor}"
            )
        self.macro_temporal_compression_factor = macro_temporal_compression_factor


class VideoChat3MacroConfig(VideoChat3Config):
    model_type = "videochat3_macro"
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": VideoChat3MacroVisionConfig,
    }

    def __init__(self, vision_config=None, **kwargs: Any):
        if isinstance(vision_config, dict):
            vision_config = VideoChat3MacroVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = VideoChat3MacroVisionConfig()
        elif not isinstance(vision_config, VideoChat3MacroVisionConfig):
            raise TypeError(
                "vision_config must be a VideoChat3MacroVisionConfig or dict, "
                f"got {type(vision_config)}"
            )
        super().__init__(vision_config=vision_config, **kwargs)


__all__ = ["VideoChat3MacroConfig", "VideoChat3MacroVisionConfig"]
