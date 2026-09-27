from .configuration_videochat3 import VideoChat3Config, VideoChat3VisionConfig


class VideoChat3TASVisionConfig(VideoChat3VisionConfig):
    model_type = "videochat3_tas_vision"

    def __init__(self, tas_memory_tokens=0, tas_max_memory_tokens=4096,
                 tas_ema_init=0.1, tas_time_encoding=True, **kwargs):
        super().__init__(**kwargs)
        if self.temporal_merge_size != 4:
            raise ValueError("TAS currently uses the original four-frame chunks")
        if tas_memory_tokens < 0 or tas_max_memory_tokens < max(1, tas_memory_tokens):
            raise ValueError("Invalid TAS slot count/capacity")
        if not 0 < tas_ema_init < 1:
            raise ValueError("TAS EMA initialization must lie in (0, 1)")
        self.tas_memory_tokens = int(tas_memory_tokens)
        self.tas_max_memory_tokens = int(tas_max_memory_tokens)
        self.tas_ema_init = float(tas_ema_init)
        self.tas_time_encoding = bool(tas_time_encoding)


class VideoChat3TASConfig(VideoChat3Config):
    model_type = "videochat3_tas"
    sub_configs = {**VideoChat3Config.sub_configs, "vision_config": VideoChat3TASVisionConfig}

    def __init__(self, vision_config=None, **kwargs):
        vision = (VideoChat3TASVisionConfig(**(vision_config or {}))
                  if isinstance(vision_config, dict) or vision_config is None else vision_config)
        super().__init__(vision_config=vision, **kwargs)
