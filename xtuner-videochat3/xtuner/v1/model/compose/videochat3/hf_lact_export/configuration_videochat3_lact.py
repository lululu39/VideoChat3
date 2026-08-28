from typing import Any

from transformers import AutoConfig

from .configuration_videochat3 import VideoChat3Config, VideoChat3VisionConfig


class VideoChat3LACTVisionConfig(VideoChat3VisionConfig):
    model_type = "videochat3_lact_vision"

    def __init__(
        self,
        fw_inter_multi: float = 2.0,
        fw_num_heads: int = 1,
        fw_base_lr: float = 0.01,
        fw_muon_update_steps: int = 5,
        fw_share_proj: bool = False,
        fw_share_init: bool = True,
        fw_norm_epsilon: float = 1e-5,
        clip_ns_grad_ratio: bool = True,
        clip_state_grad_ratio: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if self.temporal_merge_size != 4:
            raise ValueError(f"VideoChat3-LACT requires temporal_merge_size=4, got {self.temporal_merge_size}")
        if fw_inter_multi <= 0:
            raise ValueError("fw_inter_multi must be positive")
        if fw_num_heads <= 0 or self.hidden_size % fw_num_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by the positive fw_num_heads={fw_num_heads}"
            )
        if fw_share_proj and fw_share_init:
            raise ValueError(
                "fw_share_init only applies to private projections; set it to False when fw_share_proj=True"
            )
        if fw_base_lr <= 0:
            raise ValueError("fw_base_lr must be positive")
        if fw_muon_update_steps < 0:
            raise ValueError("fw_muon_update_steps must be non-negative")
        if fw_norm_epsilon <= 0:
            raise ValueError("fw_norm_epsilon must be positive")
        self.fw_inter_multi = fw_inter_multi
        self.fw_num_heads = fw_num_heads
        self.fw_base_lr = fw_base_lr
        self.fw_muon_update_steps = fw_muon_update_steps
        self.fw_share_proj = fw_share_proj
        self.fw_share_init = fw_share_init
        self.fw_norm_epsilon = fw_norm_epsilon
        self.clip_ns_grad_ratio = clip_ns_grad_ratio
        self.clip_state_grad_ratio = clip_state_grad_ratio


class VideoChat3LACTConfig(VideoChat3Config):
    model_type = "videochat3_lact"
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": VideoChat3LACTVisionConfig,
    }

    def __init__(self, vision_config=None, **kwargs: Any):
        if isinstance(vision_config, dict):
            vision_config = VideoChat3LACTVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = VideoChat3LACTVisionConfig()
        elif not isinstance(vision_config, VideoChat3LACTVisionConfig):
            raise TypeError(f"vision_config must be a VideoChat3LACTVisionConfig or dict, got {type(vision_config)}")
        super().__init__(vision_config=vision_config, **kwargs)


__all__ = ["VideoChat3LACTConfig", "VideoChat3LACTVisionConfig"]
