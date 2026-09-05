import math
from typing import Any

from transformers import AutoConfig

from .configuration_videochat3 import VideoChat3Config, VideoChat3VisionConfig


class VideoChat3LACTVisionConfig(VideoChat3VisionConfig):
    model_type = "videochat3_lact_vision"

    def __init__(
        self,
        memory_type: str = "swiglu",
        fw_inter_multi: float = 2.0,
        fw_num_heads: int = 1,
        fw_base_lr: float = 0.01,
        fw_muon_update_steps: int = 5,
        inner_optim: str = "muon",
        fw_share_proj: bool = False,
        fw_share_init: bool = True,
        fw_order: str = "serial",
        fw_norm_epsilon: float = 1e-5,
        clip_ns_grad_ratio: bool = False,
        recompute_ns5_backward: bool = True,
        clip_state_grad_ratio: bool = True,
        fw_update_layer_group_size: int = 1,
        macro_temporal_compression_factor: int = 1,
        macro_temporal_compression_mode: str = "auto",
        lact_inference_state_mode: str = "continuous",
        lact_3d_rope: bool = False,
        lact_chunk_query: bool = False,
        lact_chunk_query_mode: str = "single",
        lact_gate: str = "linear",
        lact_gate_init: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if self.temporal_merge_size != 4:
            raise ValueError(f"VideoChat3-LACT requires temporal_merge_size=4, got {self.temporal_merge_size}")
        if memory_type not in ("swiglu", "linear"):
            raise ValueError(
                f"memory_type must be 'swiglu' or 'linear', got {memory_type!r}"
            )
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
        if fw_order not in ("serial", "parallel"):
            raise ValueError(
                f"fw_order must be 'serial' or 'parallel', got {fw_order!r}"
            )
        if fw_order == "parallel" and fw_share_proj:
            raise ValueError("fw_order='parallel' requires private FW projections")
        if memory_type == "linear" and fw_share_proj:
            raise ValueError("linear memory currently requires private projections")
        if fw_base_lr <= 0:
            raise ValueError("fw_base_lr must be positive")
        if fw_muon_update_steps < 0:
            raise ValueError("fw_muon_update_steps must be non-negative")
        if inner_optim not in ("muon", "delta", "sgd"):
            raise ValueError(
                "inner_optim must be 'muon', 'delta', or the legacy 'sgd' "
                f"alias, got {inner_optim!r}"
            )
        if fw_norm_epsilon <= 0:
            raise ValueError("fw_norm_epsilon must be positive")
        if fw_update_layer_group_size <= 0:
            raise ValueError("fw_update_layer_group_size must be positive")
        if memory_type == "linear" and fw_update_layer_group_size != 1:
            raise ValueError(
                "linear memory currently requires fw_update_layer_group_size=1"
            )
        if lact_chunk_query and (
            macro_temporal_compression_factor != 1
            or macro_temporal_compression_mode != "auto"
        ):
            raise ValueError(
                "lact_chunk_query requires macro temporal compression factor 1 "
                "and mode 'auto'"
            )
        if lact_chunk_query_mode not in ("single", "spatial_quarter"):
            raise ValueError(
                "lact_chunk_query_mode must be 'single' or 'spatial_quarter', "
                f"got {lact_chunk_query_mode!r}"
            )
        if not lact_chunk_query and lact_chunk_query_mode != "single":
            raise ValueError(
                "lact_chunk_query_mode requires lact_chunk_query=True"
            )
        if macro_temporal_compression_factor not in (1, 2, 4, 8):
            raise ValueError(
                "macro_temporal_compression_factor must be one of (1, 2, 4, 8), "
                f"got {macro_temporal_compression_factor}"
            )
        if macro_temporal_compression_mode not in (
            "auto",
            "mean",
            "select_last",
            "video_last",
        ):
            raise ValueError(
                "macro_temporal_compression_mode must be one of "
                "('auto', 'mean', 'select_last', 'video_last'), got "
                f"{macro_temporal_compression_mode!r}"
            )
        if lact_inference_state_mode not in ("continuous", "reset_state"):
            raise ValueError(
                "lact_inference_state_mode must be 'continuous' or 'reset_state', "
                f"got {lact_inference_state_mode!r}"
            )
        if lact_gate not in ("linear", "tanh"):
            raise ValueError(
                f"lact_gate must be 'linear' or 'tanh', got {lact_gate!r}"
            )
        if not math.isfinite(lact_gate_init):
            raise ValueError("lact_gate_init must be finite")
        self.memory_type = memory_type
        self.fw_inter_multi = fw_inter_multi
        self.fw_num_heads = fw_num_heads
        self.fw_base_lr = fw_base_lr
        self.fw_muon_update_steps = fw_muon_update_steps
        self.inner_optim = inner_optim
        self.fw_share_proj = fw_share_proj
        self.fw_share_init = fw_share_init
        self.fw_order = fw_order
        self.fw_norm_epsilon = fw_norm_epsilon
        self.clip_ns_grad_ratio = clip_ns_grad_ratio
        self.recompute_ns5_backward = recompute_ns5_backward
        self.clip_state_grad_ratio = clip_state_grad_ratio
        self.fw_update_layer_group_size = fw_update_layer_group_size
        self.macro_temporal_compression_factor = macro_temporal_compression_factor
        self.macro_temporal_compression_mode = macro_temporal_compression_mode
        self.lact_inference_state_mode = lact_inference_state_mode
        self.lact_3d_rope = lact_3d_rope
        self.lact_chunk_query = bool(lact_chunk_query)
        self.lact_chunk_query_mode = lact_chunk_query_mode
        self.lact_gate = lact_gate
        self.lact_gate_init = float(lact_gate_init)


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
