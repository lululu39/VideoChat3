import torch
import torch.nn as nn

from transformers import AutoModel

from .configuration_videochat3_macro import (
    VideoChat3MacroConfig,
    VideoChat3MacroVisionConfig,
)
from .modeling_videochat3 import (
    VideoChat3ForConditionalGeneration,
    VideoChat3Model,
    VideoChat3MultiModalProjector,
    VideoChat3PreTrainedModel,
    VideoChat3VisionModel,
)


def _compress_chunk_outputs(
    chunk_outputs: list[torch.Tensor],
    video_clip_counts: list[int],
    factor: int,
    mode: str,
) -> list[torch.Tensor]:
    if factor not in (1, 2, 4, 8):
        raise ValueError(f"Unsupported macro temporal compression factor: {factor}")
    if sum(video_clip_counts) != len(chunk_outputs):
        raise ValueError(
            f"video_clip_counts={video_clip_counts} do not cover "
            f"{len(chunk_outputs)} outputs"
        )
    if mode == "auto":
        mode = "mean"
    if mode not in ("mean", "select_last", "video_last"):
        raise ValueError(f"Unsupported macro temporal compression mode: {mode}")
    if factor == 1 and mode != "video_last":
        return chunk_outputs
    compressed = []
    offset = 0
    for clip_count in video_clip_counts:
        video_outputs = chunk_outputs[offset : offset + clip_count]
        if mode == "video_last":
            compressed.append(video_outputs[-1])
            offset += clip_count
            continue
        for start in range(0, clip_count, factor):
            group = video_outputs[start : min(start + factor, clip_count)]
            if mode == "select_last":
                compressed.append(group[-1])
            else:
                compressed.append(
                    group[0]
                    if len(group) == 1
                    else torch.stack(group).mean(dim=0)
                )
        offset += clip_count
    return compressed


class VideoChat3MacroVisionModel(VideoChat3VisionModel):
    config_class = VideoChat3MacroVisionConfig

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor):
        chunk_outputs = super().forward(pixel_values=pixel_values, grid_thws=grid_thws)
        temporal_merge_size = self.config.temporal_merge_size
        video_clip_counts = [
            (int(time) + temporal_merge_size - 1) // temporal_merge_size
            for time, _, _ in grid_thws.tolist()
        ]
        return _compress_chunk_outputs(
            chunk_outputs,
            video_clip_counts,
            self.config.macro_temporal_compression_factor,
            getattr(self.config, "macro_temporal_compression_mode", "auto"),
        )


class VideoChat3MacroModel(VideoChat3Model):
    config_class = VideoChat3MacroConfig

    def __init__(self, config: VideoChat3MacroConfig):
        VideoChat3PreTrainedModel.__init__(self, config)
        self.vision_tower = VideoChat3MacroVisionModel._from_config(config.vision_config)
        self.multi_modal_projector = VideoChat3MultiModalProjector(config)
        self.language_model = AutoModel.from_config(
            config.text_config,
            trust_remote_code=True,
        )
        self.post_init()


class VideoChat3MacroForConditionalGeneration(VideoChat3ForConditionalGeneration):
    config_class = VideoChat3MacroConfig

    def __init__(self, config: VideoChat3MacroConfig):
        VideoChat3PreTrainedModel.__init__(self, config)
        self.model = VideoChat3MacroModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()


__all__ = [
    "VideoChat3MacroForConditionalGeneration",
    "VideoChat3MacroModel",
    "VideoChat3MacroVisionModel",
]
