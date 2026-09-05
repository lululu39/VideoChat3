import torch
import torch.nn as nn
import math

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


def _interleave_chunk_queries(
    hidden_states: torch.Tensor,
    patch_lengths: list[int],
    query_bank: torch.Tensor,
    query_counts: list[int],
) -> tuple[torch.Tensor, list[int]]:
    outputs = []
    query_indices = []
    patch_offset = 0
    packed_offset = 0
    for length, query_count in zip(patch_lengths, query_counts, strict=True):
        outputs.append(hidden_states[patch_offset : patch_offset + length])
        outputs.append(query_bank[:query_count])
        query_indices.extend(
            range(packed_offset + length, packed_offset + length + query_count)
        )
        patch_offset += length
        packed_offset += length + query_count
    return torch.cat(outputs, dim=0), query_indices


def _interleave_identity_rope(
    rope_freqs_cis: torch.Tensor,
    patch_lengths: list[int],
    query_counts: list[int],
) -> torch.Tensor:
    identity = torch.ones(
        1,
        rope_freqs_cis.shape[-1],
        device=rope_freqs_cis.device,
        dtype=rope_freqs_cis.dtype,
    )
    outputs = []
    offset = 0
    for length, query_count in zip(patch_lengths, query_counts, strict=True):
        outputs.append(rope_freqs_cis[offset : offset + length])
        outputs.append(identity.expand(query_count, -1))
        offset += length
    return torch.cat(outputs, dim=0)


def _query_counts(grid_thws, merge_kernel_size, mode):
    merge_height, merge_width = merge_kernel_size
    counts = []
    for _, height, width in grid_thws.tolist():
        if mode == "single":
            counts.append(1)
        else:
            spatial_tokens = (int(height) // merge_height) * (
                int(width) // merge_width
            )
            counts.append(max(1, spatial_tokens // 4))
    return counts


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

    def __init__(self, config: VideoChat3MacroVisionConfig):
        super().__init__(config)
        self.chunk_query = (
            nn.Parameter(
                torch.empty(
                    (16, config.hidden_size)
                    if config.chunk_query_mode == "spatial_quarter"
                    else (config.hidden_size,)
                )
            )
            if config.chunk_query
            else None
        )
        if self.chunk_query is not None:
            nn.init.trunc_normal_(self.chunk_query, std=0.02)

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor):
        if self.chunk_query is not None:
            split_grid_thws = self.split_grid_thws_clip_by_clip(grid_thws)
            hidden_states = self.patch_embed(pixel_values, split_grid_thws)
            patch_lengths = (
                split_grid_thws[:, 0]
                * split_grid_thws[:, 1]
                * split_grid_thws[:, 2]
            ).tolist()
            query_counts = _query_counts(
                split_grid_thws,
                self.config.merge_kernel_size,
                self.config.chunk_query_mode,
            )
            hidden_states, query_indices = _interleave_chunk_queries(
                hidden_states,
                patch_lengths,
                self.chunk_query,
                query_counts,
            )
            rope_freqs_cis = self.encoder.rope_2d.get_freqs_cis(
                grid_thws=split_grid_thws
            )
            rope_freqs_cis = _interleave_identity_rope(
                rope_freqs_cis,
                patch_lengths,
                query_counts,
            )
            offsets = [0]
            for length, query_count in zip(
                patch_lengths,
                query_counts,
                strict=True,
            ):
                offsets.append(offsets[-1] + int(length) + query_count)
            cu_seqlens = torch.tensor(
                offsets,
                device=hidden_states.device,
                dtype=torch.int32,
            )
            for block in self.encoder.blocks:
                hidden_states = block(
                    hidden_states,
                    cu_seqlens,
                    rope_freqs_cis=rope_freqs_cis,
                )
            hidden_states = self.encoder.final_layernorm(hidden_states)
            merge_area = math.prod(self.config.merge_kernel_size)
            return [
                hidden_states[index]
                .reshape(1, 1, -1)
                .expand(1, merge_area, -1)
                for index in query_indices
            ]
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
