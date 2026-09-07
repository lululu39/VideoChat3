"""Layout helpers for post-encoder temporal and spatial token selection.

The inputs to these helpers are the per-clip outputs produced by VideoChat3's
existing four-frame patch merger.  Macro compression is deliberately applied
only after the complete vision encoder and patch merger.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch


SUPPORTED_MACRO_TEMPORAL_COMPRESSION_FACTORS = (1, 2, 4, 8)
SUPPORTED_MACRO_TEMPORAL_COMPRESSION_MODES = (
    "auto",
    "mean",
    "select_last",
    "video_last",
    "chunk_select_last",
    "chunk_select_uniform",
)


def validate_macro_temporal_compression_factor(factor: int) -> int:
    factor = int(factor)
    if factor not in SUPPORTED_MACRO_TEMPORAL_COMPRESSION_FACTORS:
        raise ValueError(
            "macro_temporal_compression_factor must be one of "
            f"{SUPPORTED_MACRO_TEMPORAL_COMPRESSION_FACTORS}, got {factor}"
        )
    return factor


def validate_macro_temporal_compression_mode(mode: str) -> str:
    if mode not in SUPPORTED_MACRO_TEMPORAL_COMPRESSION_MODES:
        raise ValueError(
            "macro_temporal_compression_mode must be one of "
            f"{SUPPORTED_MACRO_TEMPORAL_COMPRESSION_MODES}, got {mode!r}"
        )
    return mode


def resolve_macro_temporal_compression_mode(
    mode: str,
    default: Literal["mean", "select_last"],
) -> Literal["mean", "select_last", "video_last", "chunk_select_last", "chunk_select_uniform"]:
    mode = validate_macro_temporal_compression_mode(mode)
    if mode == "auto":
        return default
    return mode  # type: ignore[return-value]


def video_clip_counts(grid_thws, temporal_merge_size: int) -> list[int]:
    """Return the number of existing temporal chunks for every input video."""
    if temporal_merge_size <= 0:
        raise ValueError(f"temporal_merge_size must be positive, got {temporal_merge_size}")
    counts = []
    for time, _, _ in grid_thws.tolist():
        if time <= 0:
            raise ValueError(f"Temporal grid size must be positive, got {time}")
        counts.append((int(time) + temporal_merge_size - 1) // temporal_merge_size)
    return counts


def macro_group_slices(clip_count: int, factor: int) -> list[tuple[int, int]]:
    """Return half-open macro groups for one video, including its short tail."""
    factor = validate_macro_temporal_compression_factor(factor)
    if clip_count <= 0:
        raise ValueError(f"clip_count must be positive, got {clip_count}")
    return [
        (start, min(start + factor, clip_count))
        for start in range(0, clip_count, factor)
    ]


def macro_clip_count(clip_count: int, factor: int, mode: str = "mean") -> int:
    mode = validate_macro_temporal_compression_mode(mode)
    if mode == "auto":
        mode = "mean"
    if clip_count <= 0:
        raise ValueError(f"clip_count must be positive, got {clip_count}")
    if mode in ("chunk_select_last", "chunk_select_uniform"):
        validate_macro_temporal_compression_factor(factor)
        return clip_count
    if mode == "video_last":
        validate_macro_temporal_compression_factor(factor)
        return 1
    return len(macro_group_slices(clip_count, factor))


def compress_chunk_outputs(
    chunk_outputs: list[torch.Tensor],
    video_clip_counts: Sequence[int],
    factor: int,
    mode: Literal["mean", "select_last", "video_last", "chunk_select_last", "chunk_select_uniform"],
) -> list[torch.Tensor]:
    """Compress per-chunk tensors without allowing groups to cross videos.

    ``mean`` averages equal spatial positions over a macro group.  The chunk
    tensors are expected to have identical shapes within a video.
    ``select_last`` keeps the last complete-encoder output in every group.
    ``video_last`` keeps one final chunk output per video after all chunks have
    already traversed the encoder.
    ``chunk_select_last`` keeps the last max(1, floor(S / factor)) merged
    spatial tokens of every chunk, preserving every chunk and timestamp.
    ``chunk_select_uniform`` keeps the same count at evenly spaced flattened
    spatial indices, including both endpoints (the center when keeping one).
    """
    factor = validate_macro_temporal_compression_factor(factor)
    if mode not in ("mean", "select_last", "video_last", "chunk_select_last", "chunk_select_uniform"):
        raise ValueError(f"Unsupported macro temporal compression mode: {mode}")
    if sum(video_clip_counts) != len(chunk_outputs):
        raise ValueError(
            f"video_clip_counts={list(video_clip_counts)} do not cover "
            f"{len(chunk_outputs)} chunk outputs"
        )
    # This is intentionally a true no-op so R=1 preserves the original list and
    # tensor objects, values, dtypes, and autograd graph exactly for normal R1.
    if factor == 1 and mode != "video_last":
        return chunk_outputs
    if mode == "chunk_select_uniform":
        if any(count <= 0 for count in video_clip_counts):
            raise ValueError("video_clip_counts must be positive")
        selected = []
        for chunk in chunk_outputs:
            size = chunk.shape[0]
            count = max(1, size // factor)
            indices = (
                torch.arange(count, device=chunk.device) * (size - 1) // (count - 1)
                if count > 1
                else torch.tensor([size // 2], device=chunk.device)
            )
            selected.append(chunk.index_select(0, indices))
        return selected
    if mode == "chunk_select_last":
        if any(count <= 0 for count in video_clip_counts):
            raise ValueError("video_clip_counts must be positive")
        return [chunk[-max(1, chunk.shape[0] // factor):] for chunk in chunk_outputs]

    compressed: list[torch.Tensor] = []
    offset = 0
    for clip_count in video_clip_counts:
        if clip_count <= 0:
            raise ValueError(f"clip_count must be positive, got {clip_count}")
        video_outputs = chunk_outputs[offset : offset + clip_count]
        if mode == "video_last":
            compressed.append(video_outputs[-1])
            offset += clip_count
            continue
        for start, end in macro_group_slices(clip_count, factor):
            group = video_outputs[start:end]
            if mode == "select_last":
                compressed.append(group[-1])
                continue
            reference_shape = group[0].shape
            if any(item.shape != reference_shape for item in group[1:]):
                raise ValueError(
                    "Chunk output shapes must match within a video for temporal mean pooling: "
                    f"{[tuple(item.shape) for item in group]}"
                )
            compressed.append(group[0] if len(group) == 1 else torch.stack(group).mean(dim=0))
        offset += clip_count
    return compressed


def compress_timestamps(
    timestamps: Sequence[float],
    factor: int,
    mode: str = "mean",
) -> list[float]:
    """Return the text timestamp placeholders matching temporal compression."""
    factor = validate_macro_temporal_compression_factor(factor)
    mode = validate_macro_temporal_compression_mode(mode)
    if not timestamps:
        raise ValueError("timestamps must not be empty")
    if mode == "auto":
        mode = "mean"
    if mode == "video_last":
        return [float(timestamps[-1])]
    if factor == 1 or mode in ("chunk_select_last", "chunk_select_uniform"):
        return list(timestamps)
    if mode == "select_last":
        return [
            float(timestamps[end - 1])
            for _, end in macro_group_slices(len(timestamps), factor)
        ]
    return [
        float(sum(timestamps[start:end]) / (end - start))
        for start, end in macro_group_slices(len(timestamps), factor)
    ]


def macro_video_token_count(
    grid_thw: Sequence[int],
    *,
    temporal_merge_size: int,
    spatial_merge_size: int,
    factor: int,
    mode: str = "mean",
) -> int:
    """Return the placeholder/projected-token count for one video."""
    time, height, width = (int(value) for value in grid_thw)
    if height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError(
            f"Spatial grid {(height, width)} is not divisible by merge size {spatial_merge_size}"
        )
    chunks = (time + temporal_merge_size - 1) // temporal_merge_size
    spatial_tokens = (height // spatial_merge_size) * (width // spatial_merge_size)
    if mode in ("chunk_select_last", "chunk_select_uniform"):
        factor = validate_macro_temporal_compression_factor(factor)
        spatial_tokens = max(1, spatial_tokens // factor)
    return macro_clip_count(chunks, factor, mode=mode) * spatial_tokens


__all__ = [
    "SUPPORTED_MACRO_TEMPORAL_COMPRESSION_FACTORS",
    "SUPPORTED_MACRO_TEMPORAL_COMPRESSION_MODES",
    "compress_chunk_outputs",
    "compress_timestamps",
    "macro_clip_count",
    "macro_group_slices",
    "macro_video_token_count",
    "resolve_macro_temporal_compression_mode",
    "validate_macro_temporal_compression_factor",
    "validate_macro_temporal_compression_mode",
    "video_clip_counts",
]
