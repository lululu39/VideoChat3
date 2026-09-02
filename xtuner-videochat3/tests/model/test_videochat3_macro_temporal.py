import torch


def _identity_compile(fn=None, *args, **kwargs):
    if fn is None:
        return lambda inner: inner
    return fn


torch.compile = _identity_compile

from xtuner.v1.model.compose.videochat3.macro_temporal import (  # noqa: E402
    compress_chunk_outputs,
    compress_timestamps,
    macro_clip_count,
    macro_video_token_count,
    resolve_macro_temporal_compression_mode,
)


def test_video_last_keeps_one_final_chunk_per_video():
    chunk_outputs = [torch.tensor([[float(idx)]]) for idx in range(5)]

    compressed = compress_chunk_outputs(
        chunk_outputs,
        video_clip_counts=[3, 2],
        factor=4,
        mode="video_last",
    )

    assert [item.item() for item in compressed] == [2.0, 4.0]


def test_video_last_token_and_timestamp_counts_ignore_macro_groups():
    assert compress_timestamps([0.5, 1.5, 2.5, 3.5, 4.5], 4, mode="video_last") == [4.5]
    assert macro_clip_count(5, 4, mode="video_last") == 1
    assert (
        macro_video_token_count(
            (20, 4, 6),
            temporal_merge_size=4,
            spatial_merge_size=2,
            factor=4,
            mode="video_last",
        )
        == 6
    )


def test_existing_auto_modes_preserve_base_and_lact_defaults():
    timestamps = [0.5, 1.5, 2.5, 3.5, 4.5]

    assert resolve_macro_temporal_compression_mode("auto", default="mean") == "mean"
    assert resolve_macro_temporal_compression_mode("auto", default="select_last") == "select_last"
    assert compress_timestamps(timestamps, 4) == [2.0, 4.5]
    assert compress_timestamps(timestamps, 4, mode="select_last") == [3.5, 4.5]
