import torch
import json
import copy
import pytest
from types import SimpleNamespace


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
from xtuner.v1.model.compose.videochat3.hf_macro_export import (  # noqa: E402
    export_macro_hf_artifacts,
)
from xtuner.v1.model.compose.videochat3.videochat3_config import (  # noqa: E402
    VideoChat3VisionConfig,
    VideoChat3LACTVisionConfig,
)
from xtuner.v1.datasets.mllm_tokenize_fn.videochat3_tokenize_fn import (  # noqa: E402
    VideoChat3TokenizeFunction,
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


def test_chunk_select_last_keeps_spatial_tails_and_their_gradients():
    chunks = [torch.randn(size, 4, 2, requires_grad=True) for size in (64, 64, 64, 64, 6, 1)]
    outputs = compress_chunk_outputs(chunks, [4, 2], 4, mode="chunk_select_last")
    assert [item.shape[0] for item in outputs] == [16, 16, 16, 16, 1, 1]
    for original, selected in zip(chunks, outputs, strict=True):
        torch.testing.assert_close(selected, original[-selected.shape[0]:], rtol=0, atol=0)
    sum(item.sum() for item in outputs).backward()
    for original, selected in zip(chunks, outputs, strict=True):
        expected = torch.zeros_like(original)
        expected[-selected.shape[0]:] = 1
        torch.testing.assert_close(original.grad, expected, rtol=0, atol=0)
    assert compress_chunk_outputs(chunks, [4, 2], 1, mode="chunk_select_last") is chunks
    timestamps = [0.5, 1.5, 2.5, 3.5, 4.0]
    assert compress_timestamps(timestamps, 4, mode="chunk_select_last") == timestamps
    assert macro_clip_count(5, 4, mode="chunk_select_last") == 5


@pytest.mark.parametrize("factor", [1, 2, 4, 8])
def test_chunk_select_last_counts_all_chunks_including_short_tails(factor):
    for frames in (1, 4, 9, 64, 224, 448):
        for height, width in ((2, 2), (4, 6), (6, 6), (12, 16), (16, 16)):
            actual = macro_video_token_count(
                (frames, height, width), temporal_merge_size=4,
                spatial_merge_size=2, factor=factor, mode="chunk_select_last",
            )
            assert actual == ((frames + 3) // 4) * max(1, (height * width // 4) // factor)


def test_chunk_select_last_matches_query_placeholder_layout():
    from xtuner.v1.data_proto.messages import ChatMessages

    query = VideoChat3TokenizeFunction.__new__(VideoChat3TokenizeFunction)
    query.lact_chunk_query = True
    query.lact_chunk_query_mode = "spatial_quarter"
    query.macro_temporal_compression_factor = 1
    query.macro_temporal_compression_mode = "mean"
    query.video_processor = SimpleNamespace(temporal_merge_size=4, merge_size=2)
    query.media_processor = SimpleNamespace(_calculate_timestamps=lambda meta, size: meta)
    query._video_meta_list = [[1.5, 5.5, 8.0], [1.5]]
    query.chat_template = SimpleNamespace(
        image_start_token="<v>", image_end_token="</v>", video_context_token="V",
    )
    selected = copy.copy(query)
    selected.lact_chunk_query = False
    selected.lact_chunk_query_mode = "single"
    selected.macro_temporal_compression_factor = 4
    selected.macro_temporal_compression_mode = "chunk_select_last"
    grids = [torch.tensor([9, 4, 8]), torch.tensor([4, 2, 2])]
    messages = ChatMessages(messages=[{
        "role": "user", "content": [{"type": "text", "text": "<VIDEO_CONTEXT> and <VIDEO_CONTEXT>"}],
    }])
    reference = copy.deepcopy(messages)
    query._replace_video_token(reference, grids)
    selected._replace_video_token(messages, grids)
    assert messages.model_dump() == reference.model_dump()
    for grid in grids:
        assert selected._get_number_of_video_tokens(grid) == query._get_number_of_video_tokens(grid)


@pytest.mark.parametrize("config_cls", [VideoChat3VisionConfig, VideoChat3LACTVisionConfig])
def test_chunk_select_last_is_only_post_encoder_selection(config_cls):
    kwargs = dict(
        hidden_size=16, intermediate_size=32, num_attention_heads=4,
        num_hidden_layers=2, patch_size=2, merge_kernel_size=[2, 2],
        temporal_merge_size=4, init_pos_emb_height=4, init_pos_emb_width=8,
        attn_impl="eager_attention",
    )
    if config_cls is VideoChat3LACTVisionConfig:
        kwargs.update(memory_type="linear", inner_optim="delta", fw_num_heads=4,
                      fw_order="parallel", lact_gate_init=0.1, clip_state_grad_ratio=False)
    model = config_cls(**kwargs).build()
    selected = copy.deepcopy(model)
    selected.config.macro_temporal_compression_mode = "chunk_select_last"
    selected.config.macro_temporal_compression_factor = 4
    assert selected.chunk_query is None
    assert model.state_dict().keys() == selected.state_dict().keys()
    grids = torch.tensor([[9, 4, 8], [4, 2, 2]], dtype=torch.int32)
    pixels = torch.randn(304, 12)
    original = model(pixels, grids)
    actual = selected(pixels, grids)
    expected = [chunk[-max(1, chunk.shape[0] // 4):] for chunk in original]
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.cat(actual).square().mean().backward()
    torch.cat(expected).square().mean().backward()
    for (name, parameter), (_, reference) in zip(selected.named_parameters(), model.named_parameters(), strict=True):
        assert "chunk_query" not in name
        if reference.grad is not None:
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)


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


def test_chunk_query_uses_one_placeholder_per_existing_four_frame_chunk():
    tokenize_fn = VideoChat3TokenizeFunction.__new__(VideoChat3TokenizeFunction)
    tokenize_fn.lact_chunk_query = True
    tokenize_fn.lact_chunk_query_mode = "single"
    tokenize_fn.video_processor = SimpleNamespace(temporal_merge_size=4, merge_size=2)

    assert tokenize_fn._get_number_of_video_tokens((9, 16, 16)) == 3

    tokenize_fn.lact_chunk_query_mode = "spatial_quarter"
    assert tokenize_fn._get_number_of_video_tokens((9, 16, 16)) == 48


def test_base_macro_export_records_video_last_mode(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"vision_config": {}}), encoding="utf-8"
    )
    (tmp_path / "processor_config.json").write_text("{}", encoding="utf-8")
    model_config = SimpleNamespace(
        vision_config=SimpleNamespace(
            macro_temporal_compression_factor=4,
            macro_temporal_compression_mode="video_last",
            chunk_query=False,
            chunk_query_mode="single",
        )
    )

    export_macro_hf_artifacts(tmp_path, model_config)

    saved_config = json.loads((tmp_path / "config.json").read_text())
    saved_processor = json.loads((tmp_path / "processor_config.json").read_text())
    assert saved_config["vision_config"]["macro_temporal_compression_mode"] == "video_last"
    assert saved_processor["macro_temporal_compression_mode"] == "video_last"


def test_base_spatial_quarter_query_returns_multiple_summaries_per_chunk():
    torch.manual_seed(37)
    model = VideoChat3VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        patch_size=2,
        merge_kernel_size=[2, 2],
        temporal_merge_size=4,
        init_pos_emb_height=4,
        init_pos_emb_width=8,
        attn_impl="eager_attention",
        chunk_query=True,
        chunk_query_mode="spatial_quarter",
    ).build()
    grid_thws = torch.tensor([[8, 4, 8]], dtype=torch.int32)
    outputs = model(torch.randn(256, 12), grid_thws)

    assert len(outputs) == 4
    assert all(output.shape == (1, 4, 16) for output in outputs)
    assert model.chunk_query.shape == (16, 16)
    assert not any("memory" in name for name, _ in model.named_parameters())
    torch.cat(outputs).square().mean().backward()
    assert model.chunk_query.grad is not None
    assert torch.count_nonzero(model.chunk_query.grad[:2]).item() > 0


def test_base_query_macro_export_and_processor_alignment(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"vision_config": {}}), encoding="utf-8"
    )
    (tmp_path / "processor_config.json").write_text("{}", encoding="utf-8")
    model_config = SimpleNamespace(
        vision_config=SimpleNamespace(
            macro_temporal_compression_factor=1,
            macro_temporal_compression_mode="auto",
            chunk_query=True,
            chunk_query_mode="spatial_quarter",
        )
    )
    export_macro_hf_artifacts(tmp_path, model_config)
    saved_config = json.loads((tmp_path / "config.json").read_text())
    saved_processor = json.loads((tmp_path / "processor_config.json").read_text())
    assert saved_config["vision_config"]["chunk_query"] is True
    assert saved_config["vision_config"]["chunk_query_mode"] == "spatial_quarter"
    assert saved_processor["chunk_query"] is True
    assert saved_processor["chunk_query_mode"] == "spatial_quarter"
