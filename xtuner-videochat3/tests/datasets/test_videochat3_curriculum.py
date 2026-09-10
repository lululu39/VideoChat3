import copy
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from xtuner.v1.datasets import VideoChat3TokenizeFnConfig
from xtuner.v1.datasets.collator import videochat3_sft_collator
from xtuner.v1.datasets.videochat3_curriculum import (
    chunk_stage_plan, retained_chunk_indices, select_packed_video_chunks, stage_for_step,
)
from xtuner.v1.model.compose.videochat3.macro_temporal import compress_chunk_outputs

CHECKPOINT = Path("/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init")
REPO = Path(__file__).resolve().parents[3]


def test_balanced_stage_boundaries_and_resumed_step():
    plan = chunk_stage_plan(333)
    assert sum(s.stop - s.start for s in plan) == 333
    assert max(s.stop - s.start for s in plan) - min(s.stop - s.start for s in plan) <= 1
    assert plan[-1].factor == 128 and plan[-1].mode == "video_last"
    for stage in plan:
        assert stage_for_step(stage.start, 333) == stage
        assert stage_for_step(stage.stop - 1, 333) == stage
    with pytest.raises(ValueError):
        chunk_stage_plan(7)


def test_partial_tail_and_video_boundaries_keep_final_chunk():
    for stage in chunk_stage_plan(16):
        counts = [5, 112]
        chunks = [torch.tensor([float(i)], requires_grad=True) for i in range(sum(counts))]
        selected = compress_chunk_outputs(chunks, counts, stage.factor, stage.mode)
        expected = retained_chunk_indices(5, stage) + [5 + i for i in retained_chunk_indices(112, stage)]
        assert [int(x.item()) for x in selected] == expected
        sum(x.sum() for x in selected).backward()
        assert [i for i, x in enumerate(chunks) if x.grad is not None] == expected
    assert retained_chunk_indices(5, chunk_stage_plan(16)[1]) == [1, 3, 4]


@pytest.fixture(scope="module")
def tokenizer():
    if not (CHECKPOINT / "tokenizer.json").exists():
        pytest.skip("Pinned VideoChat3 tokenizer is not installed")
    return AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)


def tokenize_rows(tokenizer, stage):
    fn = VideoChat3TokenizeFnConfig(
        processor_path=str(CHECKPOINT), max_length=8192,
        video_read_type="decord", video_min_frames=32, video_max_frames=32,
        video_sample_fps=2, video_frame_multiple=4,
        frame_min_pixels=28 * 28, frame_max_pixels=56 * 56,
        video_max_total_pixels=32 * 56 * 56,
        macro_temporal_compression_factor=stage.factor,
        macro_temporal_compression_mode=stage.mode,
    ).build(tokenizer)
    rows = []
    # Short source-frame metadata deliberately creates an odd five-chunk tail.
    for frames in [20, 32]:
        raw = {"messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "tennis.mp4"},
             "video_metadata": {"total_num_frames": frames, "fps": 30.0, "duration": frames / 30,
                                "height": 1080, "width": 1920, "video_backend": "decord"}},
            {"type": "text", "text": "<VIDEO_CONTEXT>\nWhat happens first?"}]},
            {"role": "assistant", "content": "The player moves."},
            {"role": "user", "content": "What happens next?"},
            {"role": "assistant", "content": "The ball moves."}]}
        fn.state = "get_item"
        row = fn(copy.deepcopy(raw), media_root=str(REPO / "xtuner-videochat3/tests/resource"))
        fn.state = "cache"
        assert fn(copy.deepcopy(raw), media_root=str(REPO / "xtuner-videochat3/tests/resource"))["num_tokens"] == len(row["input_ids"])
        rows.append(row)
    return rows


def test_all_stage_pruning_matches_native_tokenization_collation_and_labels(tokenizer):
    plan = chunk_stage_plan(16)
    original = videochat3_sft_collator([tokenize_rows(tokenizer, plan[0])],
                                     pack_max_length=8192, padding_token_idx=tokenizer.pad_token_id)[0]
    config = json.loads((CHECKPOINT / "config.json").read_text())
    for stage in plan:
        pruned = select_packed_video_chunks(original, stage, tokenizer,
            vision_start_id=config["vision_start_token_id"], vision_end_id=config["vision_end_token_id"],
            video_token_id=config["video_token_id"])
        if stage.index == 1:
            assert pruned is original
            continue
        native = videochat3_sft_collator([tokenize_rows(tokenizer, stage)],
            pack_max_length=8192, padding_token_idx=tokenizer.pad_token_id, pack_to_max_length=False)[0]
        for name in ["input_ids", "position_ids", "cu_seq_lens_q", "cu_seq_lens_k", "image_grid_thw"]:
            torch.testing.assert_close(getattr(pruned["seq_ctx"], name), getattr(native["seq_ctx"], name), rtol=0, atol=0)
        torch.testing.assert_close(pruned["shifted_labels"], native["shifted_labels"], rtol=0, atol=0)
        assert pruned["seq_ctx"].pixel_values is original["seq_ctx"].pixel_values
        assert pruned["seq_ctx"].num_img_tokens == native["seq_ctx"].num_img_tokens


def test_linear_vision_curriculum_output_is_subset_of_unchanged_recurrent_forward():
    from xtuner.v1.model.compose.videochat3.videochat3_config import VideoChat3LACTVisionConfig
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = VideoChat3LACTVisionConfig(hidden_size=16, intermediate_size=32,
        num_attention_heads=4, num_hidden_layers=2, patch_size=2, merge_kernel_size=[2, 2],
        temporal_merge_size=4, init_pos_emb_height=2, init_pos_emb_width=2,
        attn_impl="eager_attention", memory_type="linear", inner_optim="delta", fw_num_heads=4,
        fw_order="parallel", lact_3d_rope=True, lact_gate_init=0.2,
        clip_state_grad_ratio=False).build().to(device=device, dtype=dtype)
    grid = torch.tensor([[20, 2, 2], [12, 2, 2]], device=device)
    pixels = torch.randn(128, 12, device=device, dtype=dtype)
    reference = torch.cat(model(pixel_values=pixels, grid_thws=grid))
    for stage in chunk_stage_plan(16):
        model.config.macro_temporal_compression_factor = stage.factor
        model.config.macro_temporal_compression_mode = stage.mode
        actual = torch.cat(model(pixel_values=pixels, grid_thws=grid))
        indices = retained_chunk_indices(5, stage) + [5 + i for i in retained_chunk_indices(3, stage)]
        torch.testing.assert_close(actual, reference[indices], rtol=0, atol=0)
        model.zero_grad(set_to_none=True)
        actual.square().sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
