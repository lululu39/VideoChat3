"""Step-indexed chunk selection over an unchanged all-token packed dataset."""
from dataclasses import dataclass
import re

import torch

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.model.compose.videochat3.macro_temporal import macro_video_token_count


@dataclass(frozen=True)
class ChunkStage:
    index: int
    start: int  # Number of completed optimizer steps at stage entry.
    stop: int
    factor: int
    mode: str


def chunk_stage_plan(total_steps: int) -> list[ChunkStage]:
    if total_steps < 8:
        raise ValueError("The eight-stage curriculum requires at least eight steps")
    quotient, remainder = divmod(total_steps, 8)
    result, start = [], 0
    for index, factor in enumerate((1, 2, 4, 8, 16, 32, 64, 128)):
        stop = start + quotient + int(index < remainder)
        result.append(ChunkStage(index + 1, start, stop, factor,
                                 "video_last" if index == 7 else "select_last"))
        start = stop
    return result


def stage_for_step(completed_steps: int, total_steps: int) -> ChunkStage:
    if not 0 <= completed_steps < total_steps:
        raise ValueError("completed_steps is outside the training horizon")
    return next(s for s in chunk_stage_plan(total_steps) if s.start <= completed_steps < s.stop)


def retained_chunk_indices(count: int, stage: ChunkStage) -> list[int]:
    if count <= 0:
        raise ValueError("A video must contain at least one chunk")
    if stage.mode == "video_last":
        return [count - 1]
    return [min(start + stage.factor, count) - 1 for start in range(0, count, stage.factor)]


def select_packed_video_chunks(data: dict, stage: ChunkStage, tokenizer, *,
                               vision_start_id: int, vision_end_id: int, video_token_id: int) -> dict:
    """Prune LM placeholder/timestamp blocks after prefetch, before the GPU forward.

    Vision pixels and grids are deliberately untouched: every frame still
    participates in the encoder's recurrent computation. Rebuild LM positions,
    packed sequence boundaries and shifted targets, preserving the final EOS
    target that the original collator removed from input_ids.
    """
    if stage.factor == 1 and stage.mode != "video_last":
        return data
    ctx = data["seq_ctx"]
    if ctx.image_grid_thw is None:
        return data
    if ctx.input_ids.device.type != "cpu" or ctx.sequence_parallel_mesh is not None:
        raise ValueError("Chunk curriculum must run on CPU before sequence-parallel splitting")
    ids = ctx.input_ids[0]
    labels = data["shifted_labels"]
    real_length = ids.numel() - ctx.num_padding
    starts = torch.where(ids[:real_length] == vision_start_id)[0].tolist()
    ends = torch.where(ids[:real_length] == vision_end_id)[0].tolist()
    counts = [(int(t) + 3) // 4 for t, _, _ in ctx.image_grid_thw.tolist()]
    if len(starts) != len(ends) or len(starts) != sum(counts):
        raise ValueError("All-token video placeholders do not match the original grids")
    keep = torch.ones(ids.numel(), dtype=torch.bool)
    keep[real_length:] = False
    lt_ids = tokenizer.encode("<", add_special_tokens=False)
    lt_id = lt_ids[0] if len(lt_ids) == 1 else None
    offset = 0
    for (time, height, width), count in zip(ctx.image_grid_thw.tolist(), counts):
        selected = set(retained_chunk_indices(count, stage))
        spatial = int(height) * int(width) // 4
        for chunk in range(count):
            pos = offset + chunk
            start, end = starts[pos], ends[pos]
            if end - start - 1 != spatial or not torch.all(ids[start + 1:end] == video_token_id):
                raise ValueError("Expected an uncompressed full spatial grid in each video chunk")
            if chunk in selected:
                continue
            lower = max(ends[pos - 1] + 1 if pos else 0, start - 16)
            candidates = [i for i in range(lower, start) if lt_id is None or int(ids[i]) == lt_id]
            timestamp_start = None
            for i in candidates:
                text = tokenizer.decode(ids[i:start].tolist(), clean_up_tokenization_spaces=False)
                if re.fullmatch(r"<\d+\.\d seconds>", text):
                    timestamp_start = i
                    break
            if timestamp_start is None:
                raise ValueError("Cannot locate the exact token-aligned timestamp before a video chunk")
            if timestamp_start < 1 or not torch.all(labels[0, timestamp_start - 1:end] == -100):
                raise ValueError("Refusing to remove supervised tokens")
            keep[timestamp_start:end + 1] = False
        offset += count
    positions = torch.where(keep)[0]
    # The target for each retained input is the label of the next retained input.
    target_positions = torch.cat((positions[1:] - 1, positions[-1:]))
    shifted = labels.index_select(1, target_positions)
    boundaries = ctx.cu_seq_lens_q.tolist()
    if ctx.num_padding:
        boundaries = boundaries[:-1]
    lengths = [int(keep[a:b].sum()) for a, b in zip(boundaries[:-1], boundaries[1:])]
    if any(n <= 0 for n in lengths):
        raise ValueError("Chunk selection removed an entire packed example")
    cu = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).int()
    vision_mask = keep & ((ids == video_token_id) | (ids == vision_start_id) | (ids == vision_end_id))
    # This field is per packed example and includes its vision start/end markers.
    num_img_tokens = [int(vision_mask[a:b].sum()) for a, b in zip(boundaries[:-1], boundaries[1:])]
    selected_tokens = [macro_video_token_count(grid, temporal_merge_size=4, spatial_merge_size=2,
                                              factor=stage.factor, mode=stage.mode)
                       for grid in ctx.image_grid_thw.tolist()]
    result = SequenceContext(input_ids=ctx.input_ids.index_select(1, positions),
                             cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                             max_length_q=max(lengths), max_length_k=max(lengths), num_padding=0,
                             pixel_values=ctx.pixel_values, image_grid_thw=ctx.image_grid_thw,
                             num_img_tokens=num_img_tokens)
    if int((result.input_ids == video_token_id).sum()) != sum(selected_tokens):
        raise ValueError("Selected LM tokens do not match selected vision feature counts")
    if int((shifted != -100).sum()) != int((labels != -100).sum()):
        raise ValueError("Chunk selection changed the number of supervised targets")
    return {"seq_ctx": result, "shifted_labels": shifted}
