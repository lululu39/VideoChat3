"""Hugging Face processor for VideoChat3-LACT macro temporal outputs."""

from typing import Any

import torch

from .processing_videochat3 import VideoChat3Processor
from .videochat3_utils import VideoChat3VideoMetadata  # noqa: F401


class VideoChat3LACTProcessor(VideoChat3Processor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        macro_temporal_compression_factor: int = 1,
        macro_temporal_compression_mode: str = "auto",
        lact_chunk_query: bool = False,
        **kwargs: Any,
    ):
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
        self.macro_temporal_compression_factor = macro_temporal_compression_factor
        self.macro_temporal_compression_mode = macro_temporal_compression_mode
        self.lact_chunk_query = bool(lact_chunk_query)
        if self.lact_chunk_query and (
            macro_temporal_compression_factor != 1
            or macro_temporal_compression_mode != "auto"
        ):
            raise ValueError(
                "lact_chunk_query requires macro temporal compression factor 1 "
                "and mode 'auto'"
            )
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )

    def _calculate_timestamps(self, video_meta, temporal_merge_size: int = 4):
        timestamps = super()._calculate_timestamps(video_meta, temporal_merge_size)
        if self.lact_chunk_query:
            return timestamps
        factor = self.macro_temporal_compression_factor
        mode = self.macro_temporal_compression_mode
        if mode == "auto":
            mode = "mean"
        if mode == "video_last":
            return [timestamps[-1]]
        if factor == 1:
            return timestamps
        if mode == "select_last":
            return [
                timestamps[min(start + factor, len(timestamps)) - 1]
                for start in range(0, len(timestamps), factor)
            ]
        return [
            sum(timestamps[start : start + factor])
            / len(timestamps[start : start + factor])
            for start in range(0, len(timestamps), factor)
        ]

    def __call__(self, *args: Any, **kwargs: Any):
        outputs = super().__call__(*args, **kwargs)
        if not self.lact_chunk_query or "input_ids" not in outputs:
            return outputs

        input_ids = outputs["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            squeeze = input_ids.ndim == 1
            rows = input_ids.unsqueeze(0) if squeeze else input_ids
            keep_masks = [
                torch.tensor(
                    [
                        index == 0
                        or token.item() != self.video_token_id
                        or row[index - 1].item() != self.video_token_id
                        for index, token in enumerate(row)
                    ],
                    device=row.device,
                    dtype=torch.bool,
                )
                for row in rows
            ]
            aligned = {
                key: value.unsqueeze(0) if squeeze and value.ndim == 1 else value
                for key, value in outputs.items()
                if isinstance(value, torch.Tensor)
                and value.shape == input_ids.shape
            }
            max_length = max(int(mask.sum().item()) for mask in keep_masks)
            padding_side = getattr(self.tokenizer, "padding_side", "right")
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            for key, value in aligned.items():
                compressed_rows = []
                for row, mask in zip(value, keep_masks, strict=True):
                    compressed = row[mask]
                    pad_value = pad_token_id if key == "input_ids" else 0
                    padding = torch.full(
                        (max_length - compressed.shape[0],),
                        pad_value,
                        device=compressed.device,
                        dtype=compressed.dtype,
                    )
                    compressed_rows.append(
                        torch.cat((padding, compressed))
                        if padding_side == "left"
                        else torch.cat((compressed, padding))
                    )
                stacked = torch.stack(compressed_rows)
                outputs[key] = stacked.squeeze(0) if squeeze else stacked
            return outputs

        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            keep_masks = [
                [
                    index == 0
                    or token != self.video_token_id
                    or row[index - 1] != self.video_token_id
                    for index, token in enumerate(row)
                ]
                for row in input_ids
            ]
            for key, value in list(outputs.items()):
                if (
                    isinstance(value, list)
                    and len(value) == len(input_ids)
                    and all(isinstance(row, list) for row in value)
                    and all(len(row) == len(mask) for row, mask in zip(value, keep_masks, strict=True))
                ):
                    outputs[key] = [
                        [token for token, keep in zip(row, mask, strict=True) if keep]
                        for row, mask in zip(value, keep_masks, strict=True)
                    ]
        return outputs


__all__ = ["VideoChat3LACTProcessor"]
