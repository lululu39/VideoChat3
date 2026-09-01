"""Hugging Face processor for macro-compressed VideoChat3 outputs."""

from typing import Any

from .processing_videochat3 import VideoChat3Processor
from .videochat3_utils import VideoChat3VideoMetadata  # noqa: F401


class VideoChat3MacroProcessor(VideoChat3Processor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        macro_temporal_compression_factor: int = 1,
        **kwargs: Any,
    ):
        if macro_temporal_compression_factor not in (1, 2, 4, 8):
            raise ValueError(
                "macro_temporal_compression_factor must be one of (1, 2, 4, 8), "
                f"got {macro_temporal_compression_factor}"
            )
        self.macro_temporal_compression_factor = macro_temporal_compression_factor
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )

    def _calculate_timestamps(self, video_meta, temporal_merge_size: int = 4):
        timestamps = super()._calculate_timestamps(video_meta, temporal_merge_size)
        factor = self.macro_temporal_compression_factor
        if factor == 1:
            return timestamps
        return [
            sum(timestamps[start : start + factor])
            / len(timestamps[start : start + factor])
            for start in range(0, len(timestamps), factor)
        ]


__all__ = ["VideoChat3MacroProcessor"]
