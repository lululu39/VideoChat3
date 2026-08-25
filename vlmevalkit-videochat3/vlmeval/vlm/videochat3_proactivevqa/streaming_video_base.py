"""Abstract base class for online streaming video models in VLMEvalKit.

These models run a stateful streaming loop inside generate_inner, returning a
JSON-encoded record list as the prediction string.  VLMEvalKit's standard
task scheduling, per-rank pkl resume, and result aggregation all work
unchanged.

Dataset (ProactiveVideoQA) puts all streaming metadata into the message as a
text item with a special prefix; this module extracts it.

Sampling vs inference
---------------------
``target_fps`` controls frame sampling density.  ``infer_fps`` (default 1.0)
controls how often the model is invoked.  When ``target_fps > infer_fps``,
each inference step packs ``target_fps / infer_fps`` frames into one turn,
while records / time tags remain on the ``infer_fps`` grid (typically 1s).
"""
from __future__ import annotations

import json
import math
from abc import abstractmethod

from ..base import BaseModel

# Must match the prefix used in ProactiveVideoQA.build_prompt
STREAM_META_PREFIX = '__PROACTIVE_STREAM_META__'
RESPONSE_TAG = '</Response>'


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def _normalise_time(value: float):
    value = round(float(value), 6)
    nearest_int = round(value)
    if abs(value - nearest_int) < 1e-9:
        return int(nearest_int)
    return value


def _format_time(value: float) -> str:
    value = _normalise_time(value)
    if isinstance(value, int):
        return str(value)
    return f'{value:.6f}'.rstrip('0').rstrip('.')


def _round_span(round_idx: int, fps: float):
    start = round_idx / fps
    end = (round_idx + 1) / fps
    return _normalise_time(start), _normalise_time(end)


def _format_time_tag(round_idx: int, fps: float) -> str:
    start, end = _round_span(round_idx, fps)
    return f'<{_format_time(start)}s-{_format_time(end)}s>'


def _frame_span(round_idx: int, frame_idx: int, n_frames: int, infer_fps: float):
    """Wall-clock span of one packed frame inside an inference round.

    Example (infer_fps=1, n_frames=2, round_idx=0):
        frame 0 -> [0, 0.5), frame 1 -> [0.5, 1).
    """
    if n_frames < 1:
        raise ValueError(f'n_frames must be >= 1, got {n_frames}')
    if frame_idx < 0 or frame_idx >= n_frames:
        raise ValueError(f'frame_idx out of range: {frame_idx} for n_frames={n_frames}')
    round_start = round_idx / infer_fps
    frame_dur = (1.0 / infer_fps) / n_frames
    start = round_start + frame_idx * frame_dur
    end = start + frame_dur
    return _normalise_time(start), _normalise_time(end)


def _format_frame_time_tag(
    round_idx: int, frame_idx: int, n_frames: int, infer_fps: float
) -> str:
    start, end = _frame_span(round_idx, frame_idx, n_frames, infer_fps)
    return f'<{_format_time(start)}s-{_format_time(end)}s>'


def _parse_raw_answer(raw: str):
    """Map a raw tagged answer to (answerable, model_response).

    Returns ('Yes', content) for </Response>..., ('No', None) otherwise.
    """
    s = (raw or '').strip()
    if s.startswith(RESPONSE_TAG):
        content = s[len(RESPONSE_TAG):].lstrip(' \t:\n')
        return 'Yes', content
    return 'No', None


def _extract_stream_meta(message: list) -> dict:
    """Find and decode the stream_meta dict embedded in a message list."""
    for item in message:
        if item.get('type') == 'text' and isinstance(item.get('value'), str):
            if item['value'].startswith(STREAM_META_PREFIX):
                return json.loads(item['value'][len(STREAM_META_PREFIX):])
    raise ValueError(
        'No stream_meta found in message — was this dataset built with ProactiveVideoQA?'
    )


def _effective_fps(model, meta: dict) -> float:
    """Sampling fps (frame extraction density)."""
    fps = getattr(model, 'target_fps', None)
    if fps is None:
        fps = meta.get('target_fps', 1.0)
    fps = float(fps)
    if fps <= 0:
        raise ValueError(f'target_fps must be positive, got {fps}')
    return fps


def _effective_infer_fps(model) -> float:
    """Inference fps (how often the model is called). Defaults to 1.0."""
    fps = getattr(model, 'infer_fps', None)
    if fps is None:
        fps = 1.0
    fps = float(fps)
    if fps <= 0:
        raise ValueError(f'infer_fps must be positive, got {fps}')
    return fps


def _frames_per_step(sample_fps: float, infer_fps: float) -> int:
    """How many sampled frames are packed into one inference turn."""
    if sample_fps < infer_fps - 1e-9:
        raise ValueError(
            f'target_fps ({sample_fps}) must be >= infer_fps ({infer_fps})'
        )
    ratio = sample_fps / infer_fps
    n = int(round(ratio))
    if n < 1 or abs(ratio - n) > 1e-6:
        raise ValueError(
            f'target_fps / infer_fps must be a positive integer, '
            f'got {sample_fps} / {infer_fps} = {ratio}'
        )
    return n


def _compute_stream_rounds(meta: dict, fps: float) -> int:
    """Compute how many inference steps to run at the given fps grid."""
    answer = meta.get('answer') or []
    duration = meta.get('duration')
    if duration is None or not answer:
        return max(1, int(meta['target_rounds']))

    duration = float(duration)
    last_reply_end = float(answer[-1]['reply_timespan'][1])
    by_frames = int(duration * fps)
    by_reply = int(math.ceil(min(duration, last_reply_end) * fps))
    return max(1, min(by_frames, by_reply))


def _as_frame_list(frames):
    if isinstance(frames, (list, tuple)):
        return list(frames)
    return [frames]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class StreamingVideoModel(BaseModel):
    """Abstract base for online streaming video models.

    Sub-classes must implement:
        reset_session(question, extra_turns)  — initialise a fresh session
        step(frames, round_idx) -> str        — process one infer turn (1+ frames)
        _open_extractor(video_path)           — return VideoFrameExtractor-compatible obj

    The full streaming loop runs inside generate_inner; VLMEvalKit's scheduling,
    per-rank pkl resume, and result aggregation are not affected.

    Prompt convention
    -----------------
    The ProactiveVideoQA dataset builds a two-element message:
        [{'type': 'video', 'value': video_path},
         {'type': 'text',  'value': '__PROACTIVE_STREAM_META__<json>'}]
    The ``video`` entry is only a progress-tracker placeholder; frames are read
    inside ``generate_inner`` via ``meta['video_path']``. Do not run
    ``BaseModel.preproc_content`` on it (missing files on S3 mount would fail
    the ``mime is None => type must be text`` assert).
    """

    VIDEO_LLM = True
    INTERLEAVE = True

    def __init__(self):
        super().__init__()

    def generate(self, message, dataset=None):
        """Skip file-based preproc; streaming loop opens the video itself."""
        assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], (
            f'Invalid input type: {message}'
        )
        if self.check_content(message) == 'str':
            message = [dict(type='text', value=message)]
        elif self.check_content(message) == 'dict':
            message = [message]
        elif self.check_content(message) == 'liststr':
            message = [dict(type='text', value=s) for s in message]
        for item in message:
            assert item.get('type') in self.allowed_types, f'Invalid input type: {item.get("type")}'
            assert 'value' in item
        return self.generate_inner(message, dataset)

    def generate_inner(self, message: list, dataset=None) -> str:
        """Run the streaming loop; return JSON string of per-step records."""
        meta = _extract_stream_meta(message)
        sample_fps = _effective_fps(self, meta)
        infer_fps = _effective_infer_fps(self)
        frames_per_step = _frames_per_step(sample_fps, infer_fps)
        target_rounds = _compute_stream_rounds(meta, infer_fps)

        self.reset_session(
            question=meta['question'],
            extra_turns=meta['extra_turns'],
        )

        records = []
        extractor = self._open_extractor(meta['video_path'])
        try:
            for r in range(target_rounds):
                frames = []
                try:
                    for _ in range(frames_per_step):
                        frames.append(extractor.get_frame_at_round(r))
                except StopIteration:
                    if not frames:
                        break
                raw = self.step(frames, round_idx=r)
                answerable, content = _parse_raw_answer(raw)
                records.append({
                    'video_span': list(_round_span(r, infer_fps)),
                    'answerable': answerable,
                    'model_response': content,
                    'raw_answer': raw,
                })
        finally:
            extractor.close()

        return json.dumps({
            'question_id': meta['qid'],
            'question': meta['question'],
            'records': records,
        }, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def reset_session(self, question: str, extra_turns: list):
        """Initialise / reset the streaming session for a new video.

        Args:
            question: The main user question string.
            extra_turns: List of {'time': float, 'content': str} dicts for
                additional user turns (e.g. TV subtitles).
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, frames, round_idx: int) -> str:
        """Process one inference turn (one or more frames) and return raw tag.

        ``frames`` may be a single PIL Image or a list of Images packed into
        the same user turn (when target_fps > infer_fps).

        Returns one of:
            '</Silence>'
            '</Standby>'
            '</Response><text content>'
        """
        raise NotImplementedError

    @abstractmethod
    def _open_extractor(self, video_path: str):
        """Return a VideoFrameExtractor-compatible object.

        The returned object must support:
            extractor.get_frame_at_round(r)  -> frame (raises StopIteration on end)
            extractor.close()
        """
        raise NotImplementedError
