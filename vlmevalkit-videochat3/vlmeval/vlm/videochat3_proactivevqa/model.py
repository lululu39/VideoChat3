"""VideoChat3 online streaming model for ProactiveVideoQA."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .streaming_video_base import (
    StreamingVideoModel,
    _as_frame_list,
    _frame_span,
)
from .utils import smart_video_resize


def _make_proactive_session_class(StreamingSession):
    class ProactiveStreamingSession(StreamingSession):
        def __init__(
            self, *args, extra_turns=None, target_fps=1.0, infer_fps=1.0, **kwargs
        ):
            super().__init__(*args, **kwargs)
            self._extra_turns = list(extra_turns or [])
            self._target_fps = float(target_fps)
            self._infer_fps = float(infer_fps)

        def _user_content(
            self,
            frames,
            round_idx: int,
            include_question: bool,
            frame_max_pixels: int | list[int] | None = None,
        ):
            # Base already emits per-frame tags (<0s-0.5s>, img, ...); only
            # inject subtitle / extra-turn text for this inference round.
            base = super()._user_content(
                frames, round_idx, include_question, frame_max_pixels
            )
            frames = self._as_frames(frames)
            n_frames = len(frames)
            infer_fps = self._infer_fps

            # Attach each extra turn after the image/time-tag pair covering
            # its actual timestamp, rather than after the whole packed turn.
            hits_by_frame: dict[int, list[str]] = {i: [] for i in range(n_frames)}
            for turn in self._extra_turns:
                if not turn.get("content"):
                    continue
                timestamp = float(turn["time"])
                for frame_idx in range(n_frames):
                    start, end = _frame_span(
                        round_idx, frame_idx, n_frames, infer_fps
                    )
                    if start < timestamp <= end:
                        hits_by_frame[frame_idx].append(turn["content"])
                        break

            merged = []
            frame_idx = -1
            for item in base:
                merged.append(item)
                if item.get("type") == "text":
                    frame_idx += 1
                    hits = hits_by_frame[frame_idx]
                    if hits:
                        merged.append({"type": "text", "text": "\n".join(hits)})
            return merged

    return ProactiveStreamingSession


class VideoChat3ProactiveVQA(StreamingVideoModel):
    INSTALL_REQ = False

    def __init__(
        self,
        model_path: str = None,
        max_rounds: int = 32,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.8,
        top_k: int = 20,
        target_fps: float = 1.0,
        infer_fps: float = 1.0,
        global_question: bool = True,
        attn_implementation: str = "flash_attention_2",
        device: str = "auto",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        encode_frame_size: tuple[int, int] | None = None,
        encode_frame_size_standby: tuple[int, int] | None = None,
        standby_high_res_frames: int = 1,  # #frames (not rounds) after </Standby>
        force_resize: bool = True,
        debug: bool = False,
        debug_log_path: str | None = None,
        debug_max_text_chars: int = 4000,
        factor: int = 28,
        **kwargs,  # absorb any extra VLMEvalKit kwargs; NOT forwarded to the engine
    ):
        super().__init__()
        self.model_path = model_path
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.target_fps = target_fps
        self.infer_fps = float(infer_fps)
        self.global_question = bool(global_question)
        self.standby_high_res_frames = int(standby_high_res_frames)
        self.force_resize = bool(force_resize)
        self.factor = factor
        self._standby_high_res_remaining = 0

        _default_max_pixels = int(os.environ.get("MAX_PIXELS", 100352))
        if encode_frame_size is not None:
            self._frame_max_pixels = int(np.prod(encode_frame_size))
        else:
            self._frame_max_pixels = max_pixels or _default_max_pixels

        if encode_frame_size_standby is not None:
            self._frame_max_pixels_standby = int(np.prod(encode_frame_size_standby))
        else:
            # 放大一倍：线性尺寸 ×2 → 像素数 ×4
            self._frame_max_pixels_standby = self._frame_max_pixels * 4

        print(
            f"[VideoChat3ProactiveVQA] frame_max_pixels={self._frame_max_pixels}, "
            f"frame_max_pixels_standby={self._frame_max_pixels_standby}, "
            f"standby_high_res_frames={self.standby_high_res_frames}, "
            f"target_fps={self.target_fps}, infer_fps={self.infer_fps}",
            flush=True,
        )

        from .inference_fast import (
            SYSTEM,
            StreamingSession,
            VideoChat3StreamEngine,
            VideoFrameExtractor,
        )

        self._SYSTEM = SYSTEM
        self._VideoFrameExtractor = VideoFrameExtractor
        self._ProactiveStreamingSession = _make_proactive_session_class(StreamingSession)
        self._session = None

        print(
            f"[VideoChat3ProactiveVQA] Loading engine from: {model_path}",
            flush=True,
        )
        # Only pass engine-known params; extra VLMEvalKit kwargs are intentionally dropped.
        # Processor global cap must cover standby frames.
        engine_max_pixels = max(self._frame_max_pixels, self._frame_max_pixels_standby)
        if max_pixels is not None:
            engine_max_pixels = max(engine_max_pixels, max_pixels)

        self._engine = VideoChat3StreamEngine(
            model_path,
            device=device,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=engine_max_pixels,
            debug=debug,
            debug_log_path=debug_log_path,
            debug_max_text_chars=debug_max_text_chars,
        )

    def _resize_frame(self, frame: Image.Image, high_res: bool) -> Image.Image:
        w, h = frame.size
        max_px = (
            self._frame_max_pixels_standby if high_res else self._frame_max_pixels
        )
        h_bar, w_bar = smart_video_resize(
            num_frames=1,
            height=h,
            width=w,
            factor=self.factor,
            frame_min_pixels=self.factor*self.factor,
            frame_max_pixels=max_px,
            force_resize=self.force_resize,
        )
        return frame.resize((w_bar, h_bar))

    def reset_session(self, question: str, extra_turns: list):
        self._standby_high_res_remaining = 0
        self._session = self._ProactiveStreamingSession(
            engine=self._engine,
            system=self._SYSTEM,
            question=question,
            question_time=0,
            max_rounds=self.max_rounds,
            global_question=self.global_question,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            extra_turns=extra_turns,
            target_fps=self.target_fps,
            infer_fps=self.infer_fps,
        )

    def step(self, frames, round_idx: int) -> str:
        assert self._session is not None, "reset_session must be called before step"
        frames = _as_frame_list(frames)

        # standby_high_res_frames counts frames (not rounds): after </Standby>,
        # the next N frames (in temporal order, possibly spanning rounds) are high-res.
        high_res_flags = []
        resized = []
        for f in frames:
            high_res = self._standby_high_res_remaining > 0
            if high_res:
                self._standby_high_res_remaining -= 1
            high_res_flags.append(high_res)
            resized.append(self._resize_frame(f, high_res))
        frames = resized

        # Session still takes one max_pixels for the turn; use standby cap if any
        # frame in this turn is high-res so the processor does not downscale them.
        any_high = any(high_res_flags)
        frame_max_pixels = (
            self._frame_max_pixels_standby if any_high else self._frame_max_pixels
        )

        print(
            f"[Budget INFO] high_res_flags={high_res_flags}, "
            f"frame_max_pixels={frame_max_pixels}, n_frames={len(frames)}, "
            f"remaining={self._standby_high_res_remaining}",
            flush=True,
        )

        raw = self._session.step(
            frames, round_idx=round_idx, frame_max_pixels=frame_max_pixels
        )

        if "</Standby>" in raw:
            self._standby_high_res_remaining = self.standby_high_res_frames

        return raw

    def _open_extractor(self, video_path: str):
        return self._VideoFrameExtractor(video_path, target_fps=self.target_fps)
