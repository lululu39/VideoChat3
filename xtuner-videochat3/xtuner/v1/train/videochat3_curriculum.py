"""Continuous-optimizer TimeLens curriculum with fixed all-token data packing."""
import json
from dataclasses import asdict

from xtuner.v1.datasets.videochat3_curriculum import (
    chunk_stage_plan, select_packed_video_chunks, stage_for_step,
)
from xtuner.v1.train.trainer import Trainer


class VideoChat3CurriculumTrainer(Trainer):
    def _activate_video_stage(self, stage):
        if getattr(self, "_active_video_stage", None) == stage:
            return
        configs = [self._trainer_cfg.model_cfg]
        configs.extend(getattr(m, "config", None) for m in self._engine.model.modules())
        for config in configs:
            vision = getattr(config, "vision_config", config)
            if vision is not None and hasattr(vision, "macro_temporal_compression_factor"):
                if getattr(vision, "lact_chunk_query", False):
                    raise ValueError("This curriculum does not support query tokens")
                vision.macro_temporal_compression_factor = stage.factor
                vision.macro_temporal_compression_mode = stage.mode
        self._active_video_stage = stage
        self.logger.info(f"Video curriculum stage {stage.index}/8: steps {stage.start + 1}-{stage.stop}, "
                         f"factor={stage.factor}, mode={stage.mode}; all input frames and fixed packs retained")
        if self.rank == 0:
            (self.exp_dir / "video_curriculum.json").write_text(json.dumps(
                {"total_steps": self.total_step, "stages": [asdict(s) for s in chunk_stage_plan(self.total_step)]},
                indent=2) + "\n")

    def _data_iter(self):
        # Base dataloader/prefetch always emits the same all-token representation.
        # The completed optimizer step, restored from DCP, is the only stage clock.
        for batch in super()._data_iter():
            stage = stage_for_step(self.cur_step, self.total_step)
            self._activate_video_stage(stage)
            config = self._trainer_cfg.model_cfg
            yield [select_packed_video_chunks(
                data, stage, self.tokenizer, vision_start_id=config.vision_start_token_id,
                vision_end_id=config.vision_end_token_id, video_token_id=config.video_token_id)
                for data in batch]

    def _log_step(self, **kwargs):
        stage = self._active_video_stage
        kwargs["loss_log"]["curriculum_stage"] = stage.index
        kwargs["loss_log"]["curriculum_chunk_stride"] = stage.factor
        kwargs["loss_log"]["curriculum_video_last"] = int(stage.mode == "video_last")
        super()._log_step(**kwargs)

    def _stage_boundary(self):
        return self.cur_step in {s.stop for s in chunk_stage_plan(self.total_step)}

    def _maybe_save_hf(self):
        previous = self._hf_interval
        if previous is not None and self._stage_boundary():
            self._hf_interval = 1
        try:
            return super()._maybe_save_hf()
        finally:
            self._hf_interval = previous

    def _maybe_save(self, is_snapshot=False):
        previous = self._checkpoint_interval
        if not is_snapshot and previous is not None and self._stage_boundary():
            self._checkpoint_interval = 1
        try:
            return super()._maybe_save(is_snapshot=is_snapshot)
        finally:
            self._checkpoint_interval = previous
