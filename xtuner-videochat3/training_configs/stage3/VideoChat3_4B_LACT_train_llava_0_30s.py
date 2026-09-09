"""LLaVA short-video adaptation using the existing Stage-3 model/optimizer."""
import os
import runpy
from pathlib import Path

_config = runpy.run_path(str(Path(__file__).with_name("VideoChat3_4B_LACT_VE_train_stage3.py")))
trainer = _config["trainer"]
trainer.work_dir = Path(os.environ["VIDEOCHAT3_LLAVA_WORK_DIR"])
trainer.global_batch_size = int(os.getenv("VIDEOCHAT3_GLOBAL_BATCH_SIZE", "16"))
trainer.total_epoch = int(os.getenv("VIDEOCHAT3_TOTAL_EPOCH", "1"))
max_steps = int(os.getenv("VIDEOCHAT3_MAX_STEPS", "0"))
if max_steps:
    if max_steps < 2:
        raise ValueError("VIDEOCHAT3_MAX_STEPS must be zero (epoch mode) or at least two")
    trainer.total_epoch = None
    trainer.total_step = max_steps
    # The shared scheduler interprets values >= 1 as an explicit warmup-step count.
    trainer.lr_cfg.warmup_ratio = max(1, int(0.03 * max_steps))
trainer.hf_interval = int(os.getenv("VIDEOCHAT3_HF_INTERVAL", "20"))
trainer.checkpoint_interval = int(os.getenv("VIDEOCHAT3_CHECKPOINT_INTERVAL", "20"))
trainer.wandb_config.group = "videochat3-llava-0-30s-adaptation"
if os.getenv("VIDEOCHAT3_LLAVA_SMOKE") == "1":
    trainer.total_epoch = None
    trainer.total_step = 3
    trainer.debug_skip_save = True
    trainer.exp_tracker = "jsonl"
    trainer.resume_cfg.auto_resume = False
    trainer.lr_cfg.warmup_ratio = 0.0
    trainer.lr_cfg.lr_type = "constant"
