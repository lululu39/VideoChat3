import os
import runpy
from pathlib import Path

_base = runpy.run_path(str(Path(__file__).with_name("VideoChat3_4B_LACT_VE_train_stage3.py")))
trainer = _base["trainer"]
trainer.work_dir = Path(os.environ["VIDEOCHAT3_CURRICULUM_WORK_DIR"])
trainer.hf_interval = 200
trainer.checkpoint_interval = 200
trainer.hf_max_keep = 10
trainer.checkpoint_maxkeep = 2
trainer.wandb_config.group = "timelens-multi_stage_video_last"
trainer.wandb_config.tags.extend(["multi_stage_video_last", "llava-hf800-init", "fixed-all-token-packing"])
