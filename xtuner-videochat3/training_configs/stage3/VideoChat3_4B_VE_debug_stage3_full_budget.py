from pathlib import Path

from training_configs.stage3.VideoChat3_4B_LACT_VE_train_stage3 import (
    trainer as lact_trainer,
)
from xtuner.v1.config import LRConfig, VisionAdamWConfig
from xtuner.v1.model import VideoChat3Dense4BConfig
from xtuner.v1.train import ResumeConfig


# Diagnostic A/B: keep the original Stage 3 media budget but replace LACT
# with the pretrained VideoChat3 vision encoder. No checkpoints are saved.
trainer = lact_trainer.model_copy(deep=True)
trainer.model_cfg = VideoChat3Dense4BConfig(
    freeze_vision=False,
    freeze_projector=True,
    freeze_language=True,
)
trainer.load_from = Path("/mnt/localssd/VideoChat3/VideoChat3-4B")
trainer.tokenizer_path = trainer.load_from
trainer.work_dir = Path("work_dir/stage3/debug-vc3-original-vit-s3-full-budget-gb16")
trainer.exp_tracker = "jsonl"
trainer.wandb_config = None
trainer.resume_cfg = ResumeConfig(auto_resume=False)
trainer.debug_skip_save = True
trainer.global_batch_size = 16
trainer.total_epoch = None
trainer.total_step = 1
trainer.dataloader_cfg.pack_max_length = 16384 * 6

sample_max_length = 16384 * 6
for dataset_entry in trainer.dataloader_cfg.dataset_config_list:
    tokenize_fn = dataset_entry["tokenize_fn"]
    tokenize_fn.model_cfg = trainer.model_cfg
    tokenize_fn.processor_path = str(trainer.load_from)
    tokenize_fn.max_length = sample_max_length
    tokenize_fn.image_max_pixels = int(sample_max_length * 0.8 * 28 * 28)
    tokenize_fn.frame_max_pixels = 448 * 448
    tokenize_fn.video_max_total_pixels = int(sample_max_length * 0.8 * 4 * 28 * 28)
    tokenize_fn.video_max_frames = 3600

trainer.optim_cfg = VisionAdamWConfig(
    vit_lr=5e-6,
    projector_lr=2e-5,
    lr=2e-5,
    weight_decay=0.0,
    foreach=False,
)
trainer.lr_cfg = LRConfig(
    lr_type="cosine",
    warmup_ratio=0.03,
    lr_min=1e-6,
)
