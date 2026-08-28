import json
import os
from pathlib import Path

from xtuner.v1.config import FSDPConfig, LRConfig, VisionAdamWConfig
from xtuner.v1.datasets import VideoChat3TokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import VideoChat3LACTDense4BConfig
from xtuner.v1.train import ResumeConfig, TrainerConfig, WandbConfig


run_name = os.getenv(
    "WANDB_NAME",
    "vc3-4b-lact-fw4-ve-s3-full89k-8xh100-gb16-f128-s16k-vitlr2p5e6-v3",
)
model_path = Path("/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init")
metadata_path = Path(
    "/mnt/localssd/dataset/VideoChat3/VideoChat3-Stage3-Training-Data/VideoChat3_Stage3_Training_Data_full_local.json"
)
work_dir = Path("work_dir/stage3") / run_name
cache_dir = Path("dataset_cache/cache_videochat3_4B_lact_stage3")

model_cfg = VideoChat3LACTDense4BConfig()

sample_max_length = 16384
pack_max_length = 16384
frame_max_pixels = 224 * 224
video_max_total_pixels = 128 * frame_max_pixels
video_max_frames = 128
reference_global_batch_size = 128
reference_vit_lr = 8e-5 / 4
global_batch_size = 16
total_epoch = 1
hf_interval = 5000
hf_max_keep = 1
checkpoint_interval = 100
checkpoint_maxkeep = 1

vit_lr = reference_vit_lr * global_batch_size / reference_global_batch_size
lr = vit_lr
weight_decay = 0.0
warmup_ratio = 0.03
lr_min = 1e-6 * global_batch_size / reference_global_batch_size
recompute_ratio = 1.0
loss_reduction = "square"

dataset_collections = json.loads(metadata_path.read_text())
dataset_config = []
for name, data in dataset_collections.items():
    dataset_config.append(
        {
            "dataset": DatasetConfig(
                name=name,
                anno_path=data["anno_path"],
                media_root=data.get("media_root", ""),
                sample_ratio=data.get("sample_ratio", 1.0),
                class_name="VLMJsonlDataset",
                cache_dir=cache_dir,
            ),
            "tokenize_fn": VideoChat3TokenizeFnConfig(
                model_cfg=model_cfg,
                max_length=sample_max_length,
                image_min_pixels=data.get("image_min_pixels", 28 * 28),
                image_max_pixels=data.get(
                    "image_max_pixels",
                    int(sample_max_length * 0.8 * 28 * 28),
                ),
                frame_min_pixels=data.get("frame_min_pixels", 28 * 28),
                frame_max_pixels=min(
                    data.get("frame_max_pixels", frame_max_pixels),
                    frame_max_pixels,
                ),
                video_max_total_pixels=min(
                    data.get("video_max_total_pixels", video_max_total_pixels),
                    video_max_total_pixels,
                ),
                video_min_frames=data.get("video_min_frames", 1),
                video_max_frames=min(
                    data.get("video_max_frames", video_max_frames),
                    video_max_frames,
                ),
                fixed_num_sampled_frames=data.get("fixed_num_sampled_frames"),
                video_sample_fps=data.get("video_sample_fps", 2),
                processor_path=str(model_path),
                data_augment=data.get("data_augment", False),
                system_message=data.get("system_message"),
                hash=data.get("hash"),
                oss_loader_cfg=None,
            ),
        }
    )

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    collator="videochat3_sft_collator",
    num_workers=3,
    pack_extra_buffer_size=20,
)

optim_cfg = VisionAdamWConfig(
    vit_lr=vit_lr,
    projector_lr=lr,
    lr=lr,
    weight_decay=weight_decay,
    foreach=False,
)
lr_cfg = LRConfig(
    lr_type="cosine",
    warmup_ratio=warmup_ratio,
    lr_min=lr_min,
)
fsdp_cfg = FSDPConfig(
    sp_size=1,
    recompute_ratio=recompute_ratio,
    torch_compile=False,
)

trainer = TrainerConfig(
    load_from=model_path,
    resume_cfg=ResumeConfig(auto_resume=True),
    tokenizer_path=model_path,
    fsdp_cfg=fsdp_cfg,
    exp_tracker="wandb",
    wandb_config=WandbConfig(
        entity="LVSM-Experiment",
        project="videochat3",
        name=run_name,
        base_url="https://api.wandb.ai",
        run_id=os.getenv("WANDB_RUN_ID", run_name),
        group="videochat3-lact-stage3-ve",
        job_type="train",
        tags=[
            "videochat3-4b",
            "lact",
            "fast-weight",
            "fw-window-4",
            "vision-encoder-only",
            "stage3-full-local",
        ],
        resume="allow",
        mode=os.getenv("WANDB_MODE", "online"),
    ),
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(
        mode="chunk",
        chunk_size=1024,
        loss_reduction=loss_reduction,
    ),
    global_batch_size=global_batch_size,
    total_epoch=total_epoch,
    hf_interval=hf_interval,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=checkpoint_maxkeep,
    hf_max_keep=hf_max_keep,
    work_dir=work_dir,
)
