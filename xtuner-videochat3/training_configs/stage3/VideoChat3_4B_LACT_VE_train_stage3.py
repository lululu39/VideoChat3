import json
import os
from pathlib import Path

from xtuner.v1.config import FSDPConfig, LRConfig, VisionAdamWConfig
from xtuner.v1.datasets import VideoChat3TokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import VideoChat3Dense4BConfig, VideoChat3LACTDense4BConfig
from xtuner.v1.model.compose.videochat3 import (
    VideoChat3LACTVisionConfig,
    VideoChat3VisionConfig,
)
from xtuner.v1.train import ResumeConfig, TrainerConfig, WandbConfig


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes"}


run_name = os.getenv(
    "WANDB_NAME",
    "vc3-4b-lact-fw4-ve-s3-full89k-8xh100-gb16-f3600-s8k-vitlr2p5e6-ns5r1-stgr1-v1",
)
model_path = Path(
    os.getenv(
        "VIDEOCHAT3_MODEL_PATH",
        "/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init",
    )
)
metadata_path = Path(
    os.getenv(
        "VIDEOCHAT3_STAGE3_METADATA_PATH",
        "/mnt/localssd/dataset/VideoChat3/VideoChat3-Stage3-Training-Data/"
        "VideoChat3_Stage3_Training_Data_full_local.json",
    )
)
dataset_tag = os.getenv("VIDEOCHAT3_STAGE3_DATASET_TAG", "stage3-full-local")
training_tag = os.getenv("VIDEOCHAT3_TRAINING_TAG")
work_dir = Path("work_dir/stage3") / run_name
cache_dir = Path(
    os.getenv(
        "VIDEOCHAT3_STAGE3_CACHE_DIR",
        "dataset_cache/cache_videochat3_4B_lact_stage3",
    )
)
macro_temporal_compression_factor = int(
    os.getenv("VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_FACTOR", "1")
)
macro_temporal_compression_mode = os.getenv(
    "VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_MODE",
    "auto",
)
model_variant = os.getenv("VIDEOCHAT3_MODEL_VARIANT", "lact")
train_projector = env_bool("VIDEOCHAT3_TRAIN_PROJECTOR")

if model_variant == "lact":
    model_cfg = VideoChat3LACTDense4BConfig(
        train_lact_only=env_bool("VIDEOCHAT3_TRAIN_LACT_ONLY"),
        freeze_lact_memory_gate=env_bool("VIDEOCHAT3_FREEZE_LACT_MEMORY_GATE"),
        freeze_projector=not train_projector,
        vision_config=VideoChat3LACTVisionConfig(
            attn_impl="flash_attention_2",
            memory_type=os.getenv("VIDEOCHAT3_MEMORY_TYPE", "swiglu"),
            fw_num_heads=int(os.getenv("VIDEOCHAT3_FW_NUM_HEADS", "1")),
            inner_optim=os.getenv("VIDEOCHAT3_INNER_OPTIM", "muon"),
            fw_order=os.getenv("VIDEOCHAT3_FW_ORDER", "serial"),
            lact_3d_rope=env_bool("VIDEOCHAT3_LACT_3D_ROPE"),
            lact_gate=os.getenv("VIDEOCHAT3_LACT_GATE", "linear"),
            lact_gate_init=float(os.getenv("VIDEOCHAT3_LACT_GATE_INIT", "0")),
            clip_ns_grad_ratio=False,
            recompute_ns5_backward=True,
            clip_state_grad_ratio=env_bool(
                "VIDEOCHAT3_CLIP_STATE_GRAD_RATIO",
                True,
            ),
            fw_update_layer_group_size=int(
                os.getenv("VIDEOCHAT3_FW_UPDATE_LAYER_GROUP_SIZE", "1")
            ),
            macro_temporal_compression_factor=macro_temporal_compression_factor,
            macro_temporal_compression_mode=macro_temporal_compression_mode,
        ),
    )
elif model_variant == "base-vit-projector":
    model_cfg = VideoChat3Dense4BConfig(
        freeze_vision=False,
        freeze_projector=False,
        freeze_language=True,
        vision_config=VideoChat3VisionConfig(
            attn_impl="flash_attention_2",
            macro_temporal_compression_factor=macro_temporal_compression_factor,
            macro_temporal_compression_mode=macro_temporal_compression_mode,
        ),
    )
else:
    raise ValueError(f"Unsupported VIDEOCHAT3_MODEL_VARIANT={model_variant!r}")

sample_max_length = int(os.getenv("VIDEOCHAT3_SAMPLE_MAX_LENGTH", "8192"))
pack_max_length = int(os.getenv("VIDEOCHAT3_PACK_MAX_LENGTH", "8192"))
frame_max_pixels = 224 * 224
video_max_total_pixels = int(
    os.getenv("VIDEOCHAT3_VIDEO_MAX_TOTAL_PIXELS", 128 * frame_max_pixels)
)
video_max_frames = 3600
reference_global_batch_size = 128
reference_vit_lr = 8e-5 / 4
global_batch_size = 16
total_epoch = 1
hf_interval = 5000
hf_max_keep = 1
checkpoint_interval = 100
checkpoint_maxkeep = 1

vit_lr = float(
    os.getenv(
        "VIDEOCHAT3_VIT_LR",
        reference_vit_lr * global_batch_size / reference_global_batch_size,
    )
)
lact_lr = os.getenv("VIDEOCHAT3_LACT_LR")
lact_lr = float(lact_lr) if lact_lr is not None else None
lact_gate_lr = os.getenv("VIDEOCHAT3_LACT_GATE_LR")
lact_gate_lr = float(lact_gate_lr) if lact_gate_lr is not None else None
lr = vit_lr
weight_decay = 0.0
warmup_ratio = 0.03
lr_min = float(
    os.getenv(
        "VIDEOCHAT3_LR_MIN",
        1e-6 * global_batch_size / reference_global_batch_size,
    )
)
lr_min_ratio = os.getenv("VIDEOCHAT3_LR_MIN_RATIO")
lr_min_ratio = float(lr_min_ratio) if lr_min_ratio is not None else None
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
                video_read_type=data.get("video_read_type"),
                video_frame_multiple=data.get("video_frame_multiple", 1),
                macro_temporal_compression_factor=macro_temporal_compression_factor,
                macro_temporal_compression_mode=macro_temporal_compression_mode,
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
    lact_lr=lact_lr,
    lact_gate_lr=lact_gate_lr,
    projector_lr=lr,
    lr=lr,
    weight_decay=weight_decay,
    foreach=False,
)
lr_cfg = LRConfig(
    lr_type=os.getenv("VIDEOCHAT3_LR_TYPE", "cosine"),
    warmup_ratio=warmup_ratio,
    lr_min=lr_min,
    lr_min_ratio=lr_min_ratio,
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
        group=(
            "videochat3-lact-stage3-ve"
            if model_variant == "lact"
            else "videochat3-base-stage3-vit-projector"
        ),
        job_type="train",
        tags=(
            (
                [
                    "videochat3-4b",
                    "lact",
                    "fast-weight",
                    "fw-window-4",
                    f"memory-{model_cfg.vision_config.memory_type}",
                    f"inner-{model_cfg.vision_config.inner_optim}",
                    (
                        "ns5-exact-backward"
                        if model_cfg.vision_config.inner_optim == "muon"
                        else "raw-delta-update"
                    ),
                    (
                        "ns5-backward-recompute"
                        if model_cfg.vision_config.inner_optim == "muon"
                        else "no-ns5"
                    ),
                    (
                        "state-ratio-clip-rho1"
                        if model_cfg.vision_config.clip_state_grad_ratio
                        else "state-ratio-clip-off"
                    ),
                    f"fw-layer-batch-{model_cfg.vision_config.fw_update_layer_group_size}",
                    f"macro-{model_cfg.vision_config.macro_temporal_compression_mode}",
                    (
                        "fw-projector"
                        if train_projector
                        else "vision-encoder-only"
                    ),
                    dataset_tag,
                ]
                if model_variant == "lact"
                else [
                    "videochat3-4b",
                    "base",
                    f"macro-{model_cfg.vision_config.macro_temporal_compression_mode}",
                    f"macro-r{macro_temporal_compression_factor}",
                    "vit-projector",
                    dataset_tag,
                ]
            )
            + ([training_tag] if training_tag else [])
        ),
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
