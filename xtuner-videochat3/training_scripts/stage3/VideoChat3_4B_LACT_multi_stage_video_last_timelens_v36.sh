#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WANDB_NAME="vc3-lact-l16-delta-parallel-3drope-multi_stage_video_last-timelens-r12624-hf800-4xh100-gb16-s8k-v36"
export WANDB_RUN_ID="${WANDB_NAME}"
export WANDB_MODE=online
export WANDB_ENTITY=LVSM-Experiment
export WANDB_PROJECT=videochat3
export WANDB_BASE_URL=https://api.wandb.ai
if [[ -n "${WANDB_PUBLIC_API_KEY:-}" ]]; then
  export WANDB_API_KEY="${WANDB_PUBLIC_API_KEY}"
else
  unset WANDB_API_KEY
fi
export VIDEOCHAT3_MODEL_PATH="/mnt/localssd/VideoChat3/training/vc3-lact-l16-delta-3drope-parallel-alltokens-vitfwproj-llava0to30-qa495013-4xh100-gb16-f64-s8k-lr2e5-v35/20260909192723/hf-800"
export VIDEOCHAT3_STAGE3_METADATA_PATH="/mnt/localssd/dataset/VideoChat3/TimeLens-100K/TimeLens100K_Visual_Random12624_VideoChat3.json"
export VIDEOCHAT3_STAGE3_DATASET_TAG=timelens-100k-visual-random12624
export VIDEOCHAT3_STAGE3_CACHE_DIR="/mnt/localssd/dataset/VideoChat3/TimeLens-100K/token_cache/hf800-all-s8k-v36"
export VIDEOCHAT3_CURRICULUM_WORK_DIR="/mnt/localssd/VideoChat3/training/${WANDB_NAME}"
export VIDEOCHAT3_TRAINING_TAG=multi_stage_video_last
export VIDEOCHAT3_MODEL_VARIANT=lact
export VIDEOCHAT3_TRAIN_LACT_ONLY=0
export VIDEOCHAT3_TRAIN_PROJECTOR=1
export VIDEOCHAT3_CHUNK_QUERY=0
export VIDEOCHAT3_CHUNK_QUERY_MODE=single
export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_FACTOR=1
export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_MODE=select_last
export VIDEOCHAT3_VIT_LR=2e-5
export VIDEOCHAT3_LACT_LR=2e-5
export VIDEOCHAT3_LR_MIN=1e-6
export VIDEOCHAT3_LR_TYPE=cosine
export VIDEOCHAT3_SAMPLE_MAX_LENGTH=8192
export VIDEOCHAT3_PACK_MAX_LENGTH=8192
export VIDEOCHAT3_VIDEO_MAX_TOTAL_PIXELS=14680064
export VIDEOCHAT3_MEMORY_TYPE=linear
export VIDEOCHAT3_FW_NUM_HEADS=16
export VIDEOCHAT3_INNER_OPTIM=delta
export VIDEOCHAT3_FW_ORDER=parallel
export VIDEOCHAT3_FW_UPDATE_LAYER_GROUP_SIZE=1
export VIDEOCHAT3_CLIP_STATE_GRAD_RATIO=0
export VIDEOCHAT3_LACT_3D_ROPE=1
export VIDEOCHAT3_LACT_GATE=linear
export VIDEOCHAT3_LACT_GATE_INIT=0
export VIDEOCHAT3_VISION_ACTIVATION_OFFLOAD=1
unset VIDEOCHAT3_FREEZE_LACT_MEMORY_GATE VIDEOCHAT3_LACT_GATE_LR VIDEOCHAT3_LR_MIN_RATIO
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NNODES=1
export NPROC_PER_NODE=4
export MASTER_PORT=40246
export RDZV_ID="${WANDB_RUN_ID}"
export LOG_DIR="${VIDEOCHAT3_CURRICULUM_WORK_DIR}/torchrun_logs"
export PYTHONDONTWRITEBYTECODE=1
export XTUNER_SFT_ENTRYPOINT=xtuner/v1/train/cli/videochat3_curriculum.py
exec bash "${SCRIPT_DIR}/../run_sft.sh" "training_configs/stage3/VideoChat3_4B_LACT_multi_stage_video_last.py"
