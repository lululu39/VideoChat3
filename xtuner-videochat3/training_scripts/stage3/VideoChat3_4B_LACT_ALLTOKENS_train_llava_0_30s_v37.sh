#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_NAME=vc3-lact-l16-delta-3drope-parallel-alltokens-vitfwproj-llava0to30-qa495013-4xh100-gb16-f64-s8k-lr2e5-v37
export WANDB_RUN_ID="${WANDB_NAME}"
export WANDB_MODE=online
export VIDEOCHAT3_MODEL_VARIANT=lact
export VIDEOCHAT3_MODEL_PATH=/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init
export VIDEOCHAT3_LLAVA_OUTPUT=all
export VIDEOCHAT3_LLAVA_ROOT=/mnt/localssd/dataset/VideoChat3/LLaVA-Video-178K
export VIDEOCHAT3_LLAVA_MANIFEST="${VIDEOCHAT3_LLAVA_ROOT}/LLaVA_Video_0_30s_train_VideoChat3.json"
export VIDEOCHAT3_STAGE3_CACHE_DIR="${VIDEOCHAT3_LLAVA_ROOT}/token_cache/lact-all-s8192-smoke0"
export VIDEOCHAT3_LLAVA_WORK_DIR="/mnt/localssd/VideoChat3/training/${WANDB_NAME}"
export VIDEOCHAT3_MAX_STEPS=800
# Stop at 800 while retaining v35's full-epoch warmup/cosine trajectory.
export VIDEOCHAT3_LR_SCHEDULE_STEPS=5158
export VIDEOCHAT3_TOTAL_EPOCH=1
export VIDEOCHAT3_GLOBAL_BATCH_SIZE=16
export VIDEOCHAT3_SAMPLE_MAX_LENGTH=8192
export VIDEOCHAT3_PACK_MAX_LENGTH=8192
export VIDEOCHAT3_VIT_LR=2e-5
export VIDEOCHAT3_LACT_LR=2e-5
export VIDEOCHAT3_LR_MIN=1e-6
export VIDEOCHAT3_HF_INTERVAL=200
export VIDEOCHAT3_CHECKPOINT_INTERVAL=200
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NNODES=1
export NPROC_PER_NODE=4
export MASTER_PORT=40247
export PYTHONDONTWRITEBYTECODE=1
unset VIDEOCHAT3_LLAVA_SMOKE

exec bash "${SCRIPT_DIR}/VideoChat3_4B_LACT_train_llava_0_30s.sh"
