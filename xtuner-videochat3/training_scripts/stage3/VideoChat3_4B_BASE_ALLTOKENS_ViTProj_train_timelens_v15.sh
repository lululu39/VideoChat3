#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_NAME="vc3-4b-base-r1all-vitproj-timelens-rand12624-8xh100-gb16-video2fps-f448-s8k-lr2e5-v15"
export WANDB_RUN_ID="${WANDB_NAME}"
export VIDEOCHAT3_MODEL_VARIANT="base-vit-projector"
export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_FACTOR=1
export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_MODE="mean"
export VIDEOCHAT3_MODEL_PATH="/mnt/localssd/VideoChat3/VideoChat3-4B"
export VIDEOCHAT3_TIMELENS_MANIFEST="/mnt/localssd/dataset/VideoChat3/TimeLens-100K/TimeLens100K_Visual_Random12624_VideoChat3.json"
export VIDEOCHAT3_STAGE3_DATASET_TAG="timelens-100k-visual-random12624"
export VIDEOCHAT3_STAGE3_CACHE_DIR="dataset_cache/cache_videochat3_4B_base_r1all_timelens_random12624_s8k_v15"
export VIDEOCHAT3_TRAINING_TAG="base-r1all-vit-projector-s8k-fsdp-v15"
export VIDEOCHAT3_TRAIN_PROJECTOR=1
export VIDEOCHAT3_VIT_LR=2e-5
export VIDEOCHAT3_LR_MIN=1e-6
export VIDEOCHAT3_SAMPLE_MAX_LENGTH=8192
export VIDEOCHAT3_PACK_MAX_LENGTH=8192
unset VIDEOCHAT3_TRAIN_LACT_ONLY VIDEOCHAT3_FREEZE_LACT_MEMORY_GATE
unset VIDEOCHAT3_LACT_LR VIDEOCHAT3_LACT_GATE_LR VIDEOCHAT3_LR_MIN_RATIO
unset VIDEOCHAT3_PARALLEL_STRATEGY

exec bash "${SCRIPT_DIR}/VideoChat3_4B_LACT_VE_train_timelens.sh"
