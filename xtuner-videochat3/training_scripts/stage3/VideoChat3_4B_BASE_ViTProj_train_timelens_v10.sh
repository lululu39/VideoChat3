#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_NAME="vc3-4b-base-r4mean-vitproj-timelens-rand12624-8xh100-gb16-video2fps-f448-s8k-lr2e5-v10"
export WANDB_RUN_ID="${WANDB_NAME}"
export VIDEOCHAT3_MODEL_VARIANT="base-vit-projector"
export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_FACTOR=4
export VIDEOCHAT3_MODEL_PATH="/mnt/localssd/VideoChat3/VideoChat3-4B"
export VIDEOCHAT3_TIMELENS_MANIFEST="/mnt/localssd/dataset/VideoChat3/TimeLens-100K/TimeLens100K_Visual_Random12624_VideoChat3.json"
export VIDEOCHAT3_STAGE3_DATASET_TAG="timelens-100k-visual-random12624"
export VIDEOCHAT3_STAGE3_CACHE_DIR="dataset_cache/cache_videochat3_4B_base_r4mean_timelens_random12624_v10"
export VIDEOCHAT3_TRAINING_TAG="base-r4mean-vit-projector-v10"
export VIDEOCHAT3_VIT_LR=2e-5
export VIDEOCHAT3_LR_MIN=1e-6
unset VIDEOCHAT3_TRAIN_LACT_ONLY
unset VIDEOCHAT3_FREEZE_LACT_MEMORY_GATE
unset VIDEOCHAT3_LACT_LR
unset VIDEOCHAT3_LACT_GATE_LR
unset VIDEOCHAT3_LR_MIN_RATIO

exec bash "${SCRIPT_DIR}/VideoChat3_4B_LACT_VE_train_timelens.sh"
