#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_NAME="vc3-4b-lact-r4select-fwproj-timelens-rand12624-8xh100-gb16-video2fps-f448-s2k-lr2e5-v11"
export WANDB_RUN_ID="${WANDB_NAME}"
export VIDEOCHAT3_MODEL_VARIANT="lact"
export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_FACTOR=4
export VIDEOCHAT3_MODEL_PATH="/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init"
export VIDEOCHAT3_TIMELENS_MANIFEST="/mnt/localssd/dataset/VideoChat3/TimeLens-100K/TimeLens100K_Visual_Random12624_VideoChat3.json"
export VIDEOCHAT3_STAGE3_DATASET_TAG="timelens-100k-visual-random12624"
export VIDEOCHAT3_STAGE3_CACHE_DIR="dataset_cache/cache_videochat3_4B_lact_r4select_timelens_random12624_v11"
export VIDEOCHAT3_TRAINING_TAG="lact-r4select-fw-projector-v11"
export VIDEOCHAT3_TRAIN_LACT_ONLY=1
export VIDEOCHAT3_TRAIN_PROJECTOR=1
export VIDEOCHAT3_VIT_LR=2e-5
export VIDEOCHAT3_LACT_LR=2e-5
export VIDEOCHAT3_LR_MIN=1e-6
export VIDEOCHAT3_SAMPLE_MAX_LENGTH=2048
export VIDEOCHAT3_PACK_MAX_LENGTH=2048
export VIDEOCHAT3_FW_UPDATE_LAYER_GROUP_SIZE=1
export VIDEOCHAT3_INNER_OPTIM=muon
export VIDEOCHAT3_CLIP_STATE_GRAD_RATIO=1
unset VIDEOCHAT3_FREEZE_LACT_MEMORY_GATE
unset VIDEOCHAT3_LACT_GATE_LR
unset VIDEOCHAT3_LR_MIN_RATIO

exec bash "${SCRIPT_DIR}/VideoChat3_4B_LACT_VE_train_timelens.sh"
