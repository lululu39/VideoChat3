#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_NAME="vc3-4b-lact-fw4-fwonly-timelens-vis25247-8xh100-gb16-video2fps-f448-s8k-fwlr2e5-ns5r1-stgr1-v9"
export WANDB_RUN_ID="${WANDB_NAME}"
export VIDEOCHAT3_MODEL_PATH="/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init"
export VIDEOCHAT3_TIMELENS_MANIFEST="/mnt/localssd/dataset/VideoChat3/TimeLens-100K/TimeLens100K_Visual_SFT30K_VideoChat3.json"
export VIDEOCHAT3_STAGE3_DATASET_TAG="timelens-100k-visual-sft25247"
export VIDEOCHAT3_STAGE3_CACHE_DIR="dataset_cache/cache_videochat3_4B_lact_timelens_100k_visual_sft25247_v9"
export VIDEOCHAT3_TRAINING_TAG="fw-only-v9"
export VIDEOCHAT3_VIT_LR=2e-5
export VIDEOCHAT3_LR_MIN=1e-6

exec bash "${SCRIPT_DIR}/VideoChat3_4B_LACT_FW_train_timelens.sh"
