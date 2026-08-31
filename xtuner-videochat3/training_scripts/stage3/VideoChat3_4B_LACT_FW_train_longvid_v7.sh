#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_NAME="vc3-4b-lact-fw4-fwonly-longvid5870-8xh100-gb16-img1fps-f512-s8k-fwlr2e5-randgate-frozen-seed42-ns5r1-stgr1-v7"
export WANDB_RUN_ID="${WANDB_NAME}"
export VIDEOCHAT3_MODEL_PATH="/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-random-gate-init"
export VIDEOCHAT3_TRAIN_LACT_ONLY=1
export VIDEOCHAT3_FREEZE_LACT_MEMORY_GATE=1
export VIDEOCHAT3_VIT_LR=2e-5
export VIDEOCHAT3_LR_MIN=1e-6
export VIDEOCHAT3_TRAINING_TAG=fw-only-random-gate-frozen-v7
unset VIDEOCHAT3_LACT_LR
unset VIDEOCHAT3_LACT_GATE_LR
unset VIDEOCHAT3_LR_MIN_RATIO

exec bash "${SCRIPT_DIR}/VideoChat3_4B_LACT_VE_train_longvid.sh"
