#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_NAME="vc3-4b-lact-fw4-fwonly-longvid5870-8xh100-gb16-img1fps-f512-s8k-fwlr2e5-gatelr2e4-ns5r1-stgr1-v5"
export WANDB_RUN_ID="${WANDB_NAME}"
export VIDEOCHAT3_TRAIN_LACT_ONLY=1
export VIDEOCHAT3_VIT_LR=2e-5
export VIDEOCHAT3_LACT_LR=2e-5
export VIDEOCHAT3_LACT_GATE_LR=2e-4
export VIDEOCHAT3_LR_MIN_RATIO=0.05
export VIDEOCHAT3_TRAINING_TAG=fw-only-gate10x-v5

exec bash "${SCRIPT_DIR}/VideoChat3_4B_LACT_VE_train_longvid.sh"
