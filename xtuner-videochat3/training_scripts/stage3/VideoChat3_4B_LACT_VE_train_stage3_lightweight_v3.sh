#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_NAME="${WANDB_NAME:-vc3-4b-lact-fw4-ve-s3-lite31k-8xh100-gb16-f3600-s8k-vitlr2p5e6-fwlr2e5-ns5r1-stgr1-v3}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${WANDB_NAME}}"
export VIDEOCHAT3_TRAIN_LACT_ONLY=0
export VIDEOCHAT3_VIT_LR=2.5e-6
export VIDEOCHAT3_LACT_LR=2e-5
export VIDEOCHAT3_LR_MIN=1.25e-7
export VIDEOCHAT3_LR_MIN_RATIO=0.05
export VIDEOCHAT3_TRAINING_TAG=split-vit-fw-lr-v3

exec bash "${SCRIPT_DIR}/VideoChat3_4B_LACT_VE_train_stage3_lightweight.sh"
