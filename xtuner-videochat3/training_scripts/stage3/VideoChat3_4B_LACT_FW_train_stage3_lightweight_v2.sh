#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_NAME="${WANDB_NAME:-vc3-4b-lact-fw4-fwonly-s3-lite31k-8xh100-gb16-f3600-s8k-fwlr2e5-ns5r1-stgr1-v2}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${WANDB_NAME}}"
export VIDEOCHAT3_TRAIN_LACT_ONLY=1
export VIDEOCHAT3_VIT_LR=2e-5
export VIDEOCHAT3_LR_MIN=1e-6
export VIDEOCHAT3_TRAINING_TAG=fw-only-v2

exec bash "${SCRIPT_DIR}/VideoChat3_4B_LACT_VE_train_stage3_lightweight.sh"
