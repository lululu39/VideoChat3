#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${WANDB_NAME:?Set a numbered TimeLens experiment name after recording it in exp_results.md}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${WANDB_NAME}}"
export VIDEOCHAT3_TRAIN_LACT_ONLY=1
export VIDEOCHAT3_VIT_LR="${VIDEOCHAT3_VIT_LR:-2e-5}"
export VIDEOCHAT3_LR_MIN="${VIDEOCHAT3_LR_MIN:-1e-6}"
export VIDEOCHAT3_TRAINING_TAG="${VIDEOCHAT3_TRAINING_TAG:-fw-only-timelens}"
unset VIDEOCHAT3_LACT_LR
unset VIDEOCHAT3_LACT_GATE_LR
unset VIDEOCHAT3_LR_MIN_RATIO

exec bash "${SCRIPT_DIR}/VideoChat3_4B_LACT_VE_train_timelens.sh"
