#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XTUNER_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd -- "${XTUNER_ROOT}/.." && pwd)"
DATASET_ROOT="${VIDEOCHAT3_TIMELENS_ROOT:-/mnt/localssd/dataset/VideoChat3/TimeLens-100K}"
MANIFEST="${VIDEOCHAT3_TIMELENS_MANIFEST:-${DATASET_ROOT}/TimeLens100K_Visual_SFT30K_VideoChat3.json}"

if [[ ! -f "${MANIFEST}" ]]; then
  "${PROJECT_ROOT}/.venv/bin/python" \
    "${PROJECT_ROOT}/scripts/prepare_timelens_100k.py" \
    --dataset-root "${DATASET_ROOT}" \
    --filter-mode visual \
    --target-size 30000
fi

: "${WANDB_NAME:?Set WANDB_NAME after recording the numbered experiment in exp_results.md}"

export WANDB_RUN_ID="${WANDB_RUN_ID:-${WANDB_NAME}}"
export RDZV_ID="${RDZV_ID:-${WANDB_RUN_ID}}"
export LOG_DIR="${LOG_DIR:-${XTUNER_ROOT}/work_dir/stage3/${WANDB_NAME}/torchrun_logs}"
export VIDEOCHAT3_STAGE3_METADATA_PATH="${MANIFEST}"
export VIDEOCHAT3_STAGE3_DATASET_TAG="${VIDEOCHAT3_STAGE3_DATASET_TAG:-timelens-100k-visual-sft30k}"
export VIDEOCHAT3_STAGE3_CACHE_DIR="${VIDEOCHAT3_STAGE3_CACHE_DIR:-dataset_cache/cache_videochat3_4B_lact_timelens_100k_visual_sft30k}"
export VIDEOCHAT3_TRAINING_TAG="${VIDEOCHAT3_TRAINING_TAG:-video2fps-f64-448-fw4}"
export VIDEOCHAT3_VIDEO_MAX_TOTAL_PIXELS="${VIDEOCHAT3_VIDEO_MAX_TOTAL_PIXELS:-14680064}"
export BUILD_STAGE3_FULL_LOCAL_ANNOTATIONS=0

exec bash "${SCRIPT_DIR}/VideoChat3_4B_LACT_VE_train_stage3.sh"
