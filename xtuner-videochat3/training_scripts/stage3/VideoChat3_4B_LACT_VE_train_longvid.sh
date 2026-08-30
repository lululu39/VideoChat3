#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XTUNER_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd -- "${XTUNER_ROOT}/.." && pwd)"
DATASET_ROOT="${VIDEOCHAT_FLASH_LONGVID_ROOT:-/mnt/localssd/dataset/VideoChat3/VideoChat-Flash-Training-Data}"
MANIFEST="${DATASET_ROOT}/VideoChatFlash_LongVid_VideoChat3.json"

if [[ ! -f "${MANIFEST}" ]]; then
  "${PROJECT_ROOT}/.venv/bin/python" \
    "${PROJECT_ROOT}/scripts/prepare_videochat_flash_longvid.py" \
    --dataset-root "${DATASET_ROOT}"
fi

: "${WANDB_NAME:?Set WANDB_NAME after recording the numbered experiment in exp_results.md}"

export WANDB_RUN_ID="${WANDB_RUN_ID:-${WANDB_NAME}}"
export RDZV_ID="${RDZV_ID:-${WANDB_RUN_ID}}"
export LOG_DIR="${LOG_DIR:-${XTUNER_ROOT}/work_dir/stage3/${WANDB_NAME}/torchrun_logs}"
export VIDEOCHAT3_STAGE3_METADATA_PATH="${MANIFEST}"
export VIDEOCHAT3_STAGE3_DATASET_TAG="${VIDEOCHAT3_STAGE3_DATASET_TAG:-videochat-flash-longvid}"
export VIDEOCHAT3_STAGE3_CACHE_DIR="${VIDEOCHAT3_STAGE3_CACHE_DIR:-dataset_cache/cache_videochat3_4B_lact_videochat_flash_longvid}"
export VIDEOCHAT3_TRAINING_TAG="${VIDEOCHAT3_TRAINING_TAG:-img1fps-f64-512-fw4}"
export BUILD_STAGE3_FULL_LOCAL_ANNOTATIONS=0

exec bash "${SCRIPT_DIR}/VideoChat3_4B_LACT_VE_train_stage3.sh"
