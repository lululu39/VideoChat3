#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XTUNER_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export WANDB_NAME="${WANDB_NAME:-vc3-4b-lact-fw4-ve-s3-lite31k-8xh100-gb16-f3600-s8k-vitlr2p5e6-ns5r1-stgr1-v1}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${WANDB_NAME}}"
export RDZV_ID="${RDZV_ID:-${WANDB_RUN_ID}}"
export LOG_DIR="${LOG_DIR:-${XTUNER_ROOT}/work_dir/stage3/${WANDB_NAME}/torchrun_logs}"
export VIDEOCHAT3_STAGE3_METADATA_PATH="${VIDEOCHAT3_STAGE3_METADATA_PATH:-/mnt/localssd/dataset/VideoChat3/VideoChat3-Stage3-Training-Data/VideoChat3_Stage3_Training_Data_local.json}"
export VIDEOCHAT3_STAGE3_DATASET_TAG="${VIDEOCHAT3_STAGE3_DATASET_TAG:-stage3-lightweight-local}"
export VIDEOCHAT3_STAGE3_CACHE_DIR="${VIDEOCHAT3_STAGE3_CACHE_DIR:-dataset_cache/cache_videochat3_4B_lact_stage3_lightweight}"
export BUILD_STAGE3_FULL_LOCAL_ANNOTATIONS=0

exec bash "${SCRIPT_DIR}/VideoChat3_4B_LACT_VE_train_stage3.sh"
