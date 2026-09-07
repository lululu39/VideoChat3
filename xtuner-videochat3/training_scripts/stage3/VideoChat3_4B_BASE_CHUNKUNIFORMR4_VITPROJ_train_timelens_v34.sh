#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_NAME="vc3-base-chunkuniformr4-vitproj-timelens-r12624-8xh100-gb16-f448-s4k-lr2e5-v34"
export WANDB_RUN_ID="${WANDB_NAME}"
export GPU_EXCLUSIVE=0
export VIDEOCHAT3_MODEL_VARIANT="base-vit-projector"
export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_FACTOR=4
export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_MODE="chunk_select_uniform"
export VIDEOCHAT3_LACT_CHUNK_QUERY=0
export VIDEOCHAT3_LACT_CHUNK_QUERY_MODE="single"
unset VIDEOCHAT3_CHUNK_QUERY VIDEOCHAT3_CHUNK_QUERY_MODE
export VIDEOCHAT3_MODEL_PATH="/mnt/localssd/VideoChat3/VideoChat3-4B"
export VIDEOCHAT3_TIMELENS_MANIFEST="/mnt/localssd/dataset/VideoChat3/TimeLens-100K/TimeLens100K_Visual_Random12624_VideoChat3.json"
export VIDEOCHAT3_STAGE3_DATASET_TAG="timelens-100k-visual-random12624"
export VIDEOCHAT3_STAGE3_CACHE_DIR="dataset_cache/cache_videochat3_4B_base_chunkuniformr4_timelens_random12624_s4k_v34"
export VIDEOCHAT3_TRAINING_TAG="base-chunkuniformr4-vitproj-s4k-v34"
export VIDEOCHAT3_TRAIN_PROJECTOR=1
export VIDEOCHAT3_VIT_LR=2e-5
export VIDEOCHAT3_LR_TYPE=cosine
export VIDEOCHAT3_LR_MIN=1e-6
export VIDEOCHAT3_SAMPLE_MAX_LENGTH=4096
export VIDEOCHAT3_PACK_MAX_LENGTH=4096
export VIDEOCHAT3_VISION_ACTIVATION_OFFLOAD=0
unset VIDEOCHAT3_TRAIN_LACT_ONLY VIDEOCHAT3_FREEZE_LACT_MEMORY_GATE
unset VIDEOCHAT3_LACT_LR VIDEOCHAT3_LACT_GATE_LR VIDEOCHAT3_LR_MIN_RATIO
unset VIDEOCHAT3_PARALLEL_STRATEGY

exec bash "${SCRIPT_DIR}/VideoChat3_4B_LACT_VE_train_timelens.sh"
