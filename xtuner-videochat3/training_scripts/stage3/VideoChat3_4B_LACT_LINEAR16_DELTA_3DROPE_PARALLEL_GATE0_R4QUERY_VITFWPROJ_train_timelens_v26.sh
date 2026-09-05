#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_NAME="vc3-lact-l16-delta-3drope-parallel-gate0-r4query-vitfwproj-timelens-r12624-8xh100-gb16-f448-s4k-lr2e5-v26"
export WANDB_RUN_ID="${WANDB_NAME}"
export VIDEOCHAT3_MODEL_VARIANT="lact"
export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_FACTOR=1
export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_MODE="auto"
export VIDEOCHAT3_LACT_CHUNK_QUERY=1
export VIDEOCHAT3_LACT_CHUNK_QUERY_MODE="spatial_quarter"
export VIDEOCHAT3_MODEL_PATH="/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init"
export VIDEOCHAT3_TIMELENS_MANIFEST="/mnt/localssd/dataset/VideoChat3/TimeLens-100K/TimeLens100K_Visual_Random12624_VideoChat3.json"
export VIDEOCHAT3_STAGE3_DATASET_TAG="timelens-100k-visual-random12624"
export VIDEOCHAT3_STAGE3_CACHE_DIR="dataset_cache/cache_videochat3_4B_lact_linear16_delta_3drope_parallel_gate0_r4query_timelens_random12624_s4k_v24"
export VIDEOCHAT3_TRAINING_TAG="l16-delta-3dr-par-g0-r4query-vitfwproj-s4k-v26"
export VIDEOCHAT3_TRAIN_LACT_ONLY=0
export VIDEOCHAT3_TRAIN_PROJECTOR=1
export VIDEOCHAT3_VIT_LR=2e-5
export VIDEOCHAT3_LACT_LR=2e-5
export VIDEOCHAT3_LR_TYPE=cosine
export VIDEOCHAT3_LR_MIN=1e-6
export VIDEOCHAT3_SAMPLE_MAX_LENGTH=4096
export VIDEOCHAT3_PACK_MAX_LENGTH=4096
export VIDEOCHAT3_FW_UPDATE_LAYER_GROUP_SIZE=1
export VIDEOCHAT3_MEMORY_TYPE=linear
export VIDEOCHAT3_FW_NUM_HEADS=16
export VIDEOCHAT3_INNER_OPTIM=delta
export VIDEOCHAT3_CLIP_STATE_GRAD_RATIO=0
export VIDEOCHAT3_LACT_3D_ROPE=1
export VIDEOCHAT3_FW_ORDER=parallel
export VIDEOCHAT3_LACT_GATE=linear
export VIDEOCHAT3_LACT_GATE_INIT=0
unset VIDEOCHAT3_FREEZE_LACT_MEMORY_GATE
unset VIDEOCHAT3_LACT_GATE_LR
unset VIDEOCHAT3_LR_MIN_RATIO
unset VIDEOCHAT3_PARALLEL_STRATEGY

exec bash "${SCRIPT_DIR}/VideoChat3_4B_LACT_VE_train_timelens.sh"
