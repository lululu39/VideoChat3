#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
DATASET_ROOT="${VIDEOCHAT3_LLAVA_ROOT:-/mnt/localssd/dataset/VideoChat3/LLaVA-Video-178K}"
MODE="${VIDEOCHAT3_LLAVA_OUTPUT:-all}"
SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
  shift
fi
if (( $# )); then
  echo "Usage: $0 [--smoke]" >&2
  exit 2
fi
if (( SMOKE )); then
  export WANDB_NAME="${WANDB_NAME:-llava-0-30s-${MODE}-smoke-$(date -u +%Y%m%d%H%M%S)}"
  export VIDEOCHAT3_LLAVA_SMOKE=1
  export VIDEOCHAT3_LLAVA_MANIFEST="${DATASET_ROOT}/LLaVA_Video_0_30s_smoke_VideoChat3.json"
else
  : "${WANDB_NAME:?Set WANDB_NAME after recording the numbered experiment in exp_results.md}"
fi
export VIDEOCHAT3_STAGE3_METADATA_PATH="${VIDEOCHAT3_LLAVA_MANIFEST:-${DATASET_ROOT}/LLaVA_Video_0_30s_train_VideoChat3.json}"
if [[ ! -f "${VIDEOCHAT3_STAGE3_METADATA_PATH}" ]]; then
  echo "Missing ${VIDEOCHAT3_STAGE3_METADATA_PATH}; run prepare_llava_video_0_30s.py and check_llava_video_0_30s.py first." >&2
  exit 1
fi
export VIDEOCHAT3_MODEL_VARIANT="${VIDEOCHAT3_MODEL_VARIANT:-lact}"
if [[ "${VIDEOCHAT3_MODEL_VARIANT}" == "base-vit-projector" ]]; then
  export VIDEOCHAT3_MODEL_PATH="${VIDEOCHAT3_MODEL_PATH:-/mnt/localssd/VideoChat3/VideoChat3-4B}"
else
  export VIDEOCHAT3_MODEL_PATH="${VIDEOCHAT3_MODEL_PATH:-/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init}"
fi
export VIDEOCHAT3_CHUNK_QUERY=0
export VIDEOCHAT3_CHUNK_QUERY_MODE=single
export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_FACTOR=1
export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_MODE=auto
case "${MODE}" in
  r4query) export VIDEOCHAT3_CHUNK_QUERY=1
           export VIDEOCHAT3_CHUNK_QUERY_MODE=spatial_quarter ;;
  uniform) export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_FACTOR=4
           export VIDEOCHAT3_MACRO_TEMPORAL_COMPRESSION_MODE=chunk_select_uniform ;;
  all) ;;
  *) echo "VIDEOCHAT3_LLAVA_OUTPUT must be r4query, uniform, or all" >&2; exit 2 ;;
esac
export VIDEOCHAT3_TRAIN_LACT_ONLY=0
export VIDEOCHAT3_TRAIN_PROJECTOR=1
export VIDEOCHAT3_VIT_LR="${VIDEOCHAT3_VIT_LR:-2e-5}"
export VIDEOCHAT3_LACT_LR="${VIDEOCHAT3_LACT_LR:-2e-5}"
export VIDEOCHAT3_LR_TYPE=cosine
export VIDEOCHAT3_LR_MIN="${VIDEOCHAT3_LR_MIN:-1e-6}"
export VIDEOCHAT3_SAMPLE_MAX_LENGTH="${VIDEOCHAT3_SAMPLE_MAX_LENGTH:-8192}"
export VIDEOCHAT3_PACK_MAX_LENGTH="${VIDEOCHAT3_PACK_MAX_LENGTH:-8192}"
export VIDEOCHAT3_VIDEO_MAX_TOTAL_PIXELS=3211264
export VIDEOCHAT3_FW_UPDATE_LAYER_GROUP_SIZE=1
export VIDEOCHAT3_MEMORY_TYPE=linear
export VIDEOCHAT3_FW_NUM_HEADS=16
export VIDEOCHAT3_INNER_OPTIM=delta
export VIDEOCHAT3_CLIP_STATE_GRAD_RATIO=0
export VIDEOCHAT3_LACT_3D_ROPE=1
export VIDEOCHAT3_FW_ORDER=parallel
export VIDEOCHAT3_LACT_GATE=linear
export VIDEOCHAT3_LACT_GATE_INIT=0
export VIDEOCHAT3_VISION_ACTIVATION_OFFLOAD=0
unset VIDEOCHAT3_FREEZE_LACT_MEMORY_GATE VIDEOCHAT3_LACT_GATE_LR VIDEOCHAT3_LR_MIN_RATIO
export VIDEOCHAT3_STAGE3_DATASET_TAG=llava-video-0-30s-qa
export VIDEOCHAT3_TRAINING_TAG="l16-delta-parallel-3drope-${MODE}-vitfwproj-f64"
export VIDEOCHAT3_STAGE3_CACHE_DIR="${VIDEOCHAT3_STAGE3_CACHE_DIR:-${DATASET_ROOT}/token_cache/${VIDEOCHAT3_MODEL_VARIANT}-${MODE}-s${VIDEOCHAT3_SAMPLE_MAX_LENGTH}-smoke${SMOKE}}"
export VIDEOCHAT3_LLAVA_WORK_DIR="${VIDEOCHAT3_LLAVA_WORK_DIR:-/mnt/localssd/VideoChat3/training/${WANDB_NAME}}"
export WANDB_ENTITY=LVSM-Experiment
export WANDB_PROJECT=videochat3
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_RUN_ID="${WANDB_RUN_ID:-${WANDB_NAME}}"
export WANDB_MODE="${WANDB_MODE:-online}"
if [[ -n "${WANDB_PUBLIC_API_KEY:-}" ]]; then
  export WANDB_API_KEY="${WANDB_PUBLIC_API_KEY}"
else
  unset WANDB_API_KEY
fi
export NNODES="${NNODES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export RDZV_ID="${RDZV_ID:-${WANDB_RUN_ID}}"
export LOG_DIR="${LOG_DIR:-${VIDEOCHAT3_LLAVA_WORK_DIR}/torchrun_logs}"
exec bash "${SCRIPT_DIR}/../run_sft.sh" "training_configs/stage3/VideoChat3_4B_LACT_train_llava_0_30s.py"
