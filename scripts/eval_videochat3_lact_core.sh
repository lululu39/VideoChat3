#!/usr/bin/env bash

set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_ROOT="${PROJECT_ROOT}/vlmevalkit-videochat3"
EVAL_ID="videochat3-lact-lite31k-core"
WANDB_EVAL_NAME="vc3-lact-lite31k-core-eval-v1"
CONFIG_PATH="${EVAL_ROOT}/configs/videochat3_lact_core_eval.json"
OUTPUT_ROOT="/mnt/localssd/VideoChat3/eval/${EVAL_ID}"
WATCHDOG_LOG="/mnt/localssd/VideoChat3/gpu_exclusive_watchdog/eval_${EVAL_ID}.jsonl"

export LMUData="${LMUData:-/mnt/localssd/dataset/VLMEvalKit/LMUData}"
export VIDEOMME_ROOT="${VIDEOMME_ROOT:-/mnt/localssd/dataset/VLMEvalKit/Video-MME}"
export MVBENCH_ROOT="${MVBENCH_ROOT:-/mnt/localssd/dataset/VLMEvalKit/MVBench-MP4}"
export HF_HOME="${HF_HOME:-/mnt/localssd/dataset/VLMEvalKit/hf_home}"
export PYTHONPATH="${EVAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

install -d "${OUTPUT_ROOT}" "$(dirname -- "${WATCHDOG_LOG}")"
cd "${EVAL_ROOT}"

setsid "${PROJECT_ROOT}/.venv/bin/torchrun" \
  --nproc-per-node=8 \
  --master-port=41000 \
  run.py \
  --config "${CONFIG_PATH}" \
  --work-dir "${OUTPUT_ROOT}" \
  --reuse &
eval_pid=$!

"${PROJECT_ROOT}/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/gpu_exclusive_watchdog.py" \
  --allow-root-pid "${eval_pid}" \
  --log "${WATCHDOG_LOG}" \
  --interval 0.25 \
  --grace 2.0 &
watchdog_pid=$!

cleanup() {
  kill -TERM -- "-${eval_pid}" 2>/dev/null || true
  kill -TERM "${watchdog_pid}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

set +e
wait "${eval_pid}"
eval_status=$?
wait "${watchdog_pid}"
watchdog_status=$?
set -e

trap - INT TERM EXIT
if (( watchdog_status != 0 )); then
  echo "GPU watchdog exited with status ${watchdog_status}" >&2
fi

if (( eval_status == 0 )); then
  export WANDB_BASE_URL="https://api.wandb.ai"
  if [[ -n "${WANDB_PUBLIC_API_KEY:-}" ]]; then
    export WANDB_API_KEY="${WANDB_PUBLIC_API_KEY}"
  else
    unset WANDB_API_KEY
  fi
  "${PROJECT_ROOT}/.venv/bin/python" \
    "${PROJECT_ROOT}/scripts/upload_videochat3_eval_wandb.py" \
    --output-root "${OUTPUT_ROOT}" \
    --eval-config "${CONFIG_PATH}" \
    --run-name "${WANDB_EVAL_NAME}"
fi
exit "${eval_status}"
