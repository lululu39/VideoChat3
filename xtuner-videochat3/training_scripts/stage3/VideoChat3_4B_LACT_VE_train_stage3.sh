#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XTUNER_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd -- "${XTUNER_ROOT}/.." && pwd)"
WATCHDOG_SCRIPT="${PROJECT_ROOT}/scripts/gpu_exclusive_watchdog.py"
WATCHDOG_STATE_DIR="/mnt/localssd/VideoChat3/gpu_exclusive_watchdog"

export WANDB_ENTITY="LVSM-Experiment"
export WANDB_PROJECT="videochat3"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_NAME="${WANDB_NAME:-vc3-4b-lact-fw4-ve-s3-lite-v1}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${WANDB_NAME}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export NNODES="${NNODES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

if [[ -n "${WANDB_PUBLIC_API_KEY:-}" ]]; then
  export WANDB_API_KEY="${WANDB_PUBLIC_API_KEY}"
else
  unset WANDB_API_KEY
fi

echo "W&B: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "Run: ${WANDB_NAME} (id=${WANDB_RUN_ID})"

if [[ "${GPU_EXCLUSIVE:-1}" != "1" ]]; then
  exec bash "${SCRIPT_DIR}/../run_sft.sh" \
    "training_configs/stage3/VideoChat3_4B_LACT_VE_train_stage3.py"
fi

if [[ -f "${WATCHDOG_STATE_DIR}/controller.pid" ]]; then
  idle_controller_pid="$(cat "${WATCHDOG_STATE_DIR}/controller.pid")"
  if [[ -r "/proc/${idle_controller_pid}/cmdline" ]] && \
      tr '\0' ' ' < "/proc/${idle_controller_pid}/cmdline" | \
        grep -qx 'sleep 86400 '; then
    kill -TERM "${idle_controller_pid}" 2>/dev/null || true
  fi
fi
if [[ -f "${WATCHDOG_STATE_DIR}/watchdog.pid" ]]; then
  idle_watchdog_pid="$(cat "${WATCHDOG_STATE_DIR}/watchdog.pid")"
  if [[ -r "/proc/${idle_watchdog_pid}/cmdline" ]] && \
      tr '\0' ' ' < "/proc/${idle_watchdog_pid}/cmdline" | \
        grep -q 'gpu_exclusive_watchdog.py'; then
    kill -TERM "${idle_watchdog_pid}" 2>/dev/null || true
  fi
fi

watchdog_log="${WATCHDOG_STATE_DIR}/training_${WANDB_RUN_ID}.jsonl"
bash "${SCRIPT_DIR}/../run_sft.sh" \
  "training_configs/stage3/VideoChat3_4B_LACT_VE_train_stage3.py" &
training_pid=$!
"${PROJECT_ROOT}/.venv/bin/python" "${WATCHDOG_SCRIPT}" \
  --allow-root-pid "${training_pid}" \
  --log "${watchdog_log}" \
  --interval 0.25 \
  --grace 2.0 &
watchdog_pid=$!

cleanup() {
  kill -TERM "${training_pid}" 2>/dev/null || true
  kill -TERM "${watchdog_pid}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

set +e
wait "${training_pid}"
training_status=$?
wait "${watchdog_pid}"
watchdog_status=$?
set -e

trap - INT TERM EXIT
if (( watchdog_status != 0 )); then
  echo "GPU watchdog exited with status ${watchdog_status}" >&2
fi
exit "${training_status}"
