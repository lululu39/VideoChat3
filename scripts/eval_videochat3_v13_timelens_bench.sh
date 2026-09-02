#!/usr/bin/env bash

set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_ROOT="${PROJECT_ROOT}/vlmevalkit-videochat3"
EVAL_ID="videochat3-v13-timelens-bench"
CONFIG_PATH="${EVAL_ROOT}/configs/videochat3_v13_timelens_bench.json"
OUTPUT_ROOT="/mnt/localssd/VideoChat3/eval/${EVAL_ID}"
WATCHDOG_LOG="/mnt/localssd/VideoChat3/gpu_exclusive_watchdog/eval_${EVAL_ID}.jsonl"

export TIMELENS_BENCH_ROOT="${TIMELENS_BENCH_ROOT:-/mnt/localssd/dataset/VideoChat3/TimeLens-Bench}"
export LMUData="${LMUData:-/mnt/localssd/dataset/VLMEvalKit/LMUData}"
export HF_HOME="${HF_HOME:-/mnt/localssd/dataset/VLMEvalKit/hf_home}"
export PYTHONPATH="${EVAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for annotation in activitynet-timelens.json charades-timelens.json qvhighlights-timelens.json; do
  [[ -f "${TIMELENS_BENCH_ROOT}/${annotation}" ]] || {
    echo "Missing ${TIMELENS_BENCH_ROOT}/${annotation}; run scripts/prepare_timelens_bench.py first." >&2
    exit 2
  }
done

install -d "${OUTPUT_ROOT}" "$(dirname -- "${WATCHDOG_LOG}")"
cd "${EVAL_ROOT}"

setsid "${PROJECT_ROOT}/.venv/bin/torchrun" \
  --nproc-per-node=8 \
  --master-port=41013 \
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
exit "${eval_status}"
