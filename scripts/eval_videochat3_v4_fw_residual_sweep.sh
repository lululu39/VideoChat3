#!/usr/bin/env bash

set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${PROJECT_ROOT}/xtuner-videochat3/work_dir/stage3/vc3-4b-lact-fw4-fwonly-longvid5870-8xh100-gb16-img1fps-f512-s8k-fwlr2e5-ns5r1-stgr1-v4/20260830064443/hf-134"
ANNOTATION_XLSX="/mnt/localssd/VideoChat3/eval/videochat3-lact-v6-core/VideoChat3-4B-LACT-v6/T20260831_G9a1c4f2d/VideoChat3-4B-LACT-v6_Video-MME_long_2fps_limit_1024_448px_80kctx.xlsx"
FRAME_ROOT="${LMU_DATA_ROOT:-/home/sigma/LMUData}/images/Video-MME"
VIDEO_ROOT="${VIDEOMME_ROOT:-/mnt/localssd/dataset/VLMEvalKit/Video-MME}/video"
OUTPUT_DIR="/mnt/localssd/VideoChat3/eval/v4-fw-residual-alpha-sweep"
WATCHDOG_LOG="/mnt/localssd/VideoChat3/gpu_exclusive_watchdog/eval_v4-fw-residual-alpha-sweep.jsonl"

export HF_HOME="${HF_HOME:-/mnt/localssd/dataset/VLMEvalKit/hf_home}"

install -d "${OUTPUT_DIR}" "$(dirname -- "${WATCHDOG_LOG}")"

setsid "${PROJECT_ROOT}/.venv/bin/torchrun" \
  --nproc-per-node=8 \
  --master-port=42000 \
  "${PROJECT_ROOT}/scripts/eval_videochat3_v4_fw_residual_sweep.py" \
  --model-path "${MODEL_PATH}" \
  --annotation-xlsx "${ANNOTATION_XLSX}" \
  --frame-root "${FRAME_ROOT}" \
  --video-root "${VIDEO_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-videos 96 \
  --max-questions-per-video 3 \
  --seed 42 &
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
