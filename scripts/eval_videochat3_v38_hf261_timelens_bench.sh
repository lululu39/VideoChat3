#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TIMELENS_BENCH_ROOT=/mnt/localssd/dataset/VideoChat3/TimeLens-Bench
export LMUData=/mnt/localssd/dataset/VLMEvalKit/LMUData
export HF_HOME=/mnt/localssd/dataset/VLMEvalKit/hf_home
export PYTHONPATH="${PROJECT_ROOT}/vlmevalkit-videochat3${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
OUTPUT_ROOT=/mnt/localssd/VideoChat3/eval/videochat3-v38-hf261-timelens-bench
cd "${PROJECT_ROOT}"
# Validate every annotation video before starting native evaluation.
"${PROJECT_ROOT}/.venv/bin/python" -c 'from pathlib import Path; from scripts.prepare_timelens_bench import validate; print(validate(Path("/mnt/localssd/dataset/VideoChat3/TimeLens-Bench")))'
install -d "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}/vlmevalkit-videochat3"
# Do not use the historical exclusive watchdog: other GPU jobs are out of scope.
exec "${PROJECT_ROOT}/.venv/bin/torchrun" --nproc-per-node=4 --master-port=41038 \
  run.py --config configs/videochat3_v38_hf261_timelens_bench.json \
  --work-dir "${OUTPUT_ROOT}" --reuse
