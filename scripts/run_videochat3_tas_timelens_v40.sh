#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_BASE_URL=https://api.wandb.ai
unset _WANDB_SERVICE WANDB_SERVICE
export PYTHONPATH="$ROOT/scripts:$ROOT/vlmevalkit-videochat3${PYTHONPATH:+:$PYTHONPATH}"
NAME=vc3-tas-lvsm-chunkstate-time-vitproj-timelens-r12624-v29aligned-8xh100-gb16packs-v40
OUT="${TAS_OUTPUT:-/mnt/localssd/VideoChat3/training/$NAME}"
INIT=/mnt/localssd/VideoChat3/VideoChat3-4B-TAS-init
export TIMELENS_BENCH_ROOT=/mnt/localssd/dataset/VideoChat3/TimeLens-Bench
export LMUData=/mnt/localssd/dataset/VLMEvalKit/LMUData
install -d "$OUT"
if [[ ! -f "$OUT/reference_plan.json" ]]; then
    "$ROOT/.venv/bin/python" - "$OUT/reference_plan.json" <<'PY'
import json,sys
from pathlib import Path
from tas_lastchunk_recipe import build_plan
Path(sys.argv[1]).write_text(json.dumps(build_plan())+'\n')
PY
fi
resume_args=()
if [[ -f "$OUT/resume.pt" ]]; then resume_args+=(--resume); fi
"$ROOT/.venv/bin/torchrun" --standalone --nproc-per-node=8 \
    scripts/train_videochat3_tas.py --checkpoint "$INIT" --output "$OUT" \
    --global-batch 16 --reference-plan "$OUT/reference_plan.json" --save-every 20 \
    --wandb-name "$NAME" "${resume_args[@]}" > "$OUT/train.log" 2>&1
MODEL="$(cat "$OUT/latest_checkpoint.txt")"
"$ROOT/.venv/bin/python" scripts/inspect_videochat3_tas_checkpoint.py \
    --init "$INIT" --trained "$MODEL" --output "$OUT/checkpoint_inspection.json" \
    > "$OUT/inspection.log" 2>&1
"$ROOT/.venv/bin/python" - "$ROOT" "$OUT" "$MODEL" <<'PY'
import json,sys
from pathlib import Path
root,out,model=map(Path,sys.argv[1:])
config=json.loads((root/'vlmevalkit-videochat3/configs/videochat3_v26_timelens_bench.json').read_text())
settings=next(iter(config['model'].values())); settings['model_path']=str(model)
config['model']={'VideoChat3-4B-TAS-v40':settings}
config['data']={k.replace('_v26_', '_v40_'):v for k,v in config['data'].items()}
(out/'eval_config.json').write_text(json.dumps(config,indent=2)+'\n')
PY
cd "$ROOT/vlmevalkit-videochat3"
"$ROOT/.venv/bin/torchrun" --standalone --nproc-per-node=8 \
    run.py --config "$OUT/eval_config.json" \
    --work-dir /mnt/localssd/VideoChat3/eval/videochat3-tas-v40-timelens-bench --reuse \
    > "$OUT/eval.log" 2>&1
"$ROOT/.venv/bin/python" "$ROOT/scripts/summarize_videochat3_tas_eval.py" \
    --root /mnt/localssd/VideoChat3/eval/videochat3-tas-v40-timelens-bench \
    --output "$OUT/timelens_comparison.md"
