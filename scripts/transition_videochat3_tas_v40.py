#!/usr/bin/env python3
"""Start the fresh aligned run only after validated smoke and v39 retirement."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def training_processes(output):
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(output).encode() in args and any(b'train_videochat3_tas.py' in arg for arg in args):
            found.append(int(path.parent.name))
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-output', type=Path, required=True)
    p.add_argument('--smoke-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = args.output/'transition_status.json'
    previous = None
    deadline = time.monotonic()+7200
    while time.monotonic() < deadline:
        smoke_done = args.smoke_output/'training_complete.json'
        retired = args.old_output/'retired_at_checkpoint.json'
        phase = 'waiting_for_smoke'
        if smoke_done.exists():
            complete = json.loads(smoke_done.read_text())
            assert complete == {'completed_steps': 2, 'total_steps': 114}, complete
            cfg = json.loads((args.smoke_output/'training_config.json').read_text())
            assert cfg['optimizer_betas'] == [.9, .95] and cfg['loss_reduction'] == 'square'
            assert cfg['warmup_steps'] == 3 and cfg['native_training_template']
            metrics = [json.loads(line) for line in (args.smoke_output/'metrics.jsonl').read_text().splitlines()]
            assert len(metrics) == 2 and all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in metrics)
            assert all(r['runtime_info/global_examples_this_step'] == 112 for r in metrics)
            phase = 'waiting_for_v39_checkpoint_100'
            if retired.exists() and not training_processes(args.old_output):
                phase = 'ready'
        if phase != previous:
            status.write_text(json.dumps({'phase': phase, 'time': time.time()})+'\n')
            print(phase, flush=True); previous = phase
        if phase == 'ready':
            break
        time.sleep(5)
    else:
        raise TimeoutError('Prerequisites did not finish in two hours; v40 was not launched')
    # No concurrent writer remains on the retired run. Preserve its data and
    # mark the reason explicitly without deleting or relabeling its metrics.
    os.environ['WANDB_BASE_URL'] = 'https://api.wandb.ai'
    public = os.environ.get('WANDB_PUBLIC_API_KEY')
    if public:
        os.environ['WANDB_API_KEY'] = public
    else:
        os.environ.pop('WANDB_API_KEY', None)
    try:
        import wandb
        run = wandb.Api().run('LVSM-Experiment/videochat3/'+args.old_output.name)
        run.tags = list(dict.fromkeys(run.tags+['superseded-by-v40', 'recipe-mismatch-diagnostic', 'do-not-resume']))
        run.notes = 'Stopped after durable hf-100 at user request to align the TAS recipe with v19/v29. Retain checkpoint and history for diagnostics; fresh v40 does not resume these weights.'
        run.update()
        if run.state in ('running', 'pending'):
            run.update_state('failed')
    except Exception as error:
        (args.output/'retirement_metadata_error.txt').write_text(type(error).__name__+': '+str(error)+'\n')
    if (args.output/'training_config.json').exists():
        raise FileExistsError('v40 already has a training configuration; refusing duplicate launch')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    env['TAS_OUTPUT'] = str(args.output)
    with (args.output/'pipeline.log').open('a') as f:
        proc = subprocess.Popen(['bash', 'scripts/run_videochat3_tas_timelens_v40.sh'],
                                cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    launch = dict(pid=proc.pid, launch_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip())
    (args.output/'pipeline_pid.json').write_text(json.dumps(launch)+'\n')
    status.write_text(json.dumps({'phase':'launched', **launch, 'time':time.time()})+'\n')
    print(json.dumps(launch), flush=True)


if __name__ == '__main__':
    main()
