#!/usr/bin/env python3
"""Run one native generation per TimeLens subset through the production adapter."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"vlmevalkit-videochat3"))
os.environ.setdefault("TIMELENS_BENCH_ROOT", "/mnt/localssd/dataset/VideoChat3/TimeLens-Bench")
os.environ.setdefault("LMUData", "/mnt/localssd/dataset/VLMEvalKit/LMUData")


def main():
    from vlmeval.vlm.videochat3.model import VideoChat3
    from vlmeval.dataset.timelens import TimeLens_Charades, TimeLens_ActivityNet, TimeLens_QVHighlights
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = VideoChat3(args.checkpoint, use_custom_prompt=False, use_vllm=False, max_new_tokens=64)
    assert model.model.config.model_type == "videochat3_tas"
    model.model.model.vision_tower.tas.collect_metrics = True
    rows = []
    for cls in (TimeLens_Charades, TimeLens_ActivityNet, TimeLens_QVHighlights):
        dataset = cls(dataset=cls.__name__, fps=2., frames_limit=448,
                      min_pixels=784, max_pixels=50176, total_pixels=14680064,
                      check_extracted_frames=True, reason=False)
        prompt = dataset.build_prompt(0, video_llm=True)
        start = time.monotonic()
        answer = model.generate(prompt, dataset=cls.__name__)
        assert answer.strip() and "Failed to obtain answer" not in answer
        gate = model.model.model.vision_tower.tas.last_gate
        row = dict(dataset=cls.__name__, index=0, answer=answer,
                   seconds=time.monotonic()-start, terminal_gate_mean=gate.float().mean().item(),
                   memory_tokens=gate.shape[1], peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(rows, indent=2)+"\n")


if __name__ == "__main__":
    main()
