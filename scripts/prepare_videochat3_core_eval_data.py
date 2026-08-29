#!/usr/bin/env python3

import os
from pathlib import Path

from huggingface_hub import get_token, snapshot_download


DATA_ROOT = Path("/mnt/localssd/dataset/VLMEvalKit")
VIDEO_MME_ROOT = DATA_ROOT / "Video-MME"
MVBENCH_ROOT = DATA_ROOT / "MVBench-MP4"


def download(repo_id: str, local_dir: Path, *, revision: str | None = None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        max_workers=1,
        token=os.environ.get("HUGGINGFACE_TOKEN") or get_token(),
    )


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    download("lmms-eval/Video-MME", VIDEO_MME_ROOT)

    os.environ["VIDEOMME_ROOT"] = str(VIDEO_MME_ROOT)
    os.environ["MVBENCH_ROOT"] = str(MVBENCH_ROOT)
    os.environ.setdefault("LMUData", str(DATA_ROOT / "LMUData"))

    from vlmeval.dataset.mvbench import MVBench_MP4
    from vlmeval.dataset.videomme import VideoMME

    VideoMME(dataset="Video-MME", fps=2.0, frames_limit=1024)
    download("OpenGVLab/MVBench", MVBENCH_ROOT, revision="video")
    MVBench_MP4(dataset="MVBench_MP4", nframe=64)


if __name__ == "__main__":
    main()
