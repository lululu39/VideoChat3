#!/usr/bin/env python3
"""Download, extract, and validate the official TimeLens-Bench release."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "TencentARC/TimeLens-Bench"
REVISION = "5fc78c4b401b2dadf7a3a4355d51d566ff28e0c9"
DEFAULT_ROOT = Path("/mnt/localssd/dataset/VideoChat3/TimeLens-Bench")
EXPECTED = {
    "activitynet": {"videos": 1455, "annotations": 4500},
    "charades": {"videos": 1313, "annotations": 3363},
    "qvhighlights": {"videos": 1511, "annotations": 1541},
}


def extract_archive(archive: Path, video_root: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (video_root / member.name).resolve()
            if not target.is_relative_to(video_root.resolve()):
                raise ValueError(f"Unsafe archive member in {archive}: {member.name}")
        handle.extractall(video_root, filter="data")


def annotation_stats(dataset_root: Path, subset: str) -> tuple[set[str], int]:
    path = dataset_root / f"{subset}-timelens.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    annotation_count = sum(
        min(len(item["spans"]), len(item["queries"])) for item in data.values()
    )
    return set(data), annotation_count


def validate(dataset_root: Path) -> dict[str, object]:
    results: dict[str, object] = {}
    for subset, expected in EXPECTED.items():
        video_ids, annotation_count = annotation_stats(dataset_root, subset)
        video_dir = dataset_root / "videos" / subset
        actual_paths = list(video_dir.glob("*.mp4"))
        actual_ids = {path.stem for path in actual_paths}
        missing = sorted(video_ids - actual_ids)
        if missing:
            raise FileNotFoundError(
                f"{subset}: {len(missing)} annotation videos are missing; "
                f"examples={missing[:5]}"
            )
        if len(video_ids) != expected["videos"]:
            raise ValueError(
                f"{subset}: expected {expected['videos']} annotation videos, got {len(video_ids)}"
            )
        if annotation_count != expected["annotations"]:
            raise ValueError(
                f"{subset}: expected {expected['annotations']} annotations, got {annotation_count}"
            )
        results[subset] = {
            "annotation_videos": len(video_ids),
            "extracted_mp4s": len(actual_paths),
            "annotations": annotation_count,
            "bytes": sum(path.stat().st_size for path in actual_paths),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--extract-workers", type=int, default=3)
    parser.add_argument("--delete-archives", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=dataset_root,
        allow_patterns=["README.md", "*.json", "video_shards/**/*.tar.gz"],
        max_workers=4,
    )

    video_root = dataset_root / "videos"
    video_root.mkdir(parents=True, exist_ok=True)
    archives = sorted((dataset_root / "video_shards").glob("**/*.tar.gz"))
    if archives:
        with ThreadPoolExecutor(max_workers=max(1, args.extract_workers)) as pool:
            list(pool.map(lambda archive: extract_archive(archive, video_root), archives))

    subsets = validate(dataset_root)
    summary = {
        "repo_id": REPO_ID,
        "revision": args.revision,
        "dataset_root": str(dataset_root),
        "subsets": subsets,
        "total_videos": sum(item["annotation_videos"] for item in subsets.values()),
        "total_annotations": sum(item["annotations"] for item in subsets.values()),
        "total_video_bytes": sum(item["bytes"] for item in subsets.values()),
    }
    summary_path = dataset_root / "timelens_bench_prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.delete_archives and (dataset_root / "video_shards").exists():
        shutil.rmtree(dataset_root / "video_shards")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
