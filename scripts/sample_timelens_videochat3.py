#!/usr/bin/env python3
"""Create a deterministic random subset of the prepared TimeLens JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


DEFAULT_ROOT = Path("/mnt/localssd/dataset/VideoChat3/TimeLens-100K")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sample-size", type=int, default=12_624)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    source_annotation = (
        root / "videochat3_annotations/timelens_100k_visual_sft30k.videochat3.jsonl"
    )
    source_manifest = root / "TimeLens100K_Visual_SFT30K_VideoChat3.json"
    output_annotation = (
        root
        / f"videochat3_annotations/timelens_100k_visual_random_{args.sample_size}.videochat3.jsonl"
    )
    output_manifest = (
        root / f"TimeLens100K_Visual_Random{args.sample_size}_VideoChat3.json"
    )
    summary_path = root / f"timelens_100k_random_{args.sample_size}_summary.json"
    if not source_annotation.is_file() or not source_manifest.is_file():
        raise FileNotFoundError("Prepared TimeLens source annotation/manifest is missing")

    lines = [line for line in source_annotation.read_text().splitlines() if line.strip()]
    if not 0 < args.sample_size <= len(lines):
        raise ValueError(f"sample_size must be in [1, {len(lines)}]")
    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(range(len(lines)), args.sample_size))
    dataset_name = f"timelens_100k_visual_random_{args.sample_size}_seed{args.seed}"
    videos = set()
    temporary = output_annotation.with_suffix(output_annotation.suffix + ".tmp")
    with temporary.open("w") as stream:
        for output_index, source_index in enumerate(selected_indices):
            row = json.loads(lines[source_index])
            row["id"] = f"{dataset_name}:{output_index:06d}"
            row["source"] = dataset_name
            video = next(
                item
                for item in row["messages"][0]["content"]
                if item["type"] == "video_url"
            )
            videos.add(video["video_url"]["url"])
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output_annotation)

    source_data = json.loads(source_manifest.read_text())
    _, source_entry = next(iter(source_data.items()))
    entry = dict(source_entry)
    entry["anno_path"] = str(output_annotation)
    write_json_atomic(output_manifest, {dataset_name: entry})
    write_json_atomic(
        summary_path,
        {
            "source_annotation": str(source_annotation),
            "source_annotation_sha256": sha256(source_annotation),
            "source_rows": len(lines),
            "sample_size": args.sample_size,
            "sample_fraction": args.sample_size / len(lines),
            "seed": args.seed,
            "selected_unique_videos": len(videos),
            "selected_indices_sha256": hashlib.sha256(
                json.dumps(selected_indices).encode()
            ).hexdigest(),
            "output_annotation": str(output_annotation),
            "output_annotation_sha256": sha256(output_annotation),
            "output_manifest": str(output_manifest),
        },
    )
    print(
        json.dumps(
            {
                "rows": args.sample_size,
                "unique_videos": len(videos),
                "manifest": str(output_manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
