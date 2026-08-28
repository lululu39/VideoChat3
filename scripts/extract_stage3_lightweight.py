#!/usr/bin/env python3
"""Extract the retained Stage 3 lightweight tar shards into local media roots."""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path


DEFAULT_DATASET_ROOT = Path(
    "/mnt/localssd/dataset/VideoChat3/VideoChat3-Stage3-Training-Data"
)

VIDEO_ROOT_NAMES = {
    "pnorm2:s3://cinepile/ori_video/": "cinepile",
    "p2:s3://tgif/": "tgif",
    "p2:s3://tvqa/frames_fps3_hq/": "tvqa",
    "pnorm2:s3://videochat3_frames/motionbench_fps4/": "motionbench",
}

MOTION_COMPONENTS = (
    "01_atomic_motion",
    "02_atomic2_extended_motion",
    "Atomic2_OpenQA",
    "Atomic_OpenQA",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--delete-tars",
        action="store_true",
        help="Permanently delete each shard only after extraction validation.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Process at most this many unfinished shards, for a bounded smoke test.",
    )
    parser.add_argument(
        "--category",
        choices=("all", "video", "motion-videos"),
        default="all",
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="Process only this shard filename; useful for a targeted smoke test.",
    )
    return parser.parse_args()


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json_atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_manifest(dataset_root: Path, category: str):
    manifest_path = dataset_root / "data" / category / "manifest.jsonl"
    by_shard = defaultdict(list)
    with manifest_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            by_shard[record["shard"]].append(record)
    return dict(by_shard)


def motion_component(media_root: str) -> str:
    marker = "/motion-videos/"
    if marker not in media_root:
        raise ValueError(f"Cannot identify Motion-Video root: {media_root}")
    component = media_root.split(marker, 1)[1].split("/", 1)[0]
    if component not in MOTION_COMPONENTS:
        raise ValueError(f"Unknown Motion-Video component: {component}")
    return component


def extraction_root(dataset_root: Path, category: str, media_root: str) -> Path:
    if category == "video":
        try:
            name = VIDEO_ROOT_NAMES[media_root]
        except KeyError as error:
            raise ValueError(f"Unknown Video media root: {media_root}") from error
        return dataset_root / "media" / "video" / name
    component = motion_component(media_root)
    return dataset_root / "media" / "motion-videos" / component / "videos"


def tar_path(dataset_root: Path, category: str, shard: str) -> Path:
    return dataset_root / "data" / category / shard


def validate_tar_headers(archive_path: Path, expected_members: set[str]) -> None:
    """Reject unsafe or unexpected top-level members before extraction."""
    actual_members = set()
    with tarfile.open(archive_path, mode="r:") as archive:
        for member in archive:
            parts = Path(member.name).parts
            if not parts or member.name.startswith("/") or ".." in parts:
                raise ValueError(f"Unsafe tar member in {archive_path}: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Unsupported tar member in {archive_path}: {member.name}")
            actual_members.add(parts[0])
    if actual_members != expected_members:
        missing = sorted(expected_members - actual_members)[:20]
        unexpected = sorted(actual_members - expected_members)[:20]
        raise ValueError(
            f"Top-level member mismatch for {archive_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )


def extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "tar",
            "--extract",
            "--file",
            str(archive_path),
            "--directory",
            str(destination),
            "--no-same-owner",
            "--no-same-permissions",
        ],
        check=True,
    )


def tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def first_media_file(path: Path) -> Path:
    if path.is_file():
        return path
    for item in path.rglob("*"):
        if item.is_file():
            return item
    raise FileNotFoundError(f"No media files under {path}")


def decode_smoke(path: Path) -> None:
    media = first_media_file(path)
    if media.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        from PIL import Image

        with Image.open(media) as image:
            image.verify()
        return
    subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def validate_extracted(destination: Path, records: list[dict]) -> dict:
    total_bytes = 0
    for index, record in enumerate(records):
        member_path = destination / record["member"]
        actual_bytes = tree_size(member_path)
        expected_bytes = int(record["input_bytes"])
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Extracted size mismatch for {member_path}: "
                f"actual={actual_bytes}, expected={expected_bytes}"
            )
        total_bytes += actual_bytes
        if index == 0:
            decode_smoke(member_path)
    return {"members": len(records), "bytes": total_bytes}


def local_media_root(dataset_root: Path, original_root: str) -> Path:
    if original_root in VIDEO_ROOT_NAMES:
        return extraction_root(dataset_root, "video", original_root)
    if "motion-videos/" in original_root:
        component = original_root.split("motion-videos/", 1)[1].split("/", 1)[0]
        if component in MOTION_COMPONENTS:
            return dataset_root / "media" / "motion-videos" / component / "videos"
    raise ValueError(f"No local media mapping for config root: {original_root}")


def write_local_config(dataset_root: Path) -> Path:
    source_path = dataset_root / "VideoChat3_Stage3_Training_Data.json"
    output_path = dataset_root / "VideoChat3_Stage3_Training_Data_local.json"
    source = json.loads(source_path.read_text())
    local = {}
    for name, config in source.items():
        annotation = dataset_root / config["anno_path"]
        if not annotation.is_file():
            continue
        rewritten = config.copy()
        rewritten["anno_path"] = str(annotation)
        rewritten["media_root"] = str(local_media_root(dataset_root, config["media_root"]))
        local[name] = rewritten
    if len(local) != 17:
        raise ValueError(f"Expected 17 retained local datasets, found {len(local)}")
    write_json_atomic(output_path, local)
    return output_path


def process_shard(
    dataset_root: Path,
    category: str,
    shard: str,
    records: list[dict],
    state: dict,
    state_path: Path,
    delete_tars: bool,
) -> None:
    state_key = f"{category}/{shard}"
    shard_state = state.setdefault("shards", {}).get(state_key, {})
    if shard_state.get("status") == "complete":
        print(f"[skip] {state_key} already complete", flush=True)
        return

    roots = {record["media_root"] for record in records}
    if len(roots) != 1:
        raise ValueError(f"Shard {state_key} spans multiple media roots: {roots}")
    destination = extraction_root(dataset_root, category, roots.pop())
    archive_path = tar_path(dataset_root, category, shard)

    if not archive_path.exists():
        if shard_state.get("status") == "verified":
            stats = validate_extracted(destination, records)
            state["shards"][state_key] = {
                "status": "complete",
                "destination": str(destination),
                **stats,
            }
            write_json_atomic(state_path, state)
            print(f"[recover] {state_key} complete after verified deletion", flush=True)
            return
        raise FileNotFoundError(f"Missing unfinished shard: {archive_path}")

    expected_members = {record["member"] for record in records}
    print(
        f"[validate-tar] {state_key}: {len(expected_members)} members -> {destination}",
        flush=True,
    )
    validate_tar_headers(archive_path, expected_members)
    print(f"[extract] {state_key}", flush=True)
    extract_tar(archive_path, destination)
    print(f"[validate-files] {state_key}", flush=True)
    stats = validate_extracted(destination, records)
    state.setdefault("shards", {})[state_key] = {
        "status": "verified",
        "destination": str(destination),
        **stats,
    }
    write_json_atomic(state_path, state)

    if delete_tars:
        archive_path.unlink()
        state["shards"][state_key]["status"] = "complete"
        write_json_atomic(state_path, state)
        print(f"[delete] {archive_path}", flush=True)
    else:
        print(f"[keep] {archive_path}", flush=True)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    state_path = dataset_root / ".extraction_state.json"
    state = load_json(state_path, {"version": 1, "shards": {}})
    categories = (
        ("motion-videos", "video") if args.category == "all" else (args.category,)
    )
    processed = 0
    for category in categories:
        manifest = load_manifest(dataset_root, category)
        for shard, records in sorted(manifest.items()):
            if args.shard is not None and shard != args.shard:
                continue
            state_key = f"{category}/{shard}"
            if state.get("shards", {}).get(state_key, {}).get("status") == "complete":
                continue
            if args.max_shards is not None and processed >= args.max_shards:
                config_path = write_local_config(dataset_root)
                print(f"[config] {config_path}", flush=True)
                return
            process_shard(
                dataset_root,
                category,
                shard,
                records,
                state,
                state_path,
                args.delete_tars,
            )
            processed += 1
    config_path = write_local_config(dataset_root)
    print(f"[config] {config_path}", flush=True)
    if args.shard is not None:
        print(f"[done] targeted shard {args.shard} is extracted and verified", flush=True)
    else:
        print("[done] all retained Stage 3 shards are extracted and verified", flush=True)


if __name__ == "__main__":
    main()
