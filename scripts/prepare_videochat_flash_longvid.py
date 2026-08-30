#!/usr/bin/env python3
"""Convert the released VideoChat-Flash LongVid subset for VideoChat3.

The upstream Stage 3 recipe treats each extracted JPG directory as a 1 FPS
video, samples 64--512 frames, and rounds the sampled length down to a multiple
of four. This script preserves that contract in VideoChat3 JSONL metadata and
builds a local dataset collection manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


DEFAULT_DATASET_ROOT = Path(
    "/mnt/localssd/dataset/VideoChat3/VideoChat-Flash-Training-Data"
)
SOURCE_REVISION = "be87f5516a709be079cec8b727dd2287bf2dd70f"
FRAME_NAME = re.compile(r"^(\d{5})\.jpg$")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    annotation: str
    media_dir: str
    output: str


DATASETS = (
    DatasetSpec(
        name="videochat_flash_ego4dhcap_eventunderstanding_2k",
        annotation=(
            "ego4dhcap_eventunderstanding-longvideo_annos-"
            "ego4dhcap_eventunderstanding_2k_2k.json"
        ),
        media_dir="ego4dhcap_eventunderstanding_2k",
        output="ego4dhcap_eventunderstanding_2k.videochat3.jsonl",
    ),
    DatasetSpec(
        name="videochat_flash_htstep_eventcount_2k",
        annotation=(
            "htstep_eventcount-longvideo_annos-htstep_eventcount_2k_2k.json"
        ),
        media_dir="htstep_eventcount_2k",
        output="htstep_eventcount_2k.videochat3.jsonl",
    ),
    DatasetSpec(
        name="videochat_flash_htstep_eventrelationship_1k",
        annotation=(
            "htstep_eventrelationship-longvideo_annos-"
            "htstep_eventrelationship_1k_1k.json"
        ),
        media_dir="htstep_eventrelationship_1k",
        output="htstep_eventrelationship_1k.videochat3.jsonl",
    ),
    DatasetSpec(
        name="videochat_flash_htstep_eventunderstanding_1k",
        annotation=(
            "htstep_eventunderstanding-longvideo_annos-"
            "htstep_eventunderstanding_1k_1k.json"
        ),
        media_dir="htstep_eventunderstanding_1k",
        output="htstep_eventunderstanding_1k.videochat3.jsonl",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <dataset-root>/videochat3_annotations",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Default: <dataset-root>/VideoChatFlash_LongVid_VideoChat3.json",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--video-min-frames", type=int, default=64)
    parser.add_argument("--video-max-frames", type=int, default=512)
    parser.add_argument("--video-frame-multiple", type=int, default=4)
    return parser.parse_args()


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def inspect_frame_directory(frame_dir: Path, fps: float, frame_multiple: int) -> dict:
    if not frame_dir.is_dir():
        raise FileNotFoundError(frame_dir)

    count = 0
    minimum = None
    maximum = None
    unexpected = []
    with os.scandir(frame_dir) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue
            match = FRAME_NAME.fullmatch(entry.name)
            if match is None:
                unexpected.append(entry.name)
                continue
            index = int(match.group(1))
            count += 1
            minimum = index if minimum is None else min(minimum, index)
            maximum = index if maximum is None else max(maximum, index)

    if unexpected:
        raise ValueError(
            f"Unexpected frame names in {frame_dir}: {sorted(unexpected)[:10]}"
        )
    if count < frame_multiple:
        raise ValueError(
            f"{frame_dir} has {count} frames, fewer than frame_multiple={frame_multiple}"
        )
    if minimum != 1 or maximum != count:
        raise ValueError(
            f"Frames in {frame_dir} are not contiguous 00001.jpg..{count:05d}.jpg: "
            f"min={minimum}, max={maximum}"
        )

    first_frame = frame_dir / "00001.jpg"
    last_frame = frame_dir / f"{count:05d}.jpg"
    with Image.open(first_frame) as image:
        image.verify()
    with Image.open(first_frame) as image:
        width, height = image.size
    with Image.open(last_frame) as image:
        image.verify()
        if image.size != (width, height):
            raise ValueError(
                f"Frame size changes in {frame_dir}: first={(width, height)}, "
                f"last={image.size}"
            )

    return {
        "total_num_frames": count,
        "duration": count / fps,
        "fps": fps,
        "width": width,
        "height": height,
        "video_backend": "img2",
    }


def convert_conversations(conversations: list[dict], video: str, metadata: dict) -> list[dict]:
    role_map = {"human": "user", "gpt": "assistant"}
    converted = []
    video_inserted = False
    for turn in conversations:
        try:
            role = role_map[turn["from"]]
            value = turn["value"]
        except KeyError as error:
            raise ValueError(f"Malformed conversation turn: {turn}") from error
        if not isinstance(value, str):
            raise TypeError(f"Conversation value must be a string: {turn}")

        placeholder_count = value.count("<image>") + value.count("<video>")
        text = value.replace("<image>", "").replace("<video>", "").strip()
        if role == "user" and not video_inserted:
            if placeholder_count != 1:
                raise ValueError(
                    "The first user turn must contain exactly one video placeholder: "
                    f"{turn}"
                )
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": video},
                            "video_metadata": metadata,
                        },
                        {
                            "type": "text",
                            "text": "<VIDEO_CONTEXT>" + (f"\n{text}" if text else ""),
                        },
                    ],
                }
            )
            video_inserted = True
        else:
            if placeholder_count:
                raise ValueError(
                    "Only the first user turn may contain a video placeholder: "
                    f"{turn}"
                )
            converted.append({"role": role, "content": text})

    if not video_inserted:
        raise ValueError("Conversation has no user turn for video insertion")
    return converted


def convert_dataset(
    spec: DatasetSpec,
    dataset_root: Path,
    output_dir: Path,
    fps: float,
    frame_multiple: int,
) -> dict:
    source_path = dataset_root / "annotations/video" / spec.annotation
    media_root = dataset_root / "longvid_subset" / spec.media_dir
    output_path = output_dir / spec.output
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not media_root.is_dir():
        raise FileNotFoundError(media_root)

    rows = json.loads(source_path.read_text())
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON array in {source_path}")

    metadata_by_video = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for index, row in enumerate(rows):
            video = row.get("video")
            if not isinstance(video, str) or not video:
                raise ValueError(f"Missing video at row {index} in {source_path}")
            if video not in metadata_by_video:
                metadata_by_video[video] = inspect_frame_directory(
                    media_root / video,
                    fps=fps,
                    frame_multiple=frame_multiple,
                )
            messages = convert_conversations(
                row.get("conversations", []),
                video=video,
                metadata=metadata_by_video[video],
            )
            item = {
                "id": f"{spec.name}:{index:06d}",
                "source": spec.name,
                "messages": messages,
            }
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    temporary.replace(output_path)

    frame_counts = [value["total_num_frames"] for value in metadata_by_video.values()]
    resolutions = Counter(
        (value["width"], value["height"])
        for value in metadata_by_video.values()
    )
    return {
        "rows": len(rows),
        "unique_videos": len(metadata_by_video),
        "frame_count": {
            "min": min(frame_counts),
            "max": max(frame_counts),
            "mean": sum(frame_counts) / len(frame_counts),
        },
        "top_resolutions": [
            {"width": width, "height": height, "videos": count}
            for (width, height), count in resolutions.most_common(10)
        ],
        "annotation": str(output_path),
        "annotation_sha256": sha256(output_path),
        "media_root": str(media_root),
    }


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError(f"fps must be positive, got {args.fps}")
    if args.video_min_frames <= 0:
        raise ValueError("video_min_frames must be positive")
    if args.video_max_frames < args.video_min_frames:
        raise ValueError("video_max_frames must be >= video_min_frames")
    if args.video_frame_multiple <= 0:
        raise ValueError("video_frame_multiple must be positive")

    dataset_root = args.dataset_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else dataset_root / "videochat3_annotations"
    )
    manifest_path = (
        args.manifest_path.resolve()
        if args.manifest_path is not None
        else dataset_root / "VideoChatFlash_LongVid_VideoChat3.json"
    )

    reports = {}
    manifest = {}
    for spec in DATASETS:
        report = convert_dataset(
            spec,
            dataset_root=dataset_root,
            output_dir=output_dir,
            fps=args.fps,
            frame_multiple=args.video_frame_multiple,
        )
        reports[spec.name] = report
        manifest[spec.name] = {
            "anno_path": report["annotation"],
            "media_root": report["media_root"],
            "sample_ratio": 1.0,
            "data_augment": False,
            "video_read_type": "img2",
            "video_sample_fps": args.fps,
            "video_min_frames": args.video_min_frames,
            "video_max_frames": args.video_max_frames,
            "video_frame_multiple": args.video_frame_multiple,
        }

    write_json_atomic(manifest_path, manifest)
    summary = {
        "source_repo": "OpenGVLab/VideoChat-Flash-Training-Data",
        "source_revision": SOURCE_REVISION,
        "dataset_root": str(dataset_root),
        "official_recipe": {
            "video_read_type": "img2",
            "fps": args.fps,
            "video_min_frames": args.video_min_frames,
            "video_max_frames": args.video_max_frames,
            "video_frame_multiple": args.video_frame_multiple,
            "sampling": "uniform middle after rounding down to frame multiple",
        },
        "manifest": str(manifest_path),
        "datasets": reports,
        "total_rows": sum(report["rows"] for report in reports.values()),
        "total_unique_videos_by_dataset": sum(
            report["unique_videos"] for report in reports.values()
        ),
    }
    summary_path = dataset_root / "videochat3_longvid_conversion_summary.json"
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
