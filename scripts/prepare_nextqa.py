#!/usr/bin/env python3
"""Convert the NExT-QA open-ended training split for VideoChat3.

The annotations come from the public ``lmms-lab/NExTQA`` mirror.  Complete
training media comes from the NExT-QA folder in ``Video-R1/Video-R1-data``;
that release contains 3,868 of the 3,870 videos referenced by the annotation
split.  Rows for genuinely missing media are reported and omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from decord import VideoReader, cpu


DEFAULT_DATASET_ROOT = Path("/mnt/localssd/dataset/VideoChat3/NExTQA")
ANNOTATION_REVISION = "a0d7729e38399da9c8a70c59aa4ad7f6996d3c00"
MEDIA_REVISION = "9ecf5eff38945e9ae4958058b83c9337f54aadd4"
DATASET_NAME = "nextqa_oe_train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--video-min-frames", type=int, default=64)
    parser.add_argument("--video-max-frames", type=int, default=512)
    parser.add_argument("--video-sample-fps", type=float, default=1.0)
    parser.add_argument("--video-frame-multiple", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def find_media(media_root: Path) -> dict[str, Path]:
    media: dict[str, Path] = {}
    for path in sorted(media_root.rglob("*.mp4")):
        video_id = path.stem
        if video_id in media:
            raise ValueError(
                f"Duplicate NExT-QA video basename {video_id}: "
                f"{media[video_id]} and {path}"
            )
        if path.stat().st_size <= 0:
            raise ValueError(f"Empty media file: {path}")
        media[video_id] = path
    if not media:
        raise FileNotFoundError(f"No MP4 files found below {media_root}")
    return media


def probe_video(path: Path) -> dict:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    total_num_frames = len(reader)
    fps = float(reader.get_avg_fps())
    if total_num_frames <= 0 or fps <= 0:
        raise ValueError(
            f"Invalid video metadata for {path}: frames={total_num_frames}, fps={fps}"
        )
    height, width = map(int, reader[0].shape[:2])
    return {
        "total_num_frames": total_num_frames,
        "duration": total_num_frames / fps,
        "fps": fps,
        "width": width,
        "height": height,
        "video_backend": "decord",
    }


def build_messages(video: str, metadata: dict, question: str, answer: str) -> list[dict]:
    question = question.strip()
    answer = answer.strip()
    if not question or not answer:
        raise ValueError(f"Empty question or answer for {video}")
    return [
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
                    "text": f"<VIDEO_CONTEXT>\n{question}",
                },
            ],
        },
        {"role": "assistant", "content": answer},
    ]


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.video_min_frames <= 0:
        raise ValueError("video_min_frames must be positive")
    if args.video_max_frames < args.video_min_frames:
        raise ValueError("video_max_frames must be >= video_min_frames")
    if args.video_sample_fps <= 0:
        raise ValueError("video_sample_fps must be positive")
    if args.video_frame_multiple <= 0:
        raise ValueError("video_frame_multiple must be positive")

    dataset_root = args.dataset_root.resolve()
    annotation_path = dataset_root / "OE/train-00000-of-00001.parquet"
    media_root = dataset_root / "media/NextQA/NExTVideo"
    output_path = (
        dataset_root / "videochat3_annotations/nextqa_oe_train.videochat3.jsonl"
    )
    manifest_path = dataset_root / "NExTQA_OE_Train_VideoChat3.json"
    summary_path = dataset_root / "nextqa_conversion_summary.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)

    rows = pq.read_table(annotation_path).to_pylist()
    media = find_media(media_root)
    referenced_ids = {str(row["video"]) for row in rows}
    present_ids = sorted(referenced_ids & media.keys())
    missing_ids = sorted(referenced_ids - media.keys())

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        probed = list(executor.map(lambda video_id: probe_video(media[video_id]), present_ids))
    metadata_by_id = dict(zip(present_ids, probed, strict=True))

    annotation_metadata_by_id: dict[str, tuple[int, int, int]] = {}
    for row in rows:
        video_id = str(row["video"])
        candidate = (int(row["frame_count"]), int(row["width"]), int(row["height"]))
        previous = annotation_metadata_by_id.setdefault(video_id, candidate)
        if previous != candidate:
            raise ValueError(
                f"Inconsistent annotation metadata for {video_id}: {previous} vs {candidate}"
            )

    dimension_mismatches = []
    frame_count_mismatches = []
    for video_id, metadata in metadata_by_id.items():
        annotated_frames, annotated_width, annotated_height = annotation_metadata_by_id[video_id]
        if (metadata["width"], metadata["height"]) != (
            annotated_width,
            annotated_height,
        ):
            dimension_mismatches.append(
                {
                    "video": video_id,
                    "annotation": [annotated_width, annotated_height],
                    "media": [metadata["width"], metadata["height"]],
                }
            )
        if metadata["total_num_frames"] != annotated_frames:
            frame_count_mismatches.append(
                {
                    "video": video_id,
                    "annotation": annotated_frames,
                    "media": metadata["total_num_frames"],
                }
            )
    invalid_media_ids = {
        mismatch["video"]
        for mismatch in frame_count_mismatches
        if abs(mismatch["annotation"] - mismatch["media"])
        > max(4, int(0.01 * mismatch["annotation"]))
    }
    # The released MP4s are the executable source of truth.  Most dimension
    # mismatches are a one-pixel even-dimension transcode adjustment; one file
    # is stored at 640x480 while its annotation says 320x480 but has the exact
    # annotated frame count.  Record these differences and use decoded media
    # metadata.  Large frame-count disagreement instead indicates incomplete
    # media and is excluded before tokenization.
    for video_id in invalid_media_ids:
        metadata_by_id.pop(video_id)

    emitted_rows = 0
    dropped_rows_by_video = Counter()
    type_counts = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for source_index, row in enumerate(rows):
            video_id = str(row["video"])
            if video_id not in metadata_by_id:
                dropped_rows_by_video[video_id] += 1
                continue
            relative_video = media[video_id].relative_to(media_root).as_posix()
            item = {
                "id": f"{DATASET_NAME}:{source_index:06d}",
                "source": DATASET_NAME,
                "messages": build_messages(
                    video=relative_video,
                    metadata=metadata_by_id[video_id],
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                ),
            }
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            emitted_rows += 1
            type_counts[str(row["type"])] += 1
    temporary.replace(output_path)

    manifest = {
        DATASET_NAME: {
            "anno_path": str(output_path),
            "media_root": str(media_root),
            "sample_ratio": 1.0,
            "data_augment": False,
            "video_read_type": "decord",
            "video_sample_fps": args.video_sample_fps,
            "video_min_frames": args.video_min_frames,
            "video_max_frames": args.video_max_frames,
            "video_frame_multiple": args.video_frame_multiple,
        }
    }
    write_json_atomic(manifest_path, manifest)

    durations = [metadata["duration"] for metadata in metadata_by_id.values()]
    frame_counts = [metadata["total_num_frames"] for metadata in metadata_by_id.values()]
    summary = {
        "annotation_repo": "lmms-lab/NExTQA",
        "annotation_revision": ANNOTATION_REVISION,
        "media_repo": "Video-R1/Video-R1-data",
        "media_revision": MEDIA_REVISION,
        "dataset_root": str(dataset_root),
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256(annotation_path),
        "media_root": str(media_root),
        "input_rows": len(rows),
        "emitted_rows": emitted_rows,
        "input_unique_videos": len(referenced_ids),
        "covered_unique_videos": len(metadata_by_id),
        "missing_video_ids": missing_ids,
        "dropped_rows_by_video": dict(sorted(dropped_rows_by_video.items())),
        "type_counts": dict(sorted(type_counts.items())),
        "duration_seconds": {
            "min": min(durations),
            "max": max(durations),
            "mean": sum(durations) / len(durations),
        },
        "frame_count": {
            "min": min(frame_counts),
            "max": max(frame_counts),
            "mean": sum(frame_counts) / len(frame_counts),
        },
        "frame_count_mismatch_count": len(frame_count_mismatches),
        "frame_count_mismatch_examples": frame_count_mismatches[:20],
        "dimension_mismatch_count": len(dimension_mismatches),
        "dimension_mismatch_examples": dimension_mismatches[:20],
        "invalid_media_ids": sorted(invalid_media_ids),
        "recipe": {
            "video_read_type": "decord",
            "video_sample_fps": args.video_sample_fps,
            "video_min_frames": args.video_min_frames,
            "video_max_frames": args.video_max_frames,
            "video_frame_multiple": args.video_frame_multiple,
            "sampling": "uniform over the full video",
        },
        "output_annotation": str(output_path),
        "output_annotation_sha256": sha256(output_path),
        "manifest": str(manifest_path),
    }
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
