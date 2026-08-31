#!/usr/bin/env python3
"""Convert TimeLens-100K temporal-grounding supervision for VideoChat3.

The default ``visual`` mode removes queries whose answer depends explicitly on
speech or audio, then reproduces TimeLens' duration-balanced SFT selection with
a target size of 30K.  A full filtered manifest is emitted alongside the
balanced manifest so later experiments can choose scale without reconversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from decord import VideoReader, cpu


DEFAULT_DATASET_ROOT = Path("/mnt/localssd/dataset/VideoChat3/TimeLens-100K")
SOURCE_REVISION = "75e03f54a19b814de6dc8f5fceb19090625f4844"
OFFICIAL_AUDIO_QUERY_KEYWORDS = {
    "hear",
    "heard",
    "hears",
    "hearing",
    "sound",
    "sounded",
    "sounds",
    "sounding",
    "audio",
}
VISUAL_ONLY_EXCLUSION = re.compile(
    r"\b(?:hear(?:d|s|ing)?|sounds?|audio|music|song|sing(?:s|ing)?|voice|"
    r"narrat(?:e|es|ed|ing|ion)|mentions?|mentioned|says?|said|speaks?|"
    r"speaking|talks?|talking|discuss(?:es|ed|ing)?|explains?|explained|"
    r"describes?|described|asks?|asked|answers?|answered|reads?|reading|"
    r"announces?|announced|applau(?:se|ding)|babbl(?:e|es|ing))\b",
    re.IGNORECASE,
)
GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining "
    "its starting and ending times. The format should be: 'The event happens "
    "in <start time> - <end time> seconds'."
)


@dataclass(frozen=True)
class GroundingRecord:
    source: str
    video_path: str
    duration: float
    query: str
    spans: tuple[tuple[float, float], ...]
    source_row: int
    event_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--filter-mode",
        choices=("none", "official", "visual"),
        default="visual",
    )
    parser.add_argument("--target-size", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--video-sample-fps", type=float, default=2.0)
    parser.add_argument("--video-min-frames", type=int, default=64)
    parser.add_argument("--video-max-frames", type=int, default=448)
    parser.add_argument("--video-frame-multiple", type=int, default=4)
    parser.add_argument(
        "--video-max-total-pixels",
        type=int,
        default=14_336 * 32 * 32,
        help="Official TimeLens total visual pixel budget.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip().strip(".").strip()


def normalize_spans(span) -> tuple[tuple[float, float], ...]:
    if isinstance(span, list) and len(span) == 2 and all(
        isinstance(value, (int, float)) for value in span
    ):
        spans = [span]
    elif isinstance(span, list) and span and all(
        isinstance(value, (list, tuple)) and len(value) == 2 for value in span
    ):
        spans = span
    else:
        raise ValueError(f"Unsupported TimeLens span: {span!r}")
    return tuple((float(start), float(end)) for start, end in spans)


def format_response(spans: tuple[tuple[float, float], ...]) -> str:
    return (
        "The event happens in "
        + ", ".join(f"{start:.1f} - {end:.1f} seconds" for start, end in spans)
        + "."
    )


def official_audio_related(query: str) -> bool:
    words = query.strip("?").lower().split()
    return any(keyword in words for keyword in OFFICIAL_AUDIO_QUERY_KEYWORDS)


def excluded_by_filter(query: str, mode: str) -> bool:
    if mode == "none":
        return False
    if mode == "official":
        return official_audio_related(query)
    if mode == "visual":
        return bool(VISUAL_ONLY_EXCLUSION.search(query))
    raise ValueError(f"Unknown filter mode: {mode}")


def load_records(annotation_path: Path) -> tuple[list[GroundingRecord], dict]:
    records = []
    rows = 0
    empty_event_rows = 0
    source_video_counts = Counter()
    source_event_counts = Counter()
    seen_paths = set()
    with annotation_path.open() as stream:
        for source_row, line in enumerate(stream):
            if not line.strip():
                continue
            rows += 1
            raw = json.loads(line)
            required = {"source", "video_path", "duration", "events"}
            if not required.issubset(raw):
                raise ValueError(f"Missing keys at row {source_row}: {required - set(raw)}")
            video_path = str(raw["video_path"])
            if video_path in seen_paths:
                raise ValueError(f"Duplicate video_path in TimeLens source: {video_path}")
            seen_paths.add(video_path)
            duration = float(raw["duration"])
            if duration <= 0:
                raise ValueError(f"Invalid duration for {video_path}: {duration}")
            source = str(raw["source"])
            source_video_counts[source] += 1
            if not raw["events"]:
                empty_event_rows += 1
            for event_index, event in enumerate(raw["events"]):
                query = normalize_query(str(event["query"]))
                spans = normalize_spans(event["span"])
                if not query:
                    raise ValueError(f"Empty query for {video_path}, event {event_index}")
                if not all(0 <= start <= end <= duration + 1.0 for start, end in spans):
                    raise ValueError(
                        f"Out-of-range span for {video_path}: duration={duration}, spans={spans}"
                    )
                records.append(
                    GroundingRecord(
                        source=source,
                        video_path=video_path,
                        duration=duration,
                        query=query,
                        spans=spans,
                        source_row=source_row,
                        event_index=event_index,
                    )
                )
                source_event_counts[source] += 1
    return records, {
        "source_rows": rows,
        "source_events": len(records),
        "source_unique_videos": len(seen_paths),
        "empty_event_rows": empty_event_rows,
        "source_video_counts": dict(sorted(source_video_counts.items())),
        "source_event_counts": dict(sorted(source_event_counts.items())),
    }


def duration_bucket(duration: float) -> tuple[float, float]:
    for start in range(0, 240, 30):
        if start <= duration <= start + 30:
            return float(start), float(start + 30)
    return 240.0, math.inf


def duration_balanced_select(
    records: list[GroundingRecord],
    *,
    target_size: int,
    seed: int,
) -> tuple[list[GroundingRecord], dict]:
    if target_size <= 0:
        return list(records), {"target_size": target_size, "selected": len(records)}
    buckets = {
        (float(start), float(start + 30)): [] for start in range(0, 240, 30)
    }
    buckets[(240.0, math.inf)] = []
    for index, record in enumerate(records):
        buckets[duration_bucket(record.duration)].append(index)
    per_bucket = target_size // len(buckets)
    rng = random.Random(seed)
    selected_indices = set()
    report = {}
    for bounds, indices in buckets.items():
        selected_count = min(per_bucket, len(indices))
        selected_indices.update(rng.sample(indices, selected_count))
        label = f"{bounds[0]:g}-{bounds[1]:g}"
        report[label] = {"available": len(indices), "selected": selected_count}
    selected = [
        record for index, record in enumerate(records) if index in selected_indices
    ]
    return selected, {
        "target_size": target_size,
        "per_bucket_target": per_bucket,
        "selected": len(selected),
        "buckets": report,
    }


def probe_video(path: Path) -> dict:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    total_num_frames = len(reader)
    fps = float(reader.get_avg_fps())
    if total_num_frames <= 0 or fps <= 0:
        raise ValueError(f"Invalid decoded metadata: frames={total_num_frames}, fps={fps}")
    height, width = map(int, reader[0].shape[:2])
    return {
        "total_num_frames": total_num_frames,
        "duration": total_num_frames / fps,
        "fps": fps,
        "width": width,
        "height": height,
        "video_backend": "decord",
    }


def safe_probe(item: tuple[str, Path]) -> tuple[str, dict | None, str | None]:
    video_path, path = item
    try:
        return video_path, probe_video(path), None
    except Exception as error:  # report corrupt source media without hiding it
        return video_path, None, f"{type(error).__name__}: {error}"


def build_messages(record: GroundingRecord, metadata: dict) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": record.video_path},
                    "video_metadata": metadata,
                },
                {
                    "type": "text",
                    "text": "<VIDEO_CONTEXT>\n" + GROUNDING_PROMPT.format(record.query),
                },
            ],
        },
        {"role": "assistant", "content": format_response(record.spans)},
    ]


def write_annotation(
    path: Path,
    records: list[GroundingRecord],
    metadata_by_video: dict[str, dict],
    dataset_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for output_index, record in enumerate(records):
            payload = {
                "id": f"{dataset_name}:{output_index:06d}",
                "source": dataset_name,
                "messages": build_messages(record, metadata_by_video[record.video_path]),
            }
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    temporary.replace(path)


def dataset_manifest_entry(
    annotation: Path,
    media_root: Path,
    args: argparse.Namespace,
) -> dict:
    return {
        "anno_path": str(annotation),
        "media_root": str(media_root),
        "sample_ratio": 1.0,
        "data_augment": False,
        "video_read_type": "decord",
        "video_sample_fps": args.video_sample_fps,
        "video_min_frames": args.video_min_frames,
        "video_max_frames": args.video_max_frames,
        "video_frame_multiple": args.video_frame_multiple,
        "video_max_total_pixels": args.video_max_total_pixels,
    }


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.video_sample_fps <= 0:
        raise ValueError("video_sample_fps must be positive")
    if args.video_min_frames <= 0 or args.video_max_frames < args.video_min_frames:
        raise ValueError("Invalid video frame bounds")
    if args.video_frame_multiple <= 0:
        raise ValueError("video_frame_multiple must be positive")

    dataset_root = args.dataset_root.resolve()
    annotation_path = dataset_root / "timelens-100k.jsonl"
    media_root = dataset_root / "videos"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    if not media_root.is_dir():
        raise FileNotFoundError(media_root)

    records, source_summary = load_records(annotation_path)
    excluded_counts = Counter()
    filtered_records = []
    for record in records:
        if excluded_by_filter(record.query, args.filter_mode):
            excluded_counts[record.source] += 1
        else:
            filtered_records.append(record)

    annotated_duration = {}
    for record in records:
        previous = annotated_duration.setdefault(record.video_path, record.duration)
        if previous != record.duration:
            raise ValueError(f"Inconsistent duration for {record.video_path}")
    media_paths = {
        video_path: media_root / video_path for video_path in sorted(annotated_duration)
    }
    missing_media = sorted(
        video_path for video_path, path in media_paths.items() if not path.is_file()
    )
    probe_items = [
        (video_path, path)
        for video_path, path in media_paths.items()
        if video_path not in set(missing_media)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        probe_results = list(executor.map(safe_probe, probe_items))
    metadata_by_video = {
        video_path: metadata
        for video_path, metadata, error in probe_results
        if metadata is not None and error is None
    }
    decode_errors = {
        video_path: error
        for video_path, metadata, error in probe_results
        if metadata is None or error is not None
    }
    duration_mismatches = {}
    for video_path, metadata in metadata_by_video.items():
        expected = annotated_duration[video_path]
        actual = metadata["duration"]
        tolerance = max(1.0, 0.01 * expected)
        if abs(actual - expected) > tolerance:
            duration_mismatches[video_path] = {
                "annotation": expected,
                "decoded": actual,
                "tolerance": tolerance,
            }
    invalid_videos = set(missing_media) | set(decode_errors) | set(duration_mismatches)
    usable_records = [
        record for record in filtered_records if record.video_path not in invalid_videos
    ]
    balanced_records, balance_report = duration_balanced_select(
        usable_records,
        target_size=args.target_size,
        seed=args.seed,
    )

    output_dir = dataset_root / "videochat3_annotations"
    full_annotation = (
        output_dir / f"timelens_100k_{args.filter_mode}_full.videochat3.jsonl"
    )
    balanced_annotation = (
        output_dir / f"timelens_100k_{args.filter_mode}_sft30k.videochat3.jsonl"
    )
    full_name = f"timelens_100k_{args.filter_mode}_full_{len(usable_records)}"
    balanced_name = (
        f"timelens_100k_{args.filter_mode}_sft30k_{len(balanced_records)}"
    )
    write_annotation(full_annotation, usable_records, metadata_by_video, full_name)
    write_annotation(
        balanced_annotation,
        balanced_records,
        metadata_by_video,
        balanced_name,
    )

    full_manifest = dataset_root / (
        f"TimeLens100K_{args.filter_mode.title()}_Full_VideoChat3.json"
    )
    balanced_manifest = dataset_root / (
        f"TimeLens100K_{args.filter_mode.title()}_SFT30K_VideoChat3.json"
    )
    write_json_atomic(
        full_manifest,
        {
            full_name: dataset_manifest_entry(
                full_annotation,
                media_root,
                args,
            )
        },
    )
    write_json_atomic(
        balanced_manifest,
        {
            balanced_name: dataset_manifest_entry(
                balanced_annotation,
                media_root,
                args,
            )
        },
    )

    durations = [metadata["duration"] for metadata in metadata_by_video.values()]
    usable_source_counts = Counter(record.source for record in usable_records)
    balanced_source_counts = Counter(record.source for record in balanced_records)
    summary = {
        "source_repo": "TencentARC/TimeLens-100K",
        "source_revision": SOURCE_REVISION,
        "license": "bsd-3-clause",
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256(annotation_path),
        "media_root": str(media_root),
        **source_summary,
        "filter_mode": args.filter_mode,
        "excluded_query_count": sum(excluded_counts.values()),
        "excluded_query_counts_by_source": dict(sorted(excluded_counts.items())),
        "filtered_events_before_media_validation": len(filtered_records),
        "usable_full_events": len(usable_records),
        "usable_full_unique_videos": len(
            {record.video_path for record in usable_records}
        ),
        "usable_full_source_event_counts": dict(sorted(usable_source_counts.items())),
        "balanced_events": len(balanced_records),
        "balanced_unique_videos": len(
            {record.video_path for record in balanced_records}
        ),
        "balanced_source_event_counts": dict(
            sorted(balanced_source_counts.items())
        ),
        "balanced_selection": balance_report,
        "missing_media_count": len(missing_media),
        "missing_media_examples": missing_media[:20],
        "decode_error_count": len(decode_errors),
        "decode_error_examples": dict(list(sorted(decode_errors.items()))[:20]),
        "duration_mismatch_count": len(duration_mismatches),
        "duration_mismatch_examples": dict(
            list(sorted(duration_mismatches.items()))[:20]
        ),
        "decoded_video_count": len(metadata_by_video),
        "decoded_duration_seconds": {
            "min": min(durations),
            "max": max(durations),
            "mean": sum(durations) / len(durations),
        },
        "recipe": {
            "prompt": GROUNDING_PROMPT,
            "answer": "The event happens in <start> - <end> seconds.",
            "video_read_type": "decord",
            "video_sample_fps": args.video_sample_fps,
            "video_min_frames": args.video_min_frames,
            "video_max_frames": args.video_max_frames,
            "video_frame_multiple": args.video_frame_multiple,
            "video_max_total_pixels": args.video_max_total_pixels,
        },
        "full_annotation": str(full_annotation),
        "full_annotation_sha256": sha256(full_annotation),
        "full_manifest": str(full_manifest),
        "balanced_annotation": str(balanced_annotation),
        "balanced_annotation_sha256": sha256(balanced_annotation),
        "balanced_manifest": str(balanced_manifest),
    }
    summary_path = dataset_root / "timelens_100k_conversion_summary.json"
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
