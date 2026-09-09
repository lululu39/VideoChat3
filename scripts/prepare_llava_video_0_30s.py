#!/usr/bin/env python3
"""Prepare the pinned Academic + YouTube LLaVA-Video 0–30s release.

Preserve original QA conversations; split by source video, exclude known local
benchmark videos, and validate the frames that the 64-frame recipe consumes.
All downloads, media, caches, and generated annotations stay outside Git.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from huggingface_hub import HfApi, snapshot_download

REPO = "lmms-lab/LLaVA-Video-178K"
REVISION = "6d8c562dc26d70042a0d9704d1cae58c94b89098"
ROOT = Path("/mnt/localssd/dataset/VideoChat3/LLaVA-Video-178K")
SUBSETS = ("0_30_s_academic_v0_1", "0_30_s_youtube_v0_1")
EXPECTED = {"academic": {"oe": 48468, "mc": 5753, "cap": 11985},
            "youtube": {"oe": 420200, "mc": 39353, "cap": 79346}}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def video_key(path):
    """Conservative source-ID matching across repackaged benchmark paths."""
    media_path = Path(path)
    if "youcook2" in {p.lower() for p in media_path.parts} and re.fullmatch(r"split_\d+\.mp4", media_path.name):
        return media_path.parent.name
    stem = Path(path).name
    if stem.lower().endswith((".mp4", ".mkv", ".webm", ".avi")):
        stem = stem.rsplit(".", 1)[0]
    match = re.fullmatch(r"(?:ytb_|v_)([A-Za-z0-9_-]{11}(?:_\d+(?:\.\d+)?_\d+(?:\.\d+)?)?)", stem)
    if match:
        stem = match[1]
    # Charades benchmark clips append start/end to the original five-letter ID.
    if re.match(r"^[A-Z0-9]{5}_\d", stem):
        stem = stem[:5]
    # QVHighlights and other repackagings append temporal windows to YouTube IDs.
    match = re.match(r"^([A-Za-z0-9_-]{11})_\d+(?:\.\d+)?_\d+(?:\.\d+)?$", stem)
    if match:
        stem = match[1]
    return stem


def split_for_video(path, seed=42, holdout_fraction=0.02):
    value = int(hashlib.sha256(f"{seed}:{video_key(path)}".encode()).hexdigest()[:16], 16)
    return "heldout" if value / 2**64 < holdout_fraction else "train"


def extract_archive(archive, media_root, marker_root):
    """Reject traversal and links; record completion only after the full stream."""
    marker = marker_root / (archive.name + ".json")
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("archive_bytes") == archive.stat().st_size:
            return previous
    count = size = 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            target = (media_root / member.name).resolve()
            if not target.is_relative_to(media_root.resolve()) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive member: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive member: {member.name}")
            # Interrupted extraction can safely resume by replacing partial files.
            if member.isfile() and target.is_file() and target.stat().st_size == member.size:
                count += 1
                size += member.size
                continue
            tar.extract(member, media_root, filter="data")
            if member.isfile():
                count += 1
                size += member.size
    result = {"archive_bytes": archive.stat().st_size, "files": count, "extracted_bytes": size}
    write_json(marker, result)
    print(f"Extracted {archive.name}: {count} files", flush=True)
    return result


def source_files(root):
    for domain in ("academic", "youtube"):
        subset = f"0_30_s_{domain}_v0_1"
        for task in ("oe", "mc", "cap"):
            name = (f"0_30_s_{domain}_v0_1_cap_processed.json" if task == "cap"
                    else f"0_30_s_{domain}_{task}_v0_1_qa_processed.json")
            yield domain, task, root / subset / name


def normalize_conversation(row):
    conv = row["conversations"]
    if not conv or len(conv) % 2:
        raise ValueError("Expected alternating question/answer pairs")
    pairs = []
    for i in range(0, len(conv), 2):
        q, a = conv[i:i + 2]
        if q["from"] != "human" or a["from"] != "gpt":
            raise ValueError("Unexpected conversation roles")
        question = q["value"].replace("<image>", "").replace("<video>", "").strip()
        answer = a["value"].strip()
        if not question or not answer:
            raise ValueError("Empty question or answer")
        pairs.append((question, answer))
    return pairs


def build_messages(video, metadata, pairs):
    messages = []
    for i, (question, answer) in enumerate(pairs):
        content = question
        if i == 0:
            content = [{"type": "video_url", "video_url": {"url": video},
                        "video_metadata": metadata},
                       {"type": "text", "text": "<VIDEO_CONTEXT>\n" + question}]
        messages.extend([{"role": "user", "content": content},
                         {"role": "assistant", "content": answer}])
    return messages


def sample_count(metadata):
    # Exact current min/max/rounding policy; never repeat absent source frames.
    return min(64, metadata["total_num_frames"]) // 4 * 4


def probe_video(path):
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    frames, fps = len(reader), float(reader.get_avg_fps())
    if frames < 4 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid frame count/FPS: {frames}/{fps}")
    n = min(frames, 64) // 4 * 4
    indices = np.linspace(0, frames - 1, n).astype(int)
    # Decode every sampled frame, in small batches to bound worker memory.
    shape = None
    for start in range(0, n, 4):
        batch = reader.get_batch(indices[start:start + 4])
        shape = batch.shape
    return {"total_num_frames": frames, "duration": frames / fps, "fps": fps,
            "height": int(shape[1]), "width": int(shape[2]), "video_backend": "decord"}


def local_exclusions(paths):
    excluded, sources = set(), []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == ".tsv":
            with path.open() as f:
                rows = csv.DictReader(f, delimiter="\t")
                ids = {video_key(row["video"]) for row in rows}
        elif path.suffix == ".parquet":
            import pyarrow.parquet as pq
            ids = {video_key(str(v)) for v in pq.read_table(path, columns=["video"])["video"].to_pylist()}
        elif path.suffix == ".json":
            raw = json.loads(path.read_text())
            ids = {video_key(x) for x in raw}  # TimeLens-Bench video-ID dictionaries.
        else:
            ids = {video_key(x.strip()) for x in path.read_text().splitlines() if x.strip()}
        excluded.update(ids)
        sources.append({"path": str(path), "sha256": digest(path), "video_ids": len(ids)})
    return excluded, sources


def manifest_entry(annotation, media_root):
    return {"anno_path": str(annotation), "media_root": str(media_root),
            "sample_ratio": 1.0, "data_augment": False, "video_read_type": "decord",
            "video_sample_fps": 2.0, "video_min_frames": 64, "video_max_frames": 64,
            "video_frame_multiple": 4, "frame_max_pixels": 224 * 224,
            "video_max_total_pixels": 64 * 224 * 224}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--extract-workers", type=int, default=3)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.02)
    parser.add_argument("--exclude-video-ids", type=Path, action="append", default=[])
    parser.add_argument("--delete-archives", action="store_true")
    args = parser.parse_args()
    if not 0 < args.holdout_fraction < 1:
        parser.error("holdout-fraction must be between zero and one")
    root = args.dataset_root.resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("dataset-root must be outside the Git repository")
    root.mkdir(parents=True, exist_ok=True)
    media_root = root / "videos"
    media_root.mkdir(exist_ok=True)
    if not args.skip_download:
        snapshot_download(repo_id=REPO, repo_type="dataset", revision=REVISION,
                          local_dir=root, max_workers=args.download_workers,
                          allow_patterns=["README.md", "gpt4o_qa_prompt/*"] + [f"{s}/*" for s in SUBSETS])
    remote = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    official = {f.rfilename: f.size for f in remote.siblings
                if any(f.rfilename.startswith(s + "/") for s in SUBSETS)}
    for name, size in official.items():
        path = root / name
        if not path.exists() and name.endswith(".tar.gz") and args.skip_extract:
            continue
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Missing or wrong-size official file: {name}")
    archives = sorted(root.glob("0_30_s*/*.tar.gz"))
    if not args.skip_extract:
        with ThreadPoolExecutor(max_workers=args.extract_workers) as pool:
            list(pool.map(lambda p: extract_archive(p, media_root, root / "extraction"), archives))

    # Reuse a previous exclusion snapshot to keep reruns stable across machines.
    exclusion_path = root / "benchmark_exclusions.json"
    if args.exclude_video_ids:
        excluded, exclusion_sources = local_exclusions(args.exclude_video_ids)
    elif exclusion_path.exists():
        previous = json.loads(exclusion_path.read_text())
        excluded = {video_key(v) for v in previous["video_ids"]}
        exclusion_sources = previous["sources"]
    else:
        candidates = [Path("/mnt/localssd/dataset/VLMEvalKit/MVBench-MP4/MVBench_MP4.tsv"),
                      Path("/mnt/localssd/dataset/VLMEvalKit/Video-MME/Video-MME.tsv")]
        candidates += sorted(Path("/mnt/localssd/dataset/VideoChat3/TimeLens-Bench").glob("*-timelens.json"))
        candidates += sorted(Path("/mnt/localssd/dataset/VideoChat3/NExTQA/OE").glob("*.parquet"))
        excluded, exclusion_sources = local_exclusions([p for p in candidates if p.exists()])
    write_json(exclusion_path, {"video_ids": sorted(excluded), "sources": exclusion_sources})

    video_paths, source_stats = set(), {}
    for domain, task, path in source_files(root):
        rows = json.loads(path.read_text())
        if len(rows) != EXPECTED[domain][task]:
            raise ValueError(f"Unexpected annotation count: {path}")
        video_paths.update(row["video"] for row in rows)
        source_stats[f"{domain}_{task}"] = {"records": len(rows),
            "answer_turns": sum(len(row["conversations"]) // 2 for row in rows), "sha256": digest(path)}
    missing, available = {}, set()
    for video in video_paths:
        path = (media_root / video).resolve()
        if not path.is_relative_to(media_root):
            raise ValueError(f"Unsafe referenced video: {video}")
        if not path.is_file() or not path.stat().st_size:
            missing[video] = "Missing or empty media in the extracted official release"
        else:
            available.add(video)
    cache_path = root / "decoded_metadata.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cache.update({v: {"error": error} for v, error in missing.items()})
    pending = [v for v in sorted(available) if v not in cache or
               cache[v].get("size") != (media_root / v).stat().st_size or
               cache[v].get("mtime_ns") != (media_root / v).stat().st_mtime_ns]

    def probe(video):
        path = media_root / video
        record = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        try:
            record["metadata"] = probe_video(path)
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
        return video, record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (video, record) in enumerate(pool.map(probe, pending), 1):
            cache[video] = record
            if i % 1000 == 0:
                write_json(cache_path, cache)
                print(f"Validated {i}/{len(pending)} videos", flush=True)
    write_json(cache_path, cache)
    manifests = {"train": {}, "heldout": {}, "caption_train": {}}
    conversion_stats, train_videos, heldout_videos = {}, set(), set()
    for domain, task, path in source_files(root):
        label = f"llava_0_30s_{domain}_{task}"
        rows = json.loads(path.read_text())
        stats, seen = Counter(), set()
        handles, outputs = {}, {}
        for split in ("train", "heldout"):
            output = root / "videochat3_annotations" / f"{label}_{split}.jsonl"
            output.parent.mkdir(exist_ok=True)
            outputs[split] = output
            handles[split] = output.with_suffix(".jsonl.tmp").open("w")
        for index, row in enumerate(rows):
            video = row["video"]
            if video_key(video) in excluded:
                stats["benchmark_excluded_records"] += 1
                continue
            record = cache[video]
            if "error" in record:
                stats["decode_rejected_records"] += 1
                continue
            metadata = record["metadata"]
            # Allow container rounding around the advertised 30-second boundary.
            if metadata["duration"] > 30.5:
                stats["over_30_5s_records"] += 1
                continue
            pairs = normalize_conversation(row)
            key = hashlib.sha256(json.dumps([video, pairs], ensure_ascii=False).encode()).digest()
            if key in seen:
                stats["duplicate_conversations"] += 1
                continue
            seen.add(key)
            split = split_for_video(video, args.seed, args.holdout_fraction)
            (train_videos if split == "train" else heldout_videos).add(video_key(video))
            # Held-out QA is single-turn: no preceding gold answer is exposed.
            groups = [pairs] if split == "train" or task == "cap" else [[pair] for pair in pairs]
            for group_index, group in enumerate(groups):
                result = {"id": f"{label}:{index}:{group_index}", "source": label,
                          "source_record": index, "source_id": row.get("id"), "qa_format": task,
                          "messages": build_messages(video, metadata, group)}
                handles[split].write(json.dumps(result, ensure_ascii=False) + "\n")
                stats[f"{split}_records"] += 1
                stats[f"{split}_qa_turns"] += len(group)
        for split, handle in handles.items():
            handle.close()
            output = outputs[split]
            output.with_suffix(".jsonl.tmp").replace(output)
            if stats[f"{split}_records"]:
                key = "caption_train" if task == "cap" and split == "train" else split
                if task != "cap" or split == "train":
                    manifests[key][label] = manifest_entry(output, media_root)
            stats[f"{split}_sha256"] = digest(output)
        conversion_stats[label] = dict(stats)
    assert not train_videos & heldout_videos
    for split, manifest in manifests.items():
        write_json(root / f"LLaVA_Video_0_30s_{split}_VideoChat3.json", manifest)
    valid = [v for v in video_paths if "metadata" in cache[v]]
    updates = [sample_count(cache[v]["metadata"]) // 4 - 1 for v in valid]
    summary = {"repo": REPO, "revision": REVISION, "official_download_bytes": sum(official.values()),
               "source_annotations": source_stats, "conversion": conversion_stats,
               "referenced_videos": len(video_paths), "decoded_videos": len(valid),
               "decode_errors": {v: cache[v]["error"] for v in video_paths if "error" in cache[v]},
               "missing_or_empty_videos": missing,
               "referenced_media_bytes": sum((media_root / v).stat().st_size for v in available),
               "train_source_videos": len(train_videos), "heldout_source_videos": len(heldout_videos),
               "split_seed": args.seed, "holdout_fraction": args.holdout_fraction,
               "benchmark_exclusions": str(exclusion_path), "exclusion_sources": exclusion_sources,
               "update_mean_unique_decodable_video": float(np.mean(updates)),
               "update_min_max": [min(updates), max(updates)],
               "deduplication": "exact normalized conversation per media path within each QA format",
               "recipe": manifest_entry("<annotation>", media_root)}
    write_json(root / "llava_video_0_30s_prepare_summary.json", summary)
    if args.delete_archives:
        for archive in archives:
            archive.unlink()
    print(json.dumps({k: v for k, v in summary.items() if k != "decode_errors"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
