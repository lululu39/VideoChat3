#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download


ANNOTATION_REPO = "MCG-NJU/VideoChat3-Training-Data-Annotations"
ANNOTATION_REVISION = "67763d972bcefa1d812119f63e10c97a3e5b19f6"

FULL_ANNOTATIONS = {
    "full_cinepile_lv_qa": {
        "filename": (
            "videochat3_data_annotations/video/Gemini20260130/cinepile/"
            "VideoChat3_lv_cinepile_train_sample_qa_processed_xtuner_clean.jsonl"
        ),
        "media_key": "cinepile",
    },
    "full_cinepile_lv_qa_short": {
        "filename": (
            "videochat3_data_annotations/video/Gemini20260130/cinepile/"
            "VideoChat3_lv_cinepile_train_sample_qa_processed_xtuner_short_clean.jsonl"
        ),
        "media_key": "cinepile",
    },
    "full_cinepile_mc_6k": {
        "filename": (
            "videochat3_data_annotations/video/vflash_video/"
            "cinepile_v2_mc_subtitles_merged_6k.jsonl"
        ),
        "media_key": "cinepile",
    },
    "full_cinepile_mc_9k": {
        "filename": (
            "videochat3_data_annotations/video/vflash_video/"
            "cinepile_v2_mc_subtitles_merged_9k_clean.jsonl"
        ),
        "media_key": "cinepile",
    },
    "full_tgif_count": {
        "filename": (
            "videochat3_data_annotations/video/vflash_video/"
            "vqa_tgif_count-openend_qa_train_openend_26839_26839.jsonl"
        ),
        "media_key": "tgif",
        "video_read_type": "gif",
        "video_sample_fps": 4,
    },
    "full_tgif_transition": {
        "filename": (
            "videochat3_data_annotations/video/vflash_video/"
            "vqa_tgif_transition_qa-tgif_transition_qa-train_53k_52696.jsonl"
        ),
        "media_key": "tgif",
        "video_read_type": "gif",
        "video_sample_fps": 4,
    },
    "full_tvqa": {
        "filename": (
            "videochat3_data_annotations/video/vflash_video/"
            "vqa_tvqa-tvqa_123k_122039_merged_subtitles_17435_clean.jsonl"
        ),
        "media_key": "tvqa",
        "video_read_type": "img2",
    },
}

MEDIA_ROOTS = {
    "cinepile": "media/video/cinepile",
    "tgif": "media/video/tgif",
    "tvqa": "media/video/tvqa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build full local Stage 3 annotations whose media exists in the "
            "downloaded lightweight media pools."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/mnt/localssd/dataset/VideoChat3/VideoChat3-Stage3-Training-Data"
        ),
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path(
            "/mnt/localssd/dataset/VideoChat3/VideoChat3-Training-Data-Annotations"
        ),
    )
    return parser.parse_args()


def iter_video_url_parts(item: dict):
    for message in item.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "video_url":
                video_url = part.get("video_url")
                if isinstance(video_url, dict) and video_url.get("url"):
                    yield video_url


def resolve_reference(media_root: Path, reference: str) -> str | None:
    if (media_root / reference).exists():
        return reference
    if reference.endswith("/"):
        mp4_reference = reference.rstrip("/") + ".mp4"
        if (media_root / mp4_reference).is_file():
            return mp4_reference
    return None


def filter_annotation(
    source: Path,
    destination: Path,
    media_root: Path,
) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    rows = 0
    kept_rows = 0
    repaired_references = 0
    missing_references = []
    references = []
    with source.open() as input_stream, temporary.open("w") as output_stream:
        for line in input_stream:
            if not line.strip():
                continue
            rows += 1
            item = json.loads(line)
            resolved_parts = []
            keep = True
            for video_url in iter_video_url_parts(item):
                reference = video_url["url"]
                resolved = resolve_reference(media_root, reference)
                if resolved is None:
                    missing_references.append(reference)
                    keep = False
                    break
                resolved_parts.append((video_url, reference, resolved))
            if not keep or not resolved_parts:
                continue
            for video_url, original, resolved in resolved_parts:
                if original != resolved:
                    video_url["url"] = resolved
                    repaired_references += 1
                references.append(resolved)
            output_stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept_rows += 1
    temporary.replace(destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "rows": rows,
        "kept_rows": kept_rows,
        "dropped_rows": rows - kept_rows,
        "reference_occurrences": len(references),
        "unique_references": len(set(references)),
        "repaired_references": repaired_references,
        "missing_reference_examples": sorted(set(missing_references))[:20],
    }


def count_config_references(config: dict) -> dict:
    rows = 0
    references_by_root = defaultdict(set)
    missing = []
    for entry in config.values():
        media_root = Path(entry["media_root"])
        with Path(entry["anno_path"]).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                rows += 1
                item = json.loads(line)
                for video_url in iter_video_url_parts(item):
                    reference = video_url["url"]
                    references_by_root[str(media_root)].add(reference)
                    if not (media_root / reference).exists():
                        missing.append(str(media_root / reference))
    return {
        "configured_rows": rows,
        "unique_references_by_root_sum": sum(
            len(references) for references in references_by_root.values()
        ),
        "missing_references": len(missing),
        "missing_reference_examples": missing[:20],
    }


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    source_config_path = (
        args.dataset_root / "VideoChat3_Stage3_Training_Data_local.json"
    )
    if not source_config_path.is_file():
        raise FileNotFoundError(source_config_path)
    source_config = json.loads(source_config_path.read_text())

    output_dir = args.dataset_root / "full_local_annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    media_roots = {
        key: args.dataset_root / relative_path
        for key, relative_path in MEDIA_ROOTS.items()
    }

    output_config = {
        name: entry
        for name, entry in source_config.items()
        if name.startswith("atomic_") or name == "motionbench_train_4998_fps4"
    }
    file_reports = {}
    for name, spec in FULL_ANNOTATIONS.items():
        source = Path(
            hf_hub_download(
                repo_id=ANNOTATION_REPO,
                repo_type="dataset",
                filename=spec["filename"],
                revision=ANNOTATION_REVISION,
                local_dir=args.annotations_root,
            )
        )
        destination = output_dir / f"{name}.jsonl"
        media_root = media_roots[spec["media_key"]]
        report = filter_annotation(source, destination, media_root)
        file_reports[name] = report
        config_entry = {
            "anno_path": str(destination),
            "media_root": str(media_root),
            "sample_ratio": 1.0,
            "data_augment": False,
        }
        for optional_key in ("video_read_type", "video_sample_fps"):
            if optional_key in spec:
                config_entry[optional_key] = spec[optional_key]
        output_config[name] = config_entry

    output_config_path = (
        args.dataset_root / "VideoChat3_Stage3_Training_Data_full_local.json"
    )
    write_json(output_config_path, output_config)
    coverage = count_config_references(output_config)
    summary = {
        "annotation_repo": ANNOTATION_REPO,
        "annotation_revision": ANNOTATION_REVISION,
        "source_config": str(source_config_path),
        "output_config": str(output_config_path),
        "dataset_entries": len(output_config),
        "files": file_reports,
        **coverage,
    }
    if coverage["missing_references"]:
        raise RuntimeError(
            f"Full local config has missing references: "
            f"{coverage['missing_reference_examples']}"
        )
    summary_path = args.dataset_root / "full_local_annotations_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
