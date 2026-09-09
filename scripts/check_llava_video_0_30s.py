#!/usr/bin/env python3
"""Check real multi-turn tokenization and emit a bounded training smoke manifest."""
import argparse
import copy
import json
from pathlib import Path

from transformers import AutoTokenizer
from xtuner.v1.datasets import VideoChat3TokenizeFnConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/localssd/dataset/VideoChat3/LLaVA-Video-178K"))
    parser.add_argument("--processor", default="/mnt/localssd/VideoChat3/VideoChat3-4B-LACT-init")
    parser.add_argument("--manifest", type=Path, help="Check a selected prepared subset instead of the full training manifest")
    parser.add_argument("--output", choices=("r4query", "uniform", "all"), default="all")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    manifest = json.loads((args.manifest or root / "LLaVA_Video_0_30s_train_VideoChat3.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.processor, trust_remote_code=True)
    smoke, report = {}, {}
    for name, entry in manifest.items():
        with Path(entry["anno_path"]).open() as stream:
            rows = [json.loads(line) for _, line in zip(range(512), stream)]
        config = VideoChat3TokenizeFnConfig(
            processor_path=args.processor, max_length=8192, video_min_frames=64,
            video_max_frames=64, video_sample_fps=2, video_frame_multiple=4,
            frame_max_pixels=224 * 224, video_max_total_pixels=64 * 224 * 224,
            video_read_type="decord", data_augment=False,
            lact_chunk_query=args.output == "r4query",
            lact_chunk_query_mode="spatial_quarter" if args.output == "r4query" else "single",
            macro_temporal_compression_factor=4 if args.output == "uniform" else 1,
            macro_temporal_compression_mode="chunk_select_uniform" if args.output == "uniform" else "auto",
        )
        fn = config.build(tokenizer)
        checks = []
        # Exercise both shortest and longest text conversations in each source.
        ordered = sorted(rows, key=lambda r: len(json.dumps(r["messages"])))
        for row in [ordered[0], ordered[-1]]:
            fn.state = "cache"
            estimate = fn(copy.deepcopy(row), media_root=entry["media_root"])
            fn.state = "get_item"
            actual = fn(copy.deepcopy(row), media_root=entry["media_root"])
            assert estimate["num_tokens"] == len(actual["input_ids"]), (name, estimate)
            labels = actual["labels"]
            supervised = sum(int(x != -100) for x in labels)
            assert supervised > 0
            checks.append({"id": row["id"], "tokens": len(actual["input_ids"]),
                           "supervised_tokens": supervised,
                           "grid": actual["image_grid_thw"].tolist()})
        output = root / "videochat3_annotations" / f"{name}_smoke.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        smoke[name] = dict(entry, anno_path=str(output))
        report[name] = checks
    (root / "LLaVA_Video_0_30s_smoke_VideoChat3.json").write_text(json.dumps(smoke, indent=2) + "\n")
    (root / f"tokenization_check_{args.output}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
