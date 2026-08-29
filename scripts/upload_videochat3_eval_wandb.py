#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import wandb


MODEL_NAMES = {
    "VideoChat3-4B-Base": "Base",
    "VideoChat3-4B-LACT-Lite31K": "LACT",
}
BENCHMARK_NAMES = {
    "video_mme_short": "Video-MME Short",
    "video_mme_long": "Video-MME Long",
    "mvbench_64frame": "MVBench 64frame",
    "mmbench_dev_en": "MMBench DEV EN V1.1",
}


def flatten_json(value, prefix=""):
    metrics = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            metrics.update(flatten_json(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            metrics[prefix] = numeric
    elif isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            pass
        else:
            if math.isfinite(numeric):
                metrics[prefix] = numeric
    return metrics


def csv_metrics(path: Path):
    frame = pd.read_csv(path)
    metrics = {}
    for row_idx, row in frame.iterrows():
        label = str(row.iloc[0]) if len(row) else str(row_idx)
        for column, value in row.items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                metrics[f"{label}/{column}"] = float(value)
    return metrics


def collect_metrics(output_root: Path):
    metrics = {}
    for path in output_root.rglob("*_rating.json"):
        relative = path.relative_to(output_root).as_posix()
        content = json.loads(path.read_text())
        for key, value in flatten_json(content).items():
            metrics[f"metrics/{relative}/{key}"] = value
    for path in output_root.rglob("*_acc*.csv"):
        relative = path.relative_to(output_root).as_posix()
        for key, value in csv_metrics(path).items():
            metrics[f"metrics/{relative}/{key}"] = value
    return metrics


def find_native_result(output_root: Path, model_name: str, pattern: str) -> Path:
    candidates = [
        path
        for path in (output_root / model_name).glob(f"T*/{pattern}")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No result matching {model_name}/T*/{pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def collect_core_scores(output_root: Path):
    scores = {}
    for model_name, display_name in MODEL_NAMES.items():
        short_path = find_native_result(
            output_root,
            model_name,
            "*Video-MME_short*_rating.json",
        )
        long_path = find_native_result(
            output_root,
            model_name,
            "*Video-MME_long*_rating.json",
        )
        mvbench_path = find_native_result(
            output_root,
            model_name,
            "*MVBench_MP4_64frame_rating.json",
        )
        mmbench_path = find_native_result(
            output_root,
            model_name,
            "*MMBench_DEV_EN_V11_acc.csv",
        )

        short_rating = json.loads(short_path.read_text())
        long_rating = json.loads(long_path.read_text())
        mvbench_rating = json.loads(mvbench_path.read_text())
        mmbench_rating = pd.read_csv(mmbench_path).iloc[0]
        scores[display_name] = {
            "video_mme_short": float(short_rating["overall"]["overall"]) * 100,
            "video_mme_long": float(long_rating["overall"]["overall"]) * 100,
            "mvbench_64frame": float(mvbench_rating["overall"][0])
            / float(mvbench_rating["overall"][1])
            * 100,
            "mmbench_dev_en": float(mmbench_rating["Overall"]) * 100,
        }
    return scores


def log_core_visualizations(run, scores):
    comparison = wandb.Table(
        columns=["benchmark", "base", "lact", "delta_pp"],
    )
    for benchmark, display_name in BENCHMARK_NAMES.items():
        base = scores["Base"][benchmark]
        lact = scores["LACT"][benchmark]
        delta = lact - base
        comparison.add_data(display_name, base, lact, delta)
        run.summary[f"core/base/{benchmark}"] = base
        run.summary[f"core/lact/{benchmark}"] = lact
        run.summary[f"core/delta_pp/{benchmark}"] = delta

        chart_table = wandb.Table(columns=["model", "score"])
        chart_table.add_data("Base", base)
        chart_table.add_data("LACT", lact)
        run.log(
            {
                f"core_eval/{benchmark}_bar": wandb.plot.bar(
                    chart_table,
                    "model",
                    "score",
                    title=f"{display_name}: Base vs LACT",
                )
            }
        )
    run.log({"core_eval/comparison_table": comparison})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--display-name")
    parser.add_argument("--entity", default="LVSM-Experiment")
    parser.add_argument("--project", default="videochat3")
    parser.add_argument("--skip-artifact", action="store_true")
    parser.add_argument("--dashboard-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = {} if args.dashboard_only else collect_metrics(args.output_root)
    if not args.dashboard_only and not metrics:
        raise RuntimeError(f"No evaluation metric files found under {args.output_root}")

    eval_config = json.loads(args.eval_config.read_text())
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.display_name or args.run_name,
        id=args.run_name,
        resume="allow",
        group="videochat3-lact-eval",
        job_type="eval",
        tags=["videochat3-4b", "lact", "core-eval-v1", "base-vs-lact"],
        config={"eval_config": eval_config},
    )
    if metrics:
        run.summary.update(metrics)
    run.summary["evaluation_complete"] = True
    log_core_visualizations(run, collect_core_scores(args.output_root))

    if not args.skip_artifact and not args.dashboard_only:
        artifact = wandb.Artifact(
            name="videochat3-lact-core-eval-v1-results",
            type="evaluation",
            metadata={"run_name": args.run_name},
        )
        artifact.add_dir(str(args.output_root))
        run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
