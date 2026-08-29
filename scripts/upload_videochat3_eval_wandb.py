#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd
import wandb


def flatten_json(value, prefix=""):
    metrics = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            metrics.update(flatten_json(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        metrics[prefix] = float(value)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--entity", default="LVSM-Experiment")
    parser.add_argument("--project", default="videochat3")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = collect_metrics(args.output_root)
    if not metrics:
        raise RuntimeError(f"No evaluation metric files found under {args.output_root}")

    eval_config = json.loads(args.eval_config.read_text())
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        id=args.run_name,
        resume="allow",
        group="videochat3-lact-eval",
        job_type="eval",
        tags=["videochat3-4b", "lact", "core-eval-v1", "base-vs-lact"],
        config={"eval_config": eval_config},
    )
    run.summary.update(metrics)
    run.summary["evaluation_complete"] = True

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
