#!/usr/bin/env python3
"""Render complete native TimeLens scores against the fixed Base/v26 results."""
import argparse
import json
from pathlib import Path

BASE = {
    "Charades": (46.57, 31.46, 13.92, 31.48),
    "ActivityNet": (46.20, 34.42, 21.20, 33.59),
    "QVHighlights": (67.10, 54.57, 38.87, 50.42),
}
V26_MIOU = {"Charades": 36.98, "ActivityNet": 37.55, "QVHighlights": 48.75}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    found = {}
    for path in sorted(args.root.rglob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if not isinstance(value, dict) or not {"miou", "r1_03", "r1_05", "r1_07", "scored_count", "total_count"} <= value.keys():
                continue
            for subset in BASE:
                if subset.lower() in key.lower():
                    found[subset] = (path, value)
    if set(found) != set(BASE):
        raise RuntimeError(f"Missing native scores: {set(BASE)-set(found)}")
    if sum(v[1]["total_count"] for v in found.values()) != 9404:
        raise RuntimeError("Expected all 9,404 benchmark queries")
    rows = ["| Subset | Base R1@0.3 | TAS R1@0.3 | Δ | Base R1@0.5 | TAS R1@0.5 | Δ | Base R1@0.7 | TAS R1@0.7 | Δ | Base mIoU | TAS mIoU | Δ | v26 mIoU | Δ vs v26 |",
            "|---|"+"---:|"*14]
    sources = []
    for subset, base in BASE.items():
        path, value = found[subset]
        n = value["total_count"]
        if value["scored_count"] != n:
            raise RuntimeError(f"Incomplete {subset}: {value}")
        scores = [100*value[k]/n for k in ("r1_03", "r1_05", "r1_07")] + [100*value["miou"]]
        cells = [subset]
        for a, b in zip(base, scores):
            cells.extend((f"{a:.2f}", f"{b:.2f}", f"{b-a:+.2f}"))
        cells.extend((f"{V26_MIOU[subset]:.2f}", f"{scores[-1]-V26_MIOU[subset]:+.2f}"))
        rows.append("| "+" | ".join(cells)+" |")
        sources.append(f"- {subset}: `{path}` ({n} queries)")
    args.output.write_text("All values in percent; deltas in percentage points.\n\n"+"\n".join(rows)+"\n\n"+"\n".join(sources)+"\n")
    print(args.output.read_text())


if __name__ == "__main__":
    main()
