"""Check a grid's ``gate`` conditions; exit 1 if any fails.

    python -m psb.gate grids/phase0_parity.yaml

Used after the parity phase so the long sweeps only start once the JAX port
reproduces the PyTorch numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def block_scores(grid_dir: Path, block: str, metric: str) -> dict[int, float]:
    scores = {}
    for path in grid_dir.glob("*/summary.json"):
        summary = json.loads(path.read_text())
        if summary.get("complete") and summary["config"]["block"] == block:
            scores[summary["config"]["seed"]] = summary["final"][metric]
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("grid")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    grid = yaml.safe_load(Path(args.grid).read_text())
    grid_dir = Path(args.results) / grid["name"]
    failed = False
    for check in grid.get("gate", []):
        metric = check.get("metric", "average_accuracy")
        if "target" in check:
            scores = block_scores(grid_dir, check["block"], metric)
            if not scores:
                print(f"FAIL {check['block']}: no complete runs"); failed = True; continue
            mean = float(np.mean(list(scores.values())))
            ok = abs(mean - check["target"]) <= check["tolerance"]
            print(f"{'PASS' if ok else 'FAIL'} {check['block']} {metric} = {mean:.4f} "
                  f"(target {check['target']} ± {check['tolerance']}, n={len(scores)})")
            failed |= not ok
        elif "compare" in check:
            a, b = check["compare"]
            sa, sb = block_scores(grid_dir, a, metric), block_scores(grid_dir, b, metric)
            shared = sorted(set(sa) & set(sb))
            if not shared:
                print(f"FAIL {a} vs {b}: no shared seeds"); failed = True; continue
            diff = float(np.mean([sa[s] - sb[s] for s in shared]))
            ok = diff > 0 if check.get("expect", "greater") == "greater" else diff < 0
            print(f"{'PASS' if ok else 'FAIL'} {a} - {b} = {diff:+.4f} over {len(shared)} seeds "
                  f"(expect {check.get('expect', 'greater')})")
            failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
