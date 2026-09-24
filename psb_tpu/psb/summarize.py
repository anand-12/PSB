"""Tables for a finished (or partly finished) grid.

    python -m psb.summarize results/phase1b_seeds [--reference psb]

Writes summary.md and summary.csv next to the runs. With a reference tag (or a
``compare_to`` entry in the grid file) it also reports paired differences:
runs with the same seed share initialisation and data order, so per-seed
differences have far less variance than a difference of two means.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

METRICS = ("average_accuracy", "forgetting", "acquisition", "backward_transfer")


def _t95(n: int) -> float:
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, max(n - 1, 1)))
    except Exception:  # pragma: no cover - scipy missing
        return 1.96


def load(grid_dir: Path) -> dict[str, dict[int, dict]]:
    table: dict[str, dict[int, dict]] = {}
    for path in sorted(grid_dir.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        if not summary.get("complete"):
            continue
        config = summary["config"]
        table.setdefault(config["tag"], {})[config["seed"]] = summary
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("grid_dir")
    parser.add_argument("--reference", default=None)
    args = parser.parse_args()
    grid_dir = Path(args.grid_dir)
    reference = args.reference
    grid_file = grid_dir / "grid.yaml"
    if reference is None and grid_file.exists():
        reference = (yaml.safe_load(grid_file.read_text()) or {}).get("compare_to")

    table = load(grid_dir)
    if not table:
        print(f"no complete runs under {grid_dir}")
        return
    rows = []
    for tag, seeds in sorted(table.items()):
        row = {"tag": tag, "n": len(seeds)}
        for metric in METRICS:
            values = np.array([s["final"][metric] for s in seeds.values() if s["final"].get(metric) is not None])
            row[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        if reference and reference in table and tag != reference:
            shared = sorted(set(seeds) & set(table[reference]))
            diffs = np.array([seeds[s]["final"]["average_accuracy"]
                              - table[reference][s]["final"]["average_accuracy"] for s in shared])
            if diffs.size:
                sd = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
                half = _t95(diffs.size) * sd / math.sqrt(diffs.size) if diffs.size > 1 else float("nan")
                row.update(paired_n=int(diffs.size), paired_diff=float(diffs.mean()),
                           paired_ci95=half)
        rows.append(row)

    keys = sorted({k for row in rows for k in row}, key=lambda k: (k != "tag", k != "n", k))
    with (grid_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"# {grid_dir.name}", ""]
    header = "| tag | n | avg acc | forgetting | acquisition |"
    if reference:
        header += f" diff vs {reference} (paired, 95% CI) |"
    lines += [header, "|" + "---|" * (header.count("|") - 1)]
    for row in rows:
        cells = [row["tag"], str(row["n"])]
        for metric in ("average_accuracy", "forgetting", "acquisition"):
            cells.append(f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_sd']:.4f}")
        if reference:
            cells.append(f"{row['paired_diff']:+.4f} ± {row['paired_ci95']:.4f} (n={row['paired_n']})"
                         if "paired_diff" in row else ("reference" if row["tag"] == reference else "-"))
        lines.append("| " + " | ".join(cells) + " |")
    (grid_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
