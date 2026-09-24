from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed GWD runs")
    parser.add_argument("paths", nargs="+", help="summary.json files or directories containing them")
    args = parser.parse_args()
    summaries = []
    for value in args.paths:
        path = Path(value)
        candidates = sorted(path.glob("**/summary.json")) if path.is_dir() else [path]
        for candidate in candidates:
            summaries.append(json.loads(candidate.read_text()))
    if not summaries:
        raise SystemExit("No summary.json files found")
    final = [summary["history"][-1] for summary in summaries]
    for metric in ["average_accuracy", "forgetting", "backward_transfer", "predictive_loss"]:
        values = [float(row[metric]) for row in final]
        standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{metric}: {statistics.mean(values):.6f} +/- {standard_deviation:.6f} (n={len(values)})")


if __name__ == "__main__":
    main()

