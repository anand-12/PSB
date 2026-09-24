from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    rows = []
    for value in args.paths:
        path = Path(value)
        if path.is_dir():
            path = path / "summary.json"
        report = json.loads(path.read_text())
        final = report["final"]
        rows.append(
            {
                "method": report["method"],
                "seed": report["seed"],
                "average_accuracy": final["average_accuracy"],
                "forgetting": final["forgetting"],
                "acquisition": final["acquisition"],
                "mean_nll": sum(final["nlls"]) / len(final["nlls"]),
                "mean_ece": sum(final["eces"]) / len(final["eces"]),
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

