"""Plot stored IMF bridge-cycle diagnostics; these are not energy distances."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    32: ROOT / "exp_ipf2/J_w3_ipf3_seed0/metrics.jsonl",
    128: ROOT / "exp_ipf2/K_k128_ipf2_seed0/metrics.jsonl",
}
OUTPUT = ROOT / "paper/figures/bridge_cycle_residual"


def read_rounds(path: Path) -> tuple[list[int], list[float]]:
    rounds: dict[int, float] = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if "bridge_relative" not in record:
            continue
        number = int(record["imf_iteration"])
        value = float(record["bridge_relative"])
        if number in rounds and abs(rounds[number] - value) > 1e-12:
            raise ValueError(f"Conflicting round {number} values in {path}")
        rounds[number] = value
    if not rounds:
        raise ValueError(f"No bridge-relative measurements in {path}")
    numbers = sorted(rounds)
    return numbers, [rounds[number] for number in numbers]


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8795a8",
        "axes.labelcolor": "#263448",
        "xtick.color": "#526174",
        "ytick.color": "#526174",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    for particles, color in ((32, "#527a9e"), (128, "#c66e63")):
        x, y = read_rounds(RUNS[particles])
        ax.plot(x, y, "o-", lw=2.2, ms=5.5, color=color,
                label=rf"$K={particles}$")
    ax.set(xlim=(0.8, 3.2), ylim=(0, 1.0),
           xlabel="IMF round", ylabel=r"Relative cycle residual $\mathcal{C}$")
    ax.set_xticks([1, 2, 3])
    ax.grid(axis="y", color="#dfe6ee", lw=0.7, linestyle=(0, (2, 3)))
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=240)
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
