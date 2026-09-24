"""Illustrative endpoint energy-distance plot; no measured values are used."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).resolve().parents[1] / "paper/figures/endpoint_energy_distance_placeholder"


def main() -> None:
    # Placeholder values show the intended figure layout only.
    rounds = np.arange(1, 9)
    distances = {
        "Task 1→2": [0.40, 0.28, 0.21, 0.14, 0.12, 0.11, 0.10, 0.095],
        "Task 2→3": [0.52, 0.38, 0.27, 0.22, 0.18, 0.15, 0.14, 0.13],
        "Task 3→4": [0.46, 0.31, 0.23, 0.21, 0.16, 0.145, 0.15, 0.13],
        "Task 4→5": [0.60, 0.42, 0.34, 0.25, 0.22, 0.19, 0.17, 0.16],
        "Task 5→6": [0.49, 0.36, 0.29, 0.24, 0.19, 0.18, 0.15, 0.145],
    }
    colors = ["#4f789c", "#cf8b54", "#629a77", "#ad779e", "#c56f67"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#91a0b0",
        "axes.labelcolor": "#263448",
        "xtick.color": "#526174",
        "ytick.color": "#526174",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    for (label, values), color in zip(distances.items(), colors):
        ax.plot(values, rounds, color=color, lw=1.85,
                marker="o", ms=3.5, label=label)
    ax.set(xlim=(0, 0.65), ylim=(0.7, 8.3), ylabel="IMF round")
    ax.set_xlabel(
        r"Energy distance $D_E(P,Q)=2\mathbb{E}\Vert X-Y\Vert"
        r"-\mathbb{E}\Vert X-X'\Vert-\mathbb{E}\Vert Y-Y'\Vert$",
        labelpad=10,
    )
    ax.set_xticks(np.arange(0, 0.66, 0.1))
    ax.set_yticks(rounds)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#dfe6ee", lw=0.7, linestyle=(0, (2, 3)))
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.text(0.985, 0.985, "ILLUSTRATIVE · NOT MEASURED",
             ha="right", va="top", fontsize=8, color="#748397")
    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=240)
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
