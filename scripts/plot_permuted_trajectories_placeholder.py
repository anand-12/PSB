"""Average accuracy against tasks seen on Permuted MNIST, by method and K.

Without --csv the trajectories are illustrative: only the final points are the
values reported in the ablation table, the path to them is simulated and no
measured values are used. With --csv the measured values are plotted. Expected
CSV columns:
    method, K, tasks_seen, mean

Colour separates methods, line style separates the particle count: solid for
K = 256, dashed for K = 128. The average accuracy is measured once a task is
finished, so the curves are marked and bend only at task boundaries.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


OUTPUT = Path(__file__).resolve().parents[1] / "paper/figures/permuted_trajectories_placeholder"

# Four hues kept apart in both hue and lightness, ordered as they appear at
# the last task, and separable under the common forms of colour blindness.
COLORS = {"PSB": "#14508c", "EWC ensemble": "#d1622b",
          "VCL ensemble": "#a8467f", "PSB (w/o bridge)": "#8f8b81"}
STYLES = {256: "-", 128: (0, (3.2, 1.4))}
INK, INK_2 = "#0b0b0b", "#52514e"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

# Final average accuracy after 10 tasks, from the ablation table.
FINAL = {("PSB", 256): 94.3, ("PSB", 128): 93.8,
         ("EWC ensemble", 256): 92.5, ("EWC ensemble", 128): 91.9,
         ("VCL ensemble", 256): 92.0, ("VCL ensemble", 128): 91.3,
         ("PSB (w/o bridge)", 256): 90.9, ("PSB (w/o bridge)", 128): 89.2}
# Accuracy on the first task alone and how quickly the average settles.
START = {"PSB": 98.1, "EWC ensemble": 98.0,
         "VCL ensemble": 97.9, "PSB (w/o bridge)": 97.8}
RATE = {"PSB": 0.34, "EWC ensemble": 0.26,
        "VCL ensemble": 0.24, "PSB (w/o bridge)": 0.19}


def illustrative_data(n_tasks=10, seed=7):
    """Interpolate between the first-task accuracy and the reported final one.

    The average decays geometrically towards its final value at a rate set by
    the method. Every curve then takes a shock shared by all methods at each
    task, since permutations differ in difficulty, and a small jitter of its
    own.
    """
    rng = np.random.default_rng(seed)
    tasks = np.arange(1, n_tasks + 1)
    shared = rng.normal(0, 0.22, n_tasks)
    shared[0] = 0.0
    data = {}
    for (method, K), end in FINAL.items():
        rate = RATE[method] * rng.lognormal(0, 0.06)
        decay = np.exp(-rate * (tasks - 1))
        decay = (decay - decay[-1]) / (1 - decay[-1])
        curve = end + (START[method] - end) * decay
        jitter = rng.normal(0, 0.13, n_tasks)
        jitter[0] = jitter[-1] = 0.0
        weight = np.linspace(0.35, 1.0, n_tasks)       # spread grows with tasks
        curve = curve + weight * shared + jitter
        curve[-1] = end
        data[(method, K)] = dict(tasks=tasks, mean=curve)
    return data


def csv_data(path):
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[(r["method"], int(r["K"]))].append(r)
    data = {}
    for key, rs in rows.items():
        rs.sort(key=lambda r: int(r["tasks_seen"]))
        data[key] = dict(tasks=np.array([int(r["tasks_seen"]) for r in rs]),
                         mean=np.array([float(r["mean"]) for r in rs]))
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, help="measured values (see docstring)")
    args = parser.parse_args()
    data = csv_data(args.csv) if args.csv else illustrative_data()

    # IEEE style: Times text and math, fonts embedded as TrueType (Type 42).
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral"],
        "font.size": 8,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    order = ["PSB", "EWC ensemble", "VCL ensemble", "PSB (w/o bridge)"]
    for method in order:
        for K in (256, 128):
            d = data[(method, K)]
            ax.plot(d["tasks"], d["mean"], color=COLORS[method], lw=1.2,
                    ls=STYLES[K], dash_capstyle="round",
                    marker="o", ms=2.6, mec="white", mew=0.5,
                    zorder=3 if method == "PSB" else 2)

    ax.set_xlim(0.8, 10.2)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(88.0, 99.0)
    ax.set_yticks([88, 91, 94, 97])
    ax.set_xlabel("Tasks seen")
    ax.set_ylabel("Average accuracy (%)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color=GRID, lw=0.5)

    handles = [Line2D([], [], color=COLORS[m], lw=1.2, label=m) for m in order]
    handles += [Line2D([], [], color=INK, lw=1.2, ls=STYLES[K],
                       label=f"$K={K}$") for K in (256, 128)]
    ax.legend(handles=handles, frameon=False, loc="lower left", ncol=2,
              fontsize=7, handlelength=1.5, columnspacing=1.0,
              borderaxespad=0.3, handletextpad=0.4, labelspacing=0.3)
    fig.tight_layout(pad=0.3)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
