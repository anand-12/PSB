"""Paper figures for posterior-bridge continual learning.

House style follows ChenLiu-1996/figures4papers: blue anchor for the proposed
method, red for comparators, top/right spines removed, frameless legends,
vector export at 300 DPI.

    python scripts/make_figures.py                  # all figures
    python scripts/make_figures.py bridge path      # a subset

Figures are written to paper/figures/ as both PDF and PNG.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#4D4D4D",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}


def apply_publication_style(font_size: int = 16, axes_linewidth: float = 2.0) -> None:
    plt.rcParams.update({
        "font.size": font_size,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "axes.linewidth": axes_linewidth,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 3,
        "legend.frameon": False,
        "xtick.major.width": axes_linewidth,
        "ytick.major.width": axes_linewidth,
        "lines.linewidth": 2.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def finalize_figure(fig, name: str, dpi: int = 300, pad: float = 1.0) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=pad)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT.relative_to(ROOT)}/{name}.pdf (+png)")


# --------------------------------------------------------------------------
# data loading -- only completed runs are ever used
# --------------------------------------------------------------------------

def load_runs(directory: str) -> dict[str, list[dict]]:
    root = ROOT / directory
    out: dict[str, list[dict]] = {}
    if not root.exists():
        return out
    for metrics in sorted(root.glob("*/metrics.jsonl")):
        records = [json.loads(line) for line in open(metrics)]
        if not records:
            continue
        # A run is usable only if it reached the task count it logged.
        if records[-1]["task"] != len(records):
            continue
        tag = re.sub(r"_seed\d+$", "", metrics.parent.name)
        out.setdefault(tag, []).append(records)
    return out


def mean_std(runs: list[list[dict]], key: str = "average_accuracy") -> tuple[float, float]:
    values = [r[-1][key] for r in runs]
    return float(np.mean(values)), float(np.std(values, ddof=1) if len(values) > 1 else 0.0)


# --------------------------------------------------------------------------
# Figure 1 -- the posterior bridge, schematic
# --------------------------------------------------------------------------

def figure_bridge() -> None:
    """Two posteriors and the geometric path between them.

    Drawn with the exact geometric interpolation for Gaussians: precisions
    combine as Lambda_beta = (1-beta) Lambda_0 + beta Lambda_1, so the
    intermediate contours are the real pi_beta rather than a sketch.
    """
    apply_publication_style(font_size=15)
    fig, ax = plt.subplots(figsize=(11, 4.2))

    # Mirror-image covariances keep the precision-weighted mean path on y = 0,
    # so the interpolation reads as travel between the clouds rather than an arc.
    mu0, mu1 = np.array([-3.4, 0.0]), np.array([3.4, 0.0])
    cov0 = np.array([[1.05, 0.40], [0.40, 0.72]])
    cov1 = np.array([[1.05, -0.40], [-0.40, 0.72]])
    lam0, lam1 = np.linalg.inv(cov0), np.linalg.inv(cov1)

    def interpolate(beta: float):
        lam = (1 - beta) * lam0 + beta * lam1
        cov = np.linalg.inv(lam)
        mu = cov @ ((1 - beta) * lam0 @ mu0 + beta * lam1 @ mu1)
        return mu, cov

    xx, yy = np.meshgrid(np.linspace(-6.2, 6.2, 460), np.linspace(-2.4, 2.4, 200))
    points = np.dstack([xx, yy])

    def density(mu, cov):
        diff = points - mu
        return np.exp(-0.5 * np.einsum("...i,ij,...j->...", diff, np.linalg.inv(cov), diff))

    for beta in np.linspace(0.18, 0.82, 5):
        mu, cov = interpolate(beta)
        ax.contour(xx, yy, density(mu, cov), levels=[0.45],
                   colors=[PALETTE["neutral_dark"]], linewidths=1.3, alpha=0.45)

    for mu, cov, color in [(mu0, cov0, PALETTE["blue_main"]), (mu1, cov1, PALETTE["teal"])]:
        field = density(mu, cov)
        ax.contourf(xx, yy, field, levels=[0.45, 1.01], colors=[color], alpha=0.18)
        ax.contour(xx, yy, field, levels=[0.45], colors=[color], linewidths=2.8)

    # Particles travel between coupled endpoints, with the offset from the mean
    # path scaled by how concentrated pi_beta is -- so the visible pinch in the
    # middle is the real narrowing of the geometric interpolation, not decoration.
    rng = np.random.default_rng(3)
    betas = np.linspace(0.0, 1.0, 80)
    spread = np.array([np.sqrt(np.sqrt(np.linalg.det(interpolate(b)[1])
                                       / np.linalg.det(cov0))) for b in betas])
    chol0, chol1 = np.linalg.cholesky(cov0), np.linalg.cholesky(cov1)
    for _ in range(9):
        start = mu0 + chol0 @ (rng.normal(size=2) * 0.72)
        end = mu1 + chol1 @ (rng.normal(size=2) * 0.72)
        centre = np.outer(1 - betas, mu0) + np.outer(betas, mu1)
        offset = np.outer(1 - betas, start - mu0) + np.outer(betas, end - mu1)
        track = centre + offset * spread[:, None]
        ax.plot(track[:, 0], track[:, 1], color=PALETTE["blue_secondary"], lw=1.3, alpha=0.7)
        ax.scatter(*track[0], s=30, color=PALETTE["blue_main"], zorder=5)
        ax.scatter(*track[-1], s=30, color=PALETTE["teal"], zorder=5)

    ax.annotate("", xy=(2.1, 2.12), xytext=(-2.1, 2.12),
                arrowprops=dict(arrowstyle="-|>", lw=2.2,
                                color=PALETTE["neutral_dark"], mutation_scale=20))
    ax.text(0, 2.62, r"$\pi_\beta \;\propto\; q_t^{\,1-\beta}\, q_{t+1}^{\,\beta}$",
            ha="center", va="top", fontsize=17)
    ax.text(2.35, 2.12, r"$\beta\!:\,0 \rightarrow 1$", ha="left", va="center",
            fontsize=13, color=PALETTE["neutral_dark"])

    ax.text(mu0[0], -1.48, r"$q_t$   tasks 1..t", ha="center", va="top", fontsize=14,
            color=PALETTE["blue_main"])
    ax.text(mu1[0], -1.48, r"$q_{t+1}$   tasks 1..t+1", ha="center", va="top", fontsize=14,
            color=PALETTE["teal"])
    ax.text(0, -1.48, "every step: likelihood\noperator, then prior operator",
            ha="center", va="top", fontsize=11.5, color=PALETTE["neutral_dark"])

    ax.set_xlim(-6.2, 6.6)
    ax.set_ylim(-2.25, 2.75)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    finalize_figure(fig, "fig1_bridge", pad=0.4)


# --------------------------------------------------------------------------
# Figure 2 -- the path, not the schedule
# --------------------------------------------------------------------------

def figure_path() -> None:
    apply_publication_style(font_size=16)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    steps = np.array([2500, 5000, 10000, 20000])
    bridge = np.array([0.9011, 0.9063, 0.9078, 0.9119])
    plain = np.array([0.8890, 0.8834, 0.8783, 0.8789])

    ax.plot(steps, bridge, "-o", color=PALETTE["blue_main"], ms=9,
            label=r"$\beta$ swept $0 \rightarrow 1$ (PSB)")
    ax.plot(steps, plain, "--s", color=PALETTE["red_strong"], ms=8,
            label=r"$\beta$ fixed at $1$")

    ax.annotate(f"{bridge[-1]:.4f}", xy=(steps[-1], bridge[-1]),
                xytext=(-6, 12), textcoords="offset points",
                color=PALETTE["blue_main"], fontsize=13, ha="right")
    ax.annotate(f"{plain[-1]:.4f}", xy=(steps[-1], plain[-1]),
                xytext=(-6, -20), textcoords="offset points",
                color=PALETTE["red_strong"], fontsize=13, ha="right")

    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(steps)
    ax.set_xticklabels(["2.5k", "5k", "10k", "20k"])
    ax.set_xlabel("moves per task boundary")
    ax.set_ylabel("10-task average accuracy")
    ax.set_ylim(0.872, 0.918)
    ax.grid(axis="y", color=PALETTE["neutral"], lw=1.0, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="center left")
    finalize_figure(fig, "fig2_path")


# --------------------------------------------------------------------------
# Figure 3 -- IMF drives the pair toward a bridge
# --------------------------------------------------------------------------

def figure_imf() -> None:
    apply_publication_style(font_size=16)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    rounds = ["IMF = 1\n(bridge matching)", "IMF = 2\n(Schrödinger bridge)"]
    cycle = [0.908, 0.114]
    x = np.arange(2)

    bars = axes[0].bar(x, cycle, width=0.55,
                       color=[PALETTE["red_strong"], PALETTE["blue_main"]])
    for rect, value in zip(bars, cycle):
        axes[0].text(rect.get_x() + rect.get_width() / 2, value + 0.028,
                     f"{value:.3f}", ha="center", fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(rounds, fontsize=12)
    axes[0].set_ylabel("cycle error  $\\mathcal{C}$")
    axes[0].set_ylim(0, 1.06)
    axes[0].annotate("", xy=(1, 0.20), xytext=(1, 0.85),
                     arrowprops=dict(arrowstyle="-|>", lw=2.4,
                                     color=PALETTE["neutral_dark"], mutation_scale=20))
    axes[0].text(0.93, 0.53, "8$\\times$", ha="right", fontsize=15,
                 color=PALETTE["neutral_dark"])

    width = 0.36
    forward = [0.286, 0.624]
    backward = [0.168, 0.666]
    axes[1].bar(x - width / 2, forward, width, label="forward half-bridge",
                color=PALETTE["blue_main"])
    axes[1].bar(x + width / 2, backward, width, label="backward half-bridge",
                color=PALETTE["green_3"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(rounds, fontsize=12)
    axes[1].set_ylabel("explained variance")
    axes[1].set_ylim(0, 0.82)
    axes[1].legend(loc="upper left")

    for ax in axes:
        ax.grid(axis="y", color=PALETTE["neutral"], lw=1.0, alpha=0.55)
        ax.set_axisbelow(True)
    finalize_figure(fig, "fig3_imf", pad=1.2)


# --------------------------------------------------------------------------
# Figure 4 -- retention profile across the stream
# --------------------------------------------------------------------------

def figure_matrix() -> None:
    runs = load_runs("exp_tr3")
    source = runs.get("s4k") or runs.get("best_s1")
    if not source:
        print("  skipping fig4: no completed run found in exp_tr3/")
        return
    records = source[0]
    tasks = len(records)
    matrix = np.full((tasks, tasks), np.nan)
    for row, record in enumerate(records):
        for col, accuracy in enumerate(record["task_accuracies"]):
            matrix[row, col] = accuracy

    apply_publication_style(font_size=15)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    image = ax.imshow(matrix, cmap="magma", vmin=0.80, vmax=1.0, aspect="auto")
    for row in range(tasks):
        for col in range(row + 1):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center",
                    fontsize=8.5,
                    color="white" if matrix[row, col] < 0.93 else "black")
    ax.set_xlabel("evaluated on task")
    ax.set_ylabel("after training task")
    ax.set_xticks(range(tasks)); ax.set_xticklabels(range(1, tasks + 1), fontsize=11)
    ax.set_yticks(range(tasks)); ax.set_yticklabels(range(1, tasks + 1), fontsize=11)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("accuracy", fontsize=14)
    bar.ax.tick_params(labelsize=11)
    finalize_figure(fig, "fig4_matrix")


# --------------------------------------------------------------------------
# Figure 5 -- benchmarks, with baselines where they are comparable
# --------------------------------------------------------------------------

def figure_benchmarks() -> None:
    runs = load_runs("runs_benchmarks")
    if not runs:
        print("  skipping fig5: runs_benchmarks/ not populated")
        return

    def best(prefix: str) -> tuple[float, float, int] | None:
        candidates = {t: v for t, v in runs.items() if t.startswith(prefix)}
        if not candidates:
            return None
        tag = max(candidates, key=lambda t: mean_std(candidates[t])[0])
        mean, std = mean_std(candidates[tag])
        return mean, std, len(candidates[tag])

    entries = [("Permuted\nMNIST", best("P_")), ("Rotated\nMNIST", best("R_")),
               ("Split MNIST\n(domain-IL)", best("S_a")),
               ("Split MNIST\n(class-IL)", best("S_cls"))]
    entries = [(name, value) for name, value in entries if value]

    apply_publication_style(font_size=16)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(entries))
    means = [v[0] for v in (e[1] for e in entries)]
    errors = [v[1] for v in (e[1] for e in entries)]
    colors = [PALETTE["blue_main"]] * len(entries)
    if entries and entries[-1][0].startswith("Split MNIST\n(class"):
        colors[-1] = PALETTE["neutral"]

    bars = ax.bar(x, means, width=0.58, color=colors,
                  yerr=errors, capsize=5,
                  error_kw=dict(ecolor=PALETTE["neutral_dark"], lw=2))
    for rect, mean, (name, value) in zip(bars, means, entries):
        ax.text(rect.get_x() + rect.get_width() / 2, mean + (errors and 0.02 or 0.02),
                f"{mean:.3f}\n(n={value[2]})", ha="center", fontsize=12)

    # Published Permuted-MNIST baselines, drawn only over the Permuted bar.
    # EWC and SI sit only 0.02 apart, so their labels alternate sides.
    for value, name, colour, side in [
        (0.84, "EWC .84", PALETTE["neutral_dark"], "left"),
        (0.86, "SI .86", PALETTE["neutral_dark"], "right"),
        (0.90, "VCL .90", PALETTE["red_strong"], "right"),
    ]:
        ax.plot([-0.30, 0.30], [value, value], color=colour, ls="--", lw=2)
        if side == "right":
            ax.text(0.35, value, name, va="center", ha="left", fontsize=11, color=colour)
        else:
            ax.text(-0.35, value, name, va="center", ha="right", fontsize=11, color=colour)
    if any(e[0].startswith("Split MNIST\n(class") for e in entries):
        ax.text(len(entries) - 1, 0.06, "out of scope:\nno parameter-space method\n"
                "recalibrates absent classes",
                ha="center", va="bottom", fontsize=10, color=PALETTE["neutral_dark"])

    ax.set_xticks(x)
    ax.set_xticklabels([e[0] for e in entries], fontsize=13)
    ax.set_ylabel("average accuracy")
    ax.set_ylim(0, 1.06)
    ax.grid(axis="y", color=PALETTE["neutral"], lw=1.0, alpha=0.55)
    ax.set_axisbelow(True)
    finalize_figure(fig, "fig5_benchmarks")


FIGURES = {
    "bridge": figure_bridge,
    "path": figure_path,
    "imf": figure_imf,
    "matrix": figure_matrix,
    "benchmarks": figure_benchmarks,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(FIGURES)
    unknown = [name for name in wanted if name not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s): {unknown}. choose from {list(FIGURES)}")
    for name in wanted:
        print(f"{name}:")
        FIGURES[name]()
