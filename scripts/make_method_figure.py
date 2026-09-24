"""Method figure: how one task boundary is traversed.

    python scripts/make_method_figure.py

Three panels, left to right:
  (a) the tempered path, and the two operators acting at each step
  (b) reference traversal versus the learned drift
  (c) the IMF loop that makes the drift a bridge

Writes paper/figures/fig_method.{pdf,png}.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

BLUE = "#0F4D92"
BLUE_SOFT = "#3775BA"
TEAL = "#42949E"
RED = "#B64342"
GREEN = "#8BCF8B"
GREY = "#9A9A9A"
INK = "#272727"


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 1.4,
    })


def arrow(ax, start, end, color, lw=1.8, style_="-|>", ms=14, alpha=1.0, zorder=5, ls="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style_, mutation_scale=ms,
                                 lw=lw, color=color, alpha=alpha, zorder=zorder,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def blank(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# (a) the tempered path and the two operators
# ---------------------------------------------------------------------------

def panel_path(ax) -> None:
    rng = np.random.default_rng(7)
    left = np.array([0.10, 0.50]) + rng.normal(scale=[0.022, 0.085], size=(16, 2))
    right = np.array([0.90, 0.50]) + rng.normal(scale=[0.022, 0.085], size=(16, 2))
    order = np.argsort(left[:, 1]), np.argsort(right[:, 1])
    left, right = left[order[0]], right[order[1]]

    steps = np.linspace(0, 1, 120)
    for a, b in zip(left, right):
        pinch = 1.0 - 0.30 * np.sin(np.pi * steps)
        px = (1 - steps) * a[0] + steps * b[0]
        py = 0.50 + ((1 - steps) * (a[1] - 0.5) + steps * (b[1] - 0.5)) * pinch
        ax.plot(px, py, color=GREY, lw=0.9, alpha=0.55, zorder=2)

    for pts, colour in ((left, BLUE), (right, TEAL)):
        ax.scatter(pts[:, 0], pts[:, 1], s=34, c=colour, edgecolors="white",
                   linewidths=0.8, zorder=6)

    # beta axis
    ax.plot([0.10, 0.90], [0.10, 0.10], color=INK, lw=1.4, zorder=3)
    for value in np.linspace(0.10, 0.90, 9):
        ax.plot([value, value], [0.085, 0.115], color=INK, lw=1.4, zorder=3)
    ax.text(0.10, 0.035, r"$\beta=0$", ha="center", fontsize=12, color=INK)
    ax.text(0.90, 0.035, r"$\beta=1$", ha="center", fontsize=12, color=INK)
    ax.text(0.50, 0.035, r"$\pi_\beta \propto q_t^{1-\beta} q_{t+1}^{\beta}$",
            ha="center", fontsize=13, color=INK)

    # the two operators at one point on the path
    focus = np.array([0.50, 0.615])
    ax.scatter(*focus, s=110, c="white", edgecolors=INK, linewidths=1.8, zorder=8)
    arrow(ax, focus, focus + np.array([0.155, 0.075]), RED, lw=2.4, zorder=9)
    arrow(ax, focus, focus + np.array([-0.125, -0.055]), BLUE, lw=2.4, zorder=9)
    ax.text(0.695, 0.715, "likelihood", color=RED, fontsize=12, ha="center")
    ax.text(0.325, 0.522, "prior", color=BLUE, fontsize=12, ha="center")

    ax.text(0.10, 0.90, r"$q_t$", ha="center", fontsize=17, color=BLUE)
    ax.text(0.90, 0.90, r"$q_{t+1}$", ha="center", fontsize=17, color=TEAL)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
    blank(ax)
    ax.set_title("(a)  traverse the bridge", fontsize=14, color=INK, pad=8, loc="left")


# ---------------------------------------------------------------------------
# (b) reference traversal versus the learned drift
# ---------------------------------------------------------------------------

def panel_drift(ax) -> None:
    ax.axvspan(0.06, 0.94, ymin=0.0, ymax=1.0, color="#F4F6F8", zorder=0)

    # reference: many small kernel steps within the stage
    y = 0.72
    xs = np.linspace(0.10, 0.90, 17)
    for start, end in zip(xs[:-1], xs[1:]):
        wiggle = 0.028 * (1 if np.argmax(xs == start) % 2 else -1)
        arrow(ax, (start, y + wiggle), (end, y - wiggle), GREY, lw=1.5, ms=8)
    ax.text(0.50, 0.90, "reference kernel", fontsize=13, color=INK, ha="center")
    ax.text(0.50, 0.835, "many small steps per stage", fontsize=11, color=GREY, ha="center")

    # learned: one drift step, then a few corrections
    y = 0.32
    arrow(ax, (0.10, y), (0.66, y), BLUE, lw=3.4, ms=20)
    xs = np.linspace(0.66, 0.90, 5)
    for start, end in zip(xs[:-1], xs[1:]):
        arrow(ax, (start, y), (end, y), GREY, lw=1.5, ms=8)
    ax.text(0.38, y + 0.075, r"learned drift $d_\phi$", fontsize=13, color=BLUE, ha="center")
    ax.text(0.78, y + 0.075, "corrections", fontsize=11, color=GREY, ha="center")
    ax.text(0.50, 0.14, "same arrival, a quarter of the compute",
            fontsize=11, color=INK, ha="center")

    ax.scatter([0.10, 0.10], [0.72, 0.32], s=60, c=BLUE, zorder=6,
               edgecolors="white", linewidths=0.8)
    ax.scatter([0.90, 0.90], [0.72, 0.32], s=60, c=TEAL, zorder=6,
               edgecolors="white", linewidths=0.8)

    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
    blank(ax)
    ax.set_title("(b)  replace many steps with one", fontsize=14, color=INK, pad=8, loc="left")


# ---------------------------------------------------------------------------
# (c) the IMF loop
# ---------------------------------------------------------------------------

def panel_imf(ax) -> None:
    def box(centre, width, height, text, colour, fontsize=12.5):
        patch = FancyBboxPatch((centre[0] - width / 2, centre[1] - height / 2),
                               width, height, boxstyle="round,pad=0.016,rounding_size=0.03",
                               linewidth=1.8, edgecolor=colour, facecolor="white", zorder=4)
        ax.add_patch(patch)
        ax.text(*centre, text, ha="center", va="center", fontsize=fontsize,
                color=colour, zorder=5)

    box((0.28, 0.72), 0.36, 0.20, "coupling\n" + r"$(\pi_0,\pi_1)$", INK)
    box((0.74, 0.34), 0.36, 0.20, "half-bridges\n" + r"$d_\phi,\; d_\psi$", BLUE)

    ax.add_patch(FancyArrowPatch((0.45, 0.65), (0.63, 0.45),
                                 connectionstyle="arc3,rad=-0.38", arrowstyle="-|>",
                                 mutation_scale=17, lw=2.0, color=INK, zorder=3))
    ax.add_patch(FancyArrowPatch((0.58, 0.28), (0.36, 0.59),
                                 connectionstyle="arc3,rad=-0.38", arrowstyle="-|>",
                                 mutation_scale=17, lw=2.0, color=GREEN, zorder=3))
    ax.text(0.665, 0.60, "fit", fontsize=12, color=INK, ha="left")
    ax.text(0.305, 0.375, "re-run", fontsize=12, color="#4E8F4E", ha="right")

    ax.text(0.50, 0.095, "cycle error  " + r"$0.91 \rightarrow 0.11$",
            fontsize=13, color=INK, ha="center")

    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
    blank(ax)
    ax.set_title("(c)  iterate to a bridge", fontsize=14, color=INK, pad=8, loc="left")


def main() -> None:
    style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.1),
                             gridspec_kw=dict(width_ratios=[1.25, 1.0, 0.85]))
    panel_path(axes[0])
    panel_drift(axes[1])
    panel_imf(axes[2])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.03, wspace=0.10)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"fig_method.{suffix}", dpi=400, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}/fig_method.pdf (+png)")


if __name__ == "__main__":
    main()
