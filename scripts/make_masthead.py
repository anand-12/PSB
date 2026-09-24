"""Masthead: the chain of T-1 posterior bridges.

    python scripts/make_masthead.py

Horizontal axis is the task stream. Bold vertical lines mark task boundaries,
where the sequential posteriors are drawn as rotated densities. These are
deliberately multimodal: q_t is not Gaussian, and even our approximation of it
is a mixture of K kernels, one per particle. Faint intermediate marginals
pi_s sit inside each bridge. A soft tube gives the marginal envelope, and the
paths are individual particles coloured by where they started -- the same
particles throughout, transported rather than re-fitted, so retained colour
order shows the coupling the bridge determines.

Writes paper/figures/fig_masthead.{pdf,png}.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

BLUE = "#0F4D92"
TEAL = "#42949E"
ORIGIN_LOW = "#12395F"      # paths colour-coded by starting height
ORIGIN_HIGH = "#7FC6B4"
INK = "#1A1A1A"
GREY = "#6E7684"

STATIONS = [0.065, 0.315, 0.565, 0.940]
LABELS = [r"$\hat q^{(1)}$", r"$\hat q^{(2)}$", r"$\hat q^{(3)}$",
          r"$\hat q^{(T)}$"]
ELIDE_TICK = 0.755          # the ellipsis lives on the axis, not in the tube
INNER = 3
N_PATHS = 12

def random_modes(seed: int, count: int, spread: float = 0.34):
    """A distinct multimodal silhouette per slice.

    There is no reason the posterior's mode structure should be the same at
    every beta, so each marginal gets its own: random offsets, weights and
    widths, with one dominant mode and a guaranteed minimum separation so the
    bumps stay individually visible.
    """
    rng = np.random.default_rng(seed)
    while True:
        offsets = np.sort(rng.uniform(-spread, spread, count))
        if np.min(np.diff(offsets)) > 0.075:
            break
    weights = rng.uniform(0.28, 0.95, count)
    weights[rng.integers(count)] = 1.0
    scales = rng.uniform(0.17, 0.33, count)
    return list(zip(offsets, weights, scales))


def blend(t: float) -> tuple:
    a, b = np.array(to_rgb(BLUE)), np.array(to_rgb(TEAL))
    return tuple(a + float(np.clip(t, 0.0, 1.0)) * (b - a))


def multimodal(y, centre, width, modes):
    """A rotated posterior slice.

    q_t is not Gaussian: permutation symmetry alone makes it massively
    multimodal, and our own approximation is a mixture of one kernel per
    particle. A single bell would misrepresent both.
    """
    d = np.zeros_like(y)
    for offset, weight, scale in modes:
        d += weight * np.exp(-0.5 * ((y - (centre + offset)) / (width * scale)) ** 2)
    return d / d.max()


def draw_marginal(ax, x, y, prof, scale, colour, lw, alpha, fill=False, z=6):
    curve = x + scale * prof
    ax.plot(curve, y, color=colour, lw=lw, alpha=alpha, zorder=z, solid_capstyle="round")
    if fill:
        ax.fill_betweenx(y, x, curve, color=colour, alpha=0.15, lw=0, zorder=z - 1)


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    rng = np.random.default_rng(17)
    y = np.linspace(-0.06, 1.06, 600)
    anchor_x = np.array(STATIONS)
    # Each task pulls the posterior somewhere essentially arbitrary in weight
    # space, so the centre wanders rather than drifting one way.
    centres = np.array([0.46, 0.27, 0.58, 0.37])
    widths = np.array([0.150, 0.140, 0.152, 0.145])

    fig, ax = plt.subplots(figsize=(16.0, 5.4))

    grid = np.linspace(STATIONS[0], STATIONS[-1], 1800)
    mean = np.interp(grid, anchor_x, centres)
    spread = np.interp(grid, anchor_x, widths)

    # ---- density tube -------------------------------------------------------
    for w in np.linspace(2.2, 0.20, 26):
        ax.fill_between(grid, mean - w * spread, mean + w * spread,
                        color=blend(0.5), alpha=0.026, lw=0, zorder=1)

    # ---- particle paths, coloured by where they started ---------------------
    # A marginal tube alone is not enough: every transport with these endpoints
    # shares the same marginals. What the bridge determines is the *coupling*,
    # so each path's origin is encoded in its colour -- retained order reads as
    # layered bands, crossings as mixing.
    starts = np.sort(np.linspace(-1.45, 1.45, N_PATHS)
                     + rng.normal(scale=0.13, size=N_PATHS))
    paths = []
    for u in starts:
        knots = centres + u * widths * 0.92 + rng.normal(scale=0.028, size=4)
        smooth = np.interp(grid, anchor_x, knots)
        walk = np.cumsum(rng.normal(scale=0.0036, size=grid.size))
        walk -= np.linspace(0, walk[-1], grid.size)
        paths.append(smooth + walk)

    lo, hi_c = np.array(to_rgb(ORIGIN_LOW)), np.array(to_rgb(ORIGIN_HIGH))
    for k, path in enumerate(paths):
        colour = tuple(lo + (k / max(N_PATHS - 1, 1)) * (hi_c - lo))
        ax.plot(grid, path, color=colour, lw=1.45, alpha=0.88,
                zorder=4, solid_joinstyle="round")

    # ---- intermediate marginals --------------------------------------------
    slice_id = 0
    for seg in range(len(STATIONS) - 1):
        x0, x1 = STATIONS[seg], STATIONS[seg + 1]
        for j in range(1, INNER + 1):
            slice_id += 1
            xm = x0 + (x1 - x0) * j / (INNER + 1)
            c = float(np.interp(xm, anchor_x, centres))
            w = float(np.interp(xm, anchor_x, widths))
            t = (xm - STATIONS[0]) / (STATIONS[-1] - STATIONS[0])
            ax.plot([xm, xm], [-0.05, 1.05], color=GREY, lw=0.9,
                    ls=(0, (5, 4)), alpha=0.55, zorder=2)
            inner = random_modes(401 + 13 * slice_id, count=5, spread=0.30)
            draw_marginal(ax, xm, y, multimodal(y, c, w, inner),
                          0.048, blend(t), 1.5, 0.42, z=5)

    # ---- task boundaries ----------------------------------------------------
    for i, x in enumerate(STATIONS):
        colour = blend(i / (len(STATIONS) - 1))
        ax.plot([x, x], [-0.08, 1.08], color=INK, lw=2.8, zorder=7)
        modes = random_modes(101 + 7 * i, count=int(6 + (i % 2)))
        draw_marginal(ax, x, y, multimodal(y, centres[i], widths[i], modes),
                      0.074, colour, 3.0, 1.0, fill=True, z=8)

    # ---- task line ----------------------------------------------------------
    base = -0.24
    ax.annotate("", xy=(0.988, base), xytext=(0.012, base),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color=INK, mutation_scale=18))
    for i, x in enumerate(STATIONS):
        ax.plot([x, x], [base - 0.030, base + 0.030], color=INK, lw=1.8)
        ax.text(x, base - 0.082, LABELS[i], ha="center", va="top",
                fontsize=27, color=blend(i / (len(STATIONS) - 1)))
    ax.text(ELIDE_TICK, base - 0.082, r"$\cdots$", ha="center", va="top",
            fontsize=27, color=INK)

    # ---- pi_s: one intermediate tempered target, named where it sits -------
    x0, x1 = STATIONS[0], STATIONS[1]
    xm = x0 + (x1 - x0) * 2 / (INNER + 1)
    ax.plot([xm, xm], [base - 0.022, base + 0.022], color=GREY, lw=1.4)
    ax.text(xm, base - 0.082, r"$\pi^{(1)}_s$", ha="center", va="top",
            fontsize=24, color=GREY)

    ax.set_xlim(-0.01, 1.05)
    ax.set_ylim(base - 0.30, 1.16)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"fig_masthead.{suffix}", dpi=400,
                    bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}/fig_masthead.pdf (+png)")


if __name__ == "__main__":
    main()
