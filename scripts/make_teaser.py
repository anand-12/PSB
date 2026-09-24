"""Single teaser figure: a Schrodinger bridge between two task posteriors.

    python scripts/make_teaser.py

Writes paper/figures/fig_sb_teaser.{pdf,png}.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

BLUE = "#0F4D92"
TEAL = "#42949E"
RED = "#B64342"
INK = "#272727"

# Blue basins rising to warm crests, in the house palette.
TERRAIN = LinearSegmentedColormap.from_list(
    "terrain_sb",
    ["#1B3A6B", "#3C6FA8", "#7FA8CE", "#C9D8E6", "#EBD9D2", "#D99C92", "#B64342"],
)

PEAK_A = np.array([-3.30, 0.00])
PEAK_B = np.array([3.30, 0.00])
WIDTH = 1.18
RIDGE = 1.55


def surface(x, y):
    """Two solution basins on a gently textured parameter landscape."""
    a = RIDGE * np.exp(-(((x - PEAK_A[0]) ** 2 + (y - PEAK_A[1]) ** 2) / (2 * WIDTH**2)))
    b = RIDGE * np.exp(-(((x - PEAK_B[0]) ** 2 + (y - PEAK_B[1]) ** 2) / (2 * WIDTH**2)))
    texture = 0.085 * np.sin(1.15 * x) * np.cos(1.45 * y) + 0.05 * np.cos(0.8 * x + 1.3 * y)
    saddle = 0.16 * np.exp(-((x**2) / 11.0 + (y**2) / 2.2))
    return a + b + saddle + texture


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(14.0, 5.6))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    grid_x = np.linspace(-6.6, 6.6, 190)
    grid_y = np.linspace(-3.1, 3.1, 110)
    xx, yy = np.meshgrid(grid_x, grid_y)
    zz = surface(xx, yy)

    ax.plot_surface(
        xx, yy, zz,
        cmap=TERRAIN, vmin=-0.05, vmax=1.15,
        rcount=110, ccount=190,
        linewidth=0.16, edgecolors=(1, 1, 1, 0.22),
        antialiased=True, alpha=0.97, shade=True, zorder=1,
    )

    # ---- particle populations on the two terminal manifolds -----------------
    rng = np.random.default_rng(11)
    count = 26

    def cloud(centre, spread=0.58, limit=1.25):
        """Sample on the basin, rejecting points that would trail off its flank."""
        pts = []
        while len(pts) < count:
            offset = rng.normal(scale=spread, size=2)
            if np.hypot(*offset) <= limit:
                pts.append(centre + offset)
        return np.array(pts)

    start = cloud(PEAK_A)
    end = cloud(PEAK_B)
    # pair them by angle so trajectories fan out rather than crossing chaotically
    order_s = np.argsort(np.arctan2(*(start - PEAK_A).T[::-1]))
    order_e = np.argsort(np.arctan2(*(end - PEAK_B).T[::-1]))
    start, end = start[order_s], end[order_e]

    lift = 0.045
    for pts, colour in ((start, BLUE), (end, TEAL)):
        z = surface(pts[:, 0], pts[:, 1]) + lift
        ax.scatter(pts[:, 0], pts[:, 1], z, s=26, c=colour,
                   edgecolors="white", linewidths=0.6, depthshade=False, zorder=6)

    # ---- bridge trajectories ------------------------------------------------
    steps = np.linspace(0.0, 1.0, 160)
    for index in range(count):
        p0, p1 = start[index], end[index]
        # bow outward in y, and pinch toward the axis mid-path: the geometric
        # interpolation is more concentrated than either endpoint.
        pinch = 1.0 - 0.18 * np.sin(np.pi * steps)
        path_x = (1 - steps) * p0[0] + steps * p1[0]
        path_y = ((1 - steps) * p0[1] + steps * p1[1]) * pinch
        wobble = 0.030 * np.sin(2.4 * np.pi * steps + index) * np.sin(np.pi * steps)
        path_y = path_y + wobble
        path_z = surface(path_x, path_y) + 0.075 + 0.20 * np.sin(np.pi * steps)
        ax.plot(path_x, path_y, path_z, color=INK, lw=1.05, alpha=0.42,
                solid_capstyle="round", zorder=5)

    # a single emphasised trajectory
    p0, p1 = start[count // 2], end[count // 2]
    pinch = 1.0 - 0.18 * np.sin(np.pi * steps)
    hx = (1 - steps) * p0[0] + steps * p1[0]
    hy = ((1 - steps) * p0[1] + steps * p1[1]) * pinch
    hz = surface(hx, hy) + 0.09 + 0.24 * np.sin(np.pi * steps)
    ax.plot(hx, hy, hz, color=INK, lw=2.7, alpha=0.95, solid_capstyle="round", zorder=7)
    ax.quiver(hx[-6], hy[-6], hz[-6],
              hx[-1] - hx[-6], hy[-1] - hy[-6], hz[-1] - hz[-6],
              color=INK, arrow_length_ratio=0.55, lw=2.7, zorder=8)

    # No text on the figure: labels belong in the caption.

    # ---- strip the axes entirely -------------------------------------------
    ax.set_xlim(-6.6, 6.6)
    ax.set_ylim(-3.1, 3.1)
    ax.set_zlim(-0.10, 2.05)
    ax.set_box_aspect((3.05, 1.25, 1.02))
    ax.view_init(elev=24, azim=-62)
    ax.set_axis_off()
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor((1, 1, 1, 0))
        pane.line.set_color((1, 1, 1, 0))

    OUT.mkdir(parents=True, exist_ok=True)
    ax.set_position([-0.06, -0.20, 1.12, 1.44])

    # A 3-D axes with set_axis_off leaves invisible artists, so bbox_inches
    # "tight" cannot find the drawing. Rasterise once, measure the ink, and
    # reuse that box for both outputs so the PDF crops identically.
    dpi = 400
    probe = OUT / "_probe.png"
    fig.savefig(probe, dpi=dpi)
    from PIL import Image, ImageChops
    from matplotlib.transforms import Bbox
    with Image.open(probe) as image:
        rgb = image.convert("RGB")
        box = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
        height = rgb.size[1]
    probe.unlink()
    pad = int(0.012 * dpi)
    left, upper, right, lower = box
    crop = Bbox.from_extents(
        max(left - pad, 0) / dpi,
        (height - min(lower + pad, height)) / dpi,
        min(right + pad, rgb.size[0]) / dpi,
        (height - max(upper - pad, 0)) / dpi,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"fig_sb_teaser.{suffix}", dpi=dpi, bbox_inches=crop)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)}/fig_sb_teaser.pdf (+png)")


if __name__ == "__main__":
    main()
