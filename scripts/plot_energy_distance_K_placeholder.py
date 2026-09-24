"""Endpoint energy distance against IMF round for several particle counts K.

Without --csv the values are illustrative: per-seed curves are simulated, no
measured values are used. With --csv the measured values are plotted. Expected
CSV columns:
    K, round, mean, sd[, floor]
where the optional `floor` is the energy distance between two independent sets
of K networks from the same target (the estimator's noise floor at that K).

Curves are ordered by colour, light to dark for K = 2, 8, 32, 64, 128, 256, with
bands of +-1 s.d. over seeds. Floors are not plotted.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).resolve().parents[1] / "paper/figures/energy_distance_K_placeholder"

# Ordinal one-hue ramp (light -> dark = small -> large K): six steps spaced
# evenly in OKLab lightness between blue 250 and blue 700, validated with
# validate_palette.js --ordinal on a white surface.
COLORS = {2: "#86b6ef", 8: "#599be8", 32: "#307edc", 64: "#2164b7",
          128: "#164c91", 256: "#0d366b"}
INK, INK_2 = "#0b0b0b", "#52514e"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def illustrative_data(n_seeds=20, n_rounds=6, seed=11):
    """Simulate per-seed IMF curves and summarize them over seeds.

    Each K has its own decay shape (a stretched exponential with its own rate
    and exponent), its own plateau and, for some K, a rebound at one round.
    Seeds jitter around that shape, with noise whose size varies from round to
    round and shrinks as K grows, plus a per-round shock shared by all seeds of
    one K.
    """
    rng = np.random.default_rng(seed)
    rounds = np.arange(n_rounds)
    # K: (rate, exponent, plateau, rebound round, rebound size)
    shape = {
        2:   (3.5, 1.0, 0.70, None, 0.00),
        8:   (2.4, 0.7, 0.22, 4, 0.10),
        32:  (1.3, 1.6, 0.105, None, 0.00),
        64:  (0.7, 0.8, 0.080, 3, 0.14),
        128: (1.1, 1.3, 0.066, None, 0.00),
        256: (0.6, 1.0, 0.050, 2, 0.06),
    }
    data = {}
    for K, (tau, gamma, plat, bump_at, bump) in shape.items():
        small = np.sqrt(2.0 / K)                         # 1 at K=2, 1/11 at K=256
        start = (0.42 + 0.9 / K) * rng.lognormal(0, 0.10, n_seeds)
        plateau = plat * rng.lognormal(0, 0.10 + 0.25 * small, n_seeds)
        rate = tau * rng.lognormal(0, 0.25, n_seeds)
        decay = np.exp(-(rounds[None, :] / rate[:, None]) ** gamma)
        curves = plateau[:, None] + (start - plateau)[:, None] * decay
        if bump_at is not None:
            curves[:, bump_at:] *= 1 + bump * np.exp(-(rounds[bump_at:] - bump_at))
        sigma = (0.05 + 0.40 * small) * rng.lognormal(0, 0.35, n_rounds)
        noise = rng.lognormal(0, 1, (n_seeds, n_rounds)) ** sigma[None, :]
        shared = rng.lognormal(0, 0.04 + 0.15 * small, n_rounds)  # per-round shock
        noise[:, 0] = shared[0] = 1.0                    # round 0 is the reference
        curves = curves * noise * shared[None, :]
        data[K] = dict(rounds=rounds, mean=curves.mean(0), sd=curves.std(0),
                       floor=float("nan"))
    return data


def csv_data(path):
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[int(r["K"])].append(r)
    data = {}
    for K, rs in sorted(rows.items()):
        rs.sort(key=lambda r: int(r["round"]))
        data[K] = dict(
            rounds=np.array([int(r["round"]) for r in rs]),
            mean=np.array([float(r["mean"]) for r in rs]),
            sd=np.array([float(r["sd"]) for r in rs]),
            floor=float(np.mean([float(r.get("floor") or "nan") for r in rs])),
        )
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
        "ytick.minor.width": 0.4,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x_max = max(d["rounds"].max() for d in data.values())
    for K, d in data.items():
        c = COLORS.get(K, INK_2)
        lo = np.maximum(d["mean"] - d["sd"], d["mean"] * 0.2)   # keep log axis finite
        ax.fill_between(d["rounds"], lo, d["mean"] + d["sd"], color=c,
                        alpha=0.18, lw=0)
        ax.plot(d["rounds"], d["mean"], color=c, lw=1.3, marker="o", ms=3.2,
                mec="white", mew=0.6, zorder=3, label=f"$K={K}$")

    ax.set_yscale("log")
    ax.set_ylim(0.008, 1.6)
    ax.set_yticks([0.01, 0.1], [r"$10^{-2}$", r"$10^{-1}$"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(-0.2, x_max + 0.2)
    ax.set_xticks(range(0, x_max + 1))
    ax.set_xlabel("IMF round")
    ax.set_ylabel(r"$2\,\mathbb{E}\Vert X-Y\Vert-\mathbb{E}\Vert X-X'\Vert"
                  r"-\mathbb{E}\Vert Y-Y'\Vert$")
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color=GRID, lw=0.5)
    ax.legend(frameon=False, loc="lower left", ncol=3, fontsize=7,
              handlelength=1.4, columnspacing=1.0, borderaxespad=0.3,
              handletextpad=0.4)
    fig.tight_layout(pad=0.3)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
