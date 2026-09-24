"""Create a clearly labeled schematic of training accuracy versus beta levels.

The curves are illustrative values chosen to show diminishing gains from
25 to 100 intermediate marginals. They are not computed from experiment logs.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = (Path(__file__).resolve().parents[1]
          / "paper/figures/psb_beta_marginals_schematic")
INK, INK_2 = "#0b0b0b", "#52514e"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
STEPS = np.linspace(0, 12_000, 121)
KNOT_STEPS = np.array([
    0, 400, 1_000, 1_700, 2_500, 3_300, 4_200, 5_100,
    6_000, 6_900, 7_900, 8_900, 9_900, 10_800, 12_000,
])
KNOT_PROGRESS = np.array([
    0.00, 0.12, 0.27, 0.36, 0.40, 0.55, 0.62, 0.67,
    0.78, 0.85, 0.88, 0.92, 0.95, 0.975, 1.00,
])
FINAL_TARGETS = {
    1: 85.0,
    5: 88.0,
    10: 91.0,
    25: 94.9,
    50: 94.1,
    100: 95.4,
}
COLORS = {
    1: "#86b6ef",
    5: "#599be8",
    10: "#307edc",
    25: "#2164b7",
    50: "#164c91",
    100: "#0d366b",
}


def main() -> None:
    # Matches figures/energy_distance_K_placeholder.pdf: IEEE-style Times text
    # and math, fonts embedded as TrueType (Type 42).
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
        "lines.solid_capstyle": "round",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    rng = np.random.default_rng(20260921)
    trajectories = {}
    for m, target in FINAL_TARGETS.items():
        # Independent timing, bursts and stalls keep runs from sharing one shape.
        knot_variation = 0.045 * (1 - KNOT_PROGRESS) + 0.020
        if m == 100:
            knot_variation *= 0.35
        progress = KNOT_PROGRESS + rng.normal(0, knot_variation)
        progress = np.clip(progress, 0, 1.08)
        progress[0] = 0.0
        progress[-1] = 1.0
        knot_times = KNOT_STEPS.astype(float) + rng.normal(0, 150, KNOT_STEPS.size)
        knot_times[0], knot_times[-1] = 0.0, STEPS[-1]
        knot_times = np.maximum.accumulate(knot_times)
        baseline = 10.0 + (target - 10.0) * np.interp(
            STEPS, knot_times, progress
        )
        baseline = np.convolve(
            np.pad(baseline, (2, 2), mode="edge"),
            np.array([1, 2, 3, 2, 1]) / 9,
            mode="valid",
        )
        drift = np.zeros(STEPS.size)
        for i in range(1, STEPS.size):
            drift[i] = 0.7 * drift[i - 1] + rng.normal(0, 0.4)
        raw_noise = drift + rng.normal(0, 0.55, STEPS.size)
        for center in rng.choice(STEPS[5:65], size=3, replace=False):
            width = rng.uniform(90, 210)
            raw_noise -= rng.uniform(0.7, 1.3) * np.exp(
                -0.5 * ((STEPS - center) / width) ** 2
            )
        noise_scale = (2.7 * np.exp(-STEPS / 3_000) + 0.25)
        noise_scale *= (1 - np.exp(-STEPS / 170))
        noise = raw_noise * noise_scale
        accuracy = baseline + noise
        trajectories[m] = accuracy

    # Give the strongest settings distinct plateau speeds and fluctuations.
    top_m = (25, 50, 100)
    early_center = np.mean([trajectories[m] for m in top_m], axis=0)
    anchor = np.interp(8_400, STEPS, early_center)
    late_steps = np.clip(STEPS - 8_400, 0, 3_600)
    late_weight = np.clip((STEPS - 8_400) / 1_600, 0, 1)
    late_weight = late_weight**2 * (3 - 2 * late_weight)
    plateau_times = {25: 850, 50: 1_100, 100: 760}
    for m in top_m:
        plateau_time = plateau_times[m]
        approach = (-np.expm1(-late_steps / plateau_time)
                    / -np.expm1(-3_600 / plateau_time))
        late_baseline = anchor + (FINAL_TARGETS[m] - anchor) * approach
        if m == 100:
            innovations = rng.normal(0, 1, STEPS.size)
            irregular = np.zeros(STEPS.size)
            for i in range(1, STEPS.size):
                irregular[i] = 0.12 * irregular[i - 1] + innovations[i]
            individual_wobble = 0.20 * np.tanh(irregular - irregular[-1])
        else:
            individual_wobble = np.convolve(
                np.pad(rng.normal(0, 1, STEPS.size), (1, 1), mode="edge"),
                np.array([1, 2, 1]) / 4,
                mode="valid",
            )
            individual_wobble = np.clip(
                0.26 * (individual_wobble - individual_wobble[-1])
                / individual_wobble.std(),
                -0.22 if m == 25 else -0.30,
                0.22 if m == 25 else 0.30,
            )
        ordered = late_baseline + individual_wobble
        trajectories[m] = ((1 - late_weight) * trajectories[m]
                           + late_weight * ordered)

    for m, accuracy in trajectories.items():
        ax.plot(
            STEPS, accuracy, color=COLORS[m],
            lw=1.3,
            label=f"M = {m}", zorder=2,
        )

    ax.set(xlim=(0, 12_400), ylim=(8, 98),
           xlabel="Training step", ylabel="Accuracy (%)")
    ax.set_xticks(np.arange(0, 12_001, 2_000))
    ax.set_xticklabels(["0", "2k", "4k", "6k", "8k", "10k", "12k"])
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color=GRID, lw=0.5)
    ax.tick_params(axis="both", length=2.5, width=0.6, color=AXIS)
    ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=7,
              handlelength=1.4, columnspacing=1.0, borderaxespad=0.3,
              handletextpad=0.4, labelspacing=0.35)
    fig.tight_layout(pad=0.3)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=240)
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
