"""Tempering schedules beta_1..beta_M for one boundary (host side, per run).

All schedules end at beta_M = 1. ``thermo`` spaces the stages at equal
thermodynamic length, int sqrt(Var[U]) d(beta), using the across-particle loss
variance recorded at each stage of the previous boundary; the first boundary
has no profile yet and uses the linear schedule.
"""
from __future__ import annotations

import numpy as np


def betas(schedule: str, param: float, stages: int,
          previous_betas: np.ndarray | None = None,
          previous_variance: np.ndarray | None = None) -> np.ndarray:
    grid = np.arange(1, stages + 1, dtype=np.float64) / stages
    if schedule == "linear":
        out = grid
    elif schedule == "power":
        out = grid ** param
    elif schedule == "cosine":
        out = 0.5 * (1.0 - np.cos(np.pi * grid))
    elif schedule == "hold":
        out = np.minimum(1.0, grid / (1.0 - param))
    elif schedule == "constant":
        out = np.ones(stages)
    elif schedule == "thermo":
        if previous_betas is None or previous_variance is None:
            out = grid
        else:
            out = thermodynamic(previous_betas, previous_variance)
    else:
        raise ValueError(f"unknown schedule {schedule}")
    out = np.asarray(out, dtype=np.float64)
    out[-1] = 1.0
    return out.astype(np.float32)


def thermodynamic(previous_betas: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """Equal-length stages under a piecewise-constant speed sqrt(variance)."""
    ends = np.asarray(previous_betas, dtype=np.float64)
    starts = np.concatenate([[0.0], ends[:-1]])
    speed = np.sqrt(np.maximum(np.asarray(variance, dtype=np.float64), 0.0))
    speed = np.maximum(speed, 1e-3 * max(speed.max(), 1e-12))
    widths = np.maximum(ends - starts, 0.0)
    cumulative = np.concatenate([[0.0], np.cumsum(speed * widths)])
    points = np.concatenate([[0.0], ends])
    keep = np.concatenate([[True], np.diff(cumulative) > 0])
    stages = ends.shape[0]
    targets = cumulative[-1] * np.arange(1, stages + 1) / stages
    return np.interp(targets, cumulative[keep], points[keep])
