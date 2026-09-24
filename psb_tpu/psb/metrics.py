"""Continual-learning metrics from a possibly partial accuracy matrix.

A[i, j] is accuracy on task j after training on task i; NaN where not
evaluated. Rows may be diagonal-only on long streams (eval_every > 1), so every
statistic uses only the entries that exist.
"""
from __future__ import annotations

import numpy as np


def row_metrics(A: np.ndarray, i: int) -> dict:
    row = A[i, : i + 1]
    diag = np.diag(A)[: i + 1]
    out = {
        "acquisition": float(np.nanmean(diag)) if np.isfinite(diag).any() else None,
        "full_eval": bool(np.isfinite(row).all()),
    }
    if not out["full_eval"]:
        return out
    out["average_accuracy"] = float(row.mean())
    forgetting, backward = [], []
    for j in range(i):
        history = A[j:i, j]
        history = history[np.isfinite(history)]
        if history.size:
            forgetting.append(float(history.max() - A[i, j]))
        if np.isfinite(A[j, j]):
            backward.append(float(A[i, j] - A[j, j]))
    out["forgetting"] = float(np.mean(forgetting)) if forgetting else 0.0
    out["backward_transfer"] = float(np.mean(backward)) if backward else 0.0
    out["task_accuracies"] = [float(v) for v in row]
    return out
