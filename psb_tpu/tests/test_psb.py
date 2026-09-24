import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_default_matmul_precision", "highest")

from psb import schedules
from psb.config import DEFAULTS, expand, validate
from psb.metrics import row_metrics
from psb.model import init_params, loss, make_spec, per_example_squared_grads


def test_per_example_fisher_matches_autodiff():
    spec = make_spec(12, 7, 2, 4)
    theta = init_params(spec, jax.random.PRNGKey(0)) + 0.05
    x = jax.random.normal(jax.random.PRNGKey(1), (9, 12))
    y = jax.random.randint(jax.random.PRNGKey(2), (9,), 0, 4)
    manual = per_example_squared_grads(spec, theta, x, y)
    single = jax.vmap(jax.grad(lambda t, xi, yi: loss(spec, t, xi[None], yi[None])), in_axes=(None, 0, 0))
    reference = (single(theta, x, y) ** 2).sum(axis=0)
    np.testing.assert_allclose(manual, reference, rtol=1e-4, atol=1e-7)


def test_proximal_step_is_exact_minimiser():
    theta, anchor, prec = np.array([0.3, -1.0]), np.array([0.0, 2.0]), np.array([5.0, 0.1])
    w = 0.7 * prec
    closed = (theta + w * anchor) / (1 + w)
    # argmin_x 0.5||x - theta||^2 + 0.5 sum w (x - anchor)^2
    grad = (closed - theta) + w * (closed - anchor)
    np.testing.assert_allclose(grad, 0.0, atol=1e-12)


@pytest.mark.parametrize("name,param", [("linear", 1.0), ("power", 2.0), ("cosine", 1.0),
                                        ("hold", 0.2), ("constant", 1.0), ("thermo", 1.0)])
def test_schedules_end_at_one_and_are_monotone(name, param):
    b = schedules.betas(name, param, 25)
    assert b.shape == (25,) and b[-1] == 1.0
    assert np.all(np.diff(b) >= -1e-7) and b[0] > 0


def test_thermodynamic_schedule_follows_variance():
    linear = schedules.betas("linear", 1.0, 10)
    flat = schedules.thermodynamic(linear, np.ones(10))
    np.testing.assert_allclose(flat, linear, atol=1e-6)
    # large variance late in the path -> stages crowd toward beta = 1
    skewed = schedules.thermodynamic(linear, np.linspace(0.01, 100.0, 10))
    assert skewed[4] > linear[4]


def test_metrics_on_known_matrix():
    A = np.full((3, 3), np.nan)
    A[0, 0] = 0.9
    A[1, :2] = [0.8, 0.95]
    A[2, :3] = [0.7, 0.9, 0.97]
    m = row_metrics(A, 2)
    assert m["average_accuracy"] == pytest.approx((0.7 + 0.9 + 0.97) / 3)
    assert m["forgetting"] == pytest.approx(((0.9 - 0.7) + (0.95 - 0.9)) / 2)
    assert m["acquisition"] == pytest.approx((0.9 + 0.95 + 0.97) / 3)
    A[1, 0] = np.nan  # diagonal-only row
    assert not row_metrics(A, 1)["full_eval"]


def test_grid_expansion(tmp_path):
    grid = {"name": "g", "defaults": {"tasks": 3},
            "blocks": [{"name": "a", "vary": {"lam": [1.0, 2.0]}, "seeds": 2},
                       {"name": "b", "zip": {"tasks": [20, 50], "eval_every": [1, 5]}, "seeds": [7]}]}
    runs = expand(grid, tmp_path)
    assert len(runs) == 6
    assert {r["tag"] for r in runs} == {"a_lam1", "a_lam2", "b_tasks20_eval_every1", "b_tasks50_eval_every5"}
    with pytest.raises(ValueError):
        validate({**DEFAULTS, "method": "nope"})


def test_best_from_selects_highest_mean(tmp_path):
    for lam, acc in ((0.1, 0.80), (1.0, 0.85), (10.0, 0.70)):
        out = tmp_path / "tune" / f"e_lam{lam}_seed0"
        out.mkdir(parents=True)
        (out / "summary.json").write_text(json.dumps({
            "complete": True, "config": {"block": "e", "lam": lam, "gamma": 0.8},
            "final": {"average_accuracy": acc}}))
    grid = {"name": "next", "blocks": [{"name": "x", "set": {"lam": {"best_from": "tune", "block": "e"}}}]}
    assert expand(grid, tmp_path)[0]["lam"] == 1.0
