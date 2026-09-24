"""Run configuration, grid expansion and static/traced parameter split.

A run is a flat dict. Keys in STATIC_KEYS change array shapes or control flow
and therefore define a compilation group; keys in TRACED_KEYS are passed as
arrays and vmapped over, so one compilation serves an entire sweep of them.
``schedule`` and ``seed`` are handled on the host per run.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import yaml

METHODS = ("psb", "ewc", "si", "finetune")
SCHEDULES = ("linear", "power", "cosine", "hold", "constant", "thermo")
BENCHMARKS = ("permuted", "rotated", "split")
INITS = ("independent", "laplace")

DEFAULTS: dict = dict(
    # benchmark
    benchmark="permuted", tasks=10, max_angle=180.0, split_mode="domain",
    permutation_seed=0,
    # model and population
    width=100, depth=2, particles=128, init="independent", init_modes=8,
    # method
    method="psb", stages=25, moves_per_stage=100, batch_size=64,
    first_task_steps=3000, fisher_examples=20000, fisher_batch_size=256,
    probe_batch_size=1024,
    # traced hyperparameters
    lam=4.0, gamma=0.8, lr=1e-3, lr_decay=0.0, floor=1e-3, jitter=0.02,
    schedule_param=1.0, laplace_temperature=1.0, si_xi=0.1,
    # host-side per run
    schedule="linear", seed=0,
    # evaluation and bookkeeping
    eval_every=1, eval_batch_size=1000, checkpoint_every=1,
)

STATIC_KEYS = (
    "benchmark", "tasks", "max_angle", "split_mode", "permutation_seed",
    "width", "depth", "particles", "init", "init_modes",
    "method", "stages", "moves_per_stage", "batch_size",
    "first_task_steps", "fisher_examples", "fisher_batch_size", "probe_batch_size",
    "eval_every", "eval_batch_size", "checkpoint_every",
)
TRACED_KEYS = (
    "lam", "gamma", "lr", "lr_decay", "floor", "jitter",
    "schedule_param", "laplace_temperature", "si_xi",
)
HOST_KEYS = ("schedule", "seed")
META_KEYS = ("grid", "block", "tag", "varied")


def validate(run: dict) -> dict:
    unknown = set(run) - set(DEFAULTS) - set(META_KEYS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if run["method"] not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {run['method']}")
    if run["schedule"] not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, got {run['schedule']}")
    if run["benchmark"] not in BENCHMARKS:
        raise ValueError(f"benchmark must be one of {BENCHMARKS}, got {run['benchmark']}")
    if run["init"] not in INITS:
        raise ValueError(f"init must be one of {INITS}, got {run['init']}")
    if run["benchmark"] == "split" and run["tasks"] != 5:
        raise ValueError("split MNIST has exactly 5 tasks; set tasks: 5")
    if run["init"] == "laplace" and run["particles"] % run["init_modes"]:
        raise ValueError("particles must be divisible by init_modes for laplace init")
    if run["schedule"] == "hold" and not 0.0 <= run["schedule_param"] < 1.0:
        raise ValueError("hold schedule needs 0 <= schedule_param < 1")
    if not 0.0 <= run["lr_decay"] <= 1.0:
        raise ValueError("lr_decay must lie in [0, 1] (0 = constant, 1 = cosine to zero)")
    if not 0.0 < run["gamma"] <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    return run


def static_signature(run: dict) -> tuple:
    return tuple((key, run[key]) for key in STATIC_KEYS)


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _seeds(spec) -> list[int]:
    if isinstance(spec, int):
        return list(range(spec))
    if isinstance(spec, dict):
        return list(range(spec.get("start", 0), spec.get("start", 0) + spec["count"]))
    return [int(s) for s in spec]


def load_grid(path: str | Path) -> dict:
    grid = yaml.safe_load(Path(path).read_text())
    if "name" not in grid or "blocks" not in grid:
        raise ValueError(f"{path}: a grid needs 'name' and 'blocks'")
    return grid


def resolve_best(reference: dict, results_root: Path, key: str) -> float:
    """Pick the value of ``key`` with the best mean final accuracy in an earlier grid."""
    grid_dir = results_root / reference["best_from"]
    metric = reference.get("metric", "average_accuracy")
    where = reference.get("where", {})
    scores: dict = {}
    for summary_path in grid_dir.glob("*/summary.json"):
        summary = json.loads(summary_path.read_text())
        if not summary.get("complete"):
            continue
        config = summary["config"]
        if config.get("block") != reference["block"]:
            continue
        if any(config.get(k) != v for k, v in where.items()):
            continue
        scores.setdefault(config[key], []).append(summary["final"][metric])
    if not scores:
        raise RuntimeError(
            f"cannot resolve {key}: no completed runs in {grid_dir} for block "
            f"{reference['block']!r} with {where}. Run that grid first."
        )
    best = max(scores, key=lambda value: sum(scores[value]) / len(scores[value]))
    return best


def expand(grid: dict, results_root: Path) -> list[dict]:
    """Expand a grid into concrete, validated run configs."""
    base = dict(DEFAULTS)
    base.update(grid.get("defaults", {}))
    runs: list[dict] = []
    for block in grid["blocks"]:
        fixed = dict(block.get("set", {}))
        for key, value in list(fixed.items()):
            if isinstance(value, dict) and "best_from" in value:
                fixed[key] = resolve_best(value, results_root, key)
                print(f"[{grid['name']}/{block['name']}] {key} <- {fixed[key]} "
                      f"(best in {value['best_from']}/{value['block']})", flush=True)
        vary = block.get("vary", {})
        zipped = block.get("zip", {})
        zip_rows = [dict(zip(zipped, row)) for row in zip(*zipped.values())] if zipped else [{}]
        vary_rows = [dict(zip(vary, combo)) for combo in itertools.product(*vary.values())] if vary else [{}]
        for zip_row in zip_rows:
            for vary_row in vary_rows:
                varied = {**zip_row, **vary_row}
                tag = block["name"] + "".join(f"_{k}{_fmt(v)}" for k, v in varied.items())
                for seed in _seeds(block.get("seeds", 1)):
                    run = dict(base)
                    run.update(fixed)
                    run.update(varied)
                    run.update(seed=seed, grid=grid["name"], block=block["name"],
                               tag=tag, varied=varied)
                    runs.append(validate(run))
    return runs
