"""Run a chunk of same-shaped runs through a whole task stream.

Per task: (task 1) train K networks, or (later tasks) traverse the boundary;
evaluate; estimate the Fisher; absorb. After every ``checkpoint_every`` tasks
the chunk state is written so a preempted VM resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from . import schedules
from .config import TRACED_KEYS
from .core import make_group
from .data import TaskArrays, build_tasks, load_mnist
from .metrics import row_metrics
from .model import make_spec


def _clean(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    return value


def run_dir(results_root: Path, run: dict) -> Path:
    return results_root / run["grid"] / f"{run['tag']}_seed{run['seed']}"


def is_complete(results_root: Path, run: dict) -> bool:
    path = run_dir(results_root, run) / "summary.json"
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text()).get("complete"))
    except json.JSONDecodeError:
        return False


class Engine:
    def __init__(self, results_root: Path, max_runs_per_device: int = 64) -> None:
        self.results_root = results_root
        self.max_runs_per_device = max_runs_per_device
        self.mesh = Mesh(np.array(jax.devices()), ("r",))
        self.devices = self.mesh.size
        self.runs_sharding = NamedSharding(self.mesh, PartitionSpec("r"))
        self.replicated = NamedSharding(self.mesh, PartitionSpec())
        self.mnist = load_mnist()
        self._tasks: dict = {}
        self._groups: dict = {}

    # -- placement -------------------------------------------------------------
    def put_runs(self, array):
        return jax.device_put(np.asarray(array), self.runs_sharding)

    def tasks_for(self, static: dict) -> tuple[list[TaskArrays], int]:
        key = tuple(static[k] for k in ("benchmark", "tasks", "max_angle", "split_mode",
                                        "permutation_seed", "eval_batch_size"))
        if key not in self._tasks:
            host, out_dim = build_tasks(static, self.mnist, static["eval_batch_size"])
            placed: dict[int, jax.Array] = {}

            def put(array):
                # Permuted MNIST shares its image arrays across tasks; keep one copy.
                ident = id(array)
                if ident not in placed:
                    placed[ident] = jax.device_put(np.asarray(array), self.replicated)
                return placed[ident]

            self._tasks[key] = ([TaskArrays(*[put(a) for a in t]) for t in host], out_dim)
        return self._tasks[key]

    def group(self, static: dict, out_dim: int):
        key = tuple(sorted(static.items()))
        if key not in self._groups:
            spec = make_spec(784, static["width"], static["depth"], out_dim)
            self._groups[key] = (spec, make_group(spec, static, self.mesh))
        return self._groups[key]

    # -- sizing --------------------------------------------------------------------
    def runs_per_device(self, static: dict) -> int:
        tasks, out_dim = self.tasks_for(static)
        spec = make_spec(784, static["width"], static["depth"], out_dim)
        K = max(static["particles"], static["init_modes"] if static["init"] == "laplace" else 0)
        P, width = spec.num_parameters, static["width"]
        # theta, anchor, Adam m and v, gradient, update (+ two SI accumulators)
        state = K * P * (6 + (2 if static["method"] == "si" else 0))
        batch = K * static["batch_size"] * (784 * 2 + 4 * width)
        fisher = K * static["fisher_batch_size"] * (3 * width + out_dim) * 3 + K * P
        evaluation = K * static["eval_batch_size"] * (2 * width + out_dim) * 2
        per_run = 4 * (state + max(batch, fisher, evaluation)) * 1.3
        stats = jax.devices()[0].memory_stats() or {}
        limit = stats.get("bytes_limit", 8 * 1024 ** 3)
        data = sum({id(a): a.nbytes for t in tasks for a in t}.values())
        usable = 0.6 * limit - data
        return int(max(1, min(self.max_runs_per_device, usable // per_run)))

    # -- one chunk -------------------------------------------------------------------
    def run_chunk(self, runs: list[dict], static: dict, label: str) -> None:
        real = len(runs)
        total = -(-real // self.devices) * self.devices
        padded = runs + [runs[-1]] * (total - real)
        tasks, out_dim = self.tasks_for(static)
        spec, fns = self.group(static, out_dim)
        T, M = static["tasks"], static["stages"]
        method = static["method"]

        ident = hashlib.sha1(json.dumps(sorted(f"{r['tag']}_seed{r['seed']}" for r in runs)).encode()).hexdigest()[:12]
        state_dir = self.results_root / runs[0]["grid"] / "_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = state_dir / f"{ident}.npz"

        hp = {k: self.put_runs(np.array([r[k] for r in padded], np.float32)) for k in TRACED_KEYS}
        keys = self.put_runs(np.stack([np.asarray(jax.random.PRNGKey(r["seed"])) for r in padded]))
        A = np.full((total, T, T), np.nan)
        histories: list[list[dict]] = [[] for _ in range(real)]
        prev_betas: np.ndarray | None = None
        prev_var: np.ndarray | None = None
        state = None
        start = 0
        wall = 0.0

        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as ck:
                start = int(ck["task"]) + 1
                A = ck["A"]
                wall = float(ck["wall"])
                prev_betas = ck["prev_betas"] if ck["prev_betas"].size else None
                prev_var = ck["prev_var"] if ck["prev_var"].size else None
                state = {"theta": self.put_runs(ck["theta"]), "prec": self.put_runs(ck["prec"])}
                state["anchor"] = self.put_runs(ck["theta"])
                if method == "si":
                    state["omega"] = self.put_runs(ck["omega"])
                    state["big_omega"] = self.put_runs(ck["big_omega"])
            histories = json.loads(checkpoint.with_suffix(".json").read_text())
            print(f"[{label}] resuming at task {start + 1}/{T}", flush=True)

        for i, run in enumerate(runs):
            out = run_dir(self.results_root, run)
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text(json.dumps(_clean(run), indent=2) + "\n")
            with (out / "metrics.jsonl").open("w") as handle:
                for record in histories[i]:
                    handle.write(json.dumps(_clean(record)) + "\n")

        zeros = None
        for t in range(start, T):
            began = time.time()
            task = tasks[t]
            betas = variance = None
            if t == 0:
                state = fns["first_task"](fns["derive"](keys, jnp.int32(0), jnp.int32(1)), hp, task)
            else:
                betas = np.stack([
                    schedules.betas(r["schedule"], r["schedule_param"], M,
                                    None if prev_betas is None else prev_betas[i],
                                    None if prev_var is None else prev_var[i])
                    for i, r in enumerate(padded)])
                state, variance = fns["boundary"](state, hp, self.put_runs(betas),
                                                  fns["derive"](keys, jnp.int32(t), jnp.int32(2)), task)
                variance = np.asarray(variance)
                prev_betas, prev_var = betas, variance

            full = static["eval_every"] == 1 or (t + 1) % static["eval_every"] == 0 or t == T - 1
            for j in (range(t + 1) if full else [t]):
                A[:, t, j] = np.asarray(fns["evaluate"](state["theta"], tasks[j]))

            if method in ("psb", "ewc"):
                fisher = fns["fisher"](state["theta"], fns["derive"](keys, jnp.int32(t), jnp.int32(3)), task)
            else:
                if zeros is None:
                    zeros = self.put_runs(np.zeros((total, spec.num_parameters), np.float32))
                fisher = zeros
            state = fns["absorb"](state, fisher, hp)
            jax.block_until_ready(state["theta"])
            seconds = time.time() - began
            wall += seconds

            for i, run in enumerate(runs):
                record = {"task": t + 1, **row_metrics(A[i], t), "task_seconds": seconds,
                          "elapsed_seconds": wall}
                if betas is not None:
                    record["betas"] = betas[i].tolist()
                    record["stage_variance"] = variance[i].tolist()
                histories[i].append(record)
                with (run_dir(self.results_root, run) / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(_clean(record)) + "\n")

            accs = [h[-1]["average_accuracy"] for h in histories if "average_accuracy" in h[-1]]
            shown = f" mean avg acc {np.mean(accs):.4f}" if accs else ""
            print(f"[{label}] task {t + 1}/{T} runs={real} {seconds:.1f}s{shown}", flush=True)

            last = t == T - 1
            if not last and (t + 1) % static["checkpoint_every"] == 0:
                arrays = {"task": np.int32(t), "A": A, "wall": np.float64(wall),
                          "prev_betas": prev_betas if prev_betas is not None else np.zeros(0),
                          "prev_var": prev_var if prev_var is not None else np.zeros(0),
                          "theta": np.asarray(state["theta"]), "prec": np.asarray(state["prec"])}
                if method == "si":
                    arrays["omega"] = np.asarray(state["omega"])
                    arrays["big_omega"] = np.asarray(state["big_omega"])
                tmp = checkpoint.with_name(checkpoint.stem + ".tmp.npz")
                np.savez(tmp, **arrays)
                checkpoint.with_suffix(".json.tmp").write_text(json.dumps(_clean(histories)))
                os.replace(checkpoint.with_suffix(".json.tmp"), checkpoint.with_suffix(".json"))
                os.replace(tmp, checkpoint)
                # Test hook: simulate a preemption right after a checkpoint.
                if os.environ.get("PSB_FAIL_AFTER_TASK") == str(t + 1):
                    raise SystemExit(f"PSB_FAIL_AFTER_TASK={t + 1}: simulated preemption")

        for i, run in enumerate(runs):
            final = histories[i][-1]
            summary = {
                "config": run,
                "complete": True,
                "final": {k: final.get(k) for k in ("average_accuracy", "forgetting",
                                                     "backward_transfer", "acquisition")},
                "history": histories[i],
                "accuracy_matrix": A[i].tolist(),
                "parameter_count_per_network": spec.num_parameters,
                "wall_seconds": wall,
                "devices": self.devices,
                "platform": jax.devices()[0].platform,
            }
            (run_dir(self.results_root, run) / "summary.json").write_text(
                json.dumps(_clean(summary), indent=1) + "\n")
        for path in (checkpoint, checkpoint.with_suffix(".json")):
            path.unlink(missing_ok=True)
