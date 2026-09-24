"""Device functions for one run, vmapped over the run axis by ``make_group``.

A run holds K particles (networks) in ``state["theta"]`` [K, P]. Methods:

* ``psb``      tempered data step (scaled by beta) + exact proximal step toward
               each particle's own anchor with Fisher precision (the paper).
               ``schedule: constant`` gives the beta = 1 variant.
* ``ewc``      online EWC: penalty (lam/2) sum prec (theta - anchor)^2 in the
               loss, through Adam; prec = gamma * prec + F.
* ``si``       Synaptic Intelligence (Zenke et al. 2017): path-integral
               importance, penalty lam * sum Omega (theta - anchor)^2.
* ``finetune`` Adam on the new task only.

Adam moments are reset at every boundary, as in the PyTorch reference.
"""
from __future__ import annotations

from functools import partial
import math

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from .data import MNIST_MEAN, MNIST_STD
from .model import (MLPSpec, block_rms, fan_in_vector, init_params, logits, loss,
                    per_example_squared_grads)

B1, B2, EPS = 0.9, 0.999, 1e-8


def fetch_train(task, idx):
    x = task.train_x[idx].astype(jnp.float32)
    x = jnp.take(x, task.perm, axis=-1)
    return (x / 255.0 - MNIST_MEAN) / MNIST_STD, task.train_y[idx]


def fetch_test(task, start, size):
    x = jax.lax.dynamic_slice_in_dim(task.test_x, start, size, 0).astype(jnp.float32)
    x = jnp.take(x, task.perm, axis=-1)
    y = jax.lax.dynamic_slice_in_dim(task.test_y, start, size, 0)
    return (x / 255.0 - MNIST_MEAN) / MNIST_STD, y


class RunFunctions:
    """Single-run functions closed over the static configuration."""

    def __init__(self, spec: MLPSpec, static: dict) -> None:
        self.spec = spec
        self.s = static
        self.method = static["method"]
        self.K = static["particles"]
        self.P = spec.num_parameters
        self._grad = jax.vmap(jax.grad(partial(loss, spec)))
        self._shared_losses = jax.vmap(partial(loss, spec), in_axes=(0, None, None))
        self._shared_sq = jax.vmap(partial(per_example_squared_grads, spec), in_axes=(0, None, None))
        self._shared_logits = jax.vmap(partial(logits, spec), in_axes=(0, None))

    # -- one optimisation move -------------------------------------------------
    def _carry(self, theta):
        carry = {"theta": theta, "m": jnp.zeros_like(theta), "v": jnp.zeros_like(theta),
                 "step": jnp.zeros((), jnp.float32)}
        if self.method == "si":
            carry["omega"] = jnp.zeros_like(theta)
        return carry

    def _move(self, carry, key, beta, lr, hp, task, ctx, prior: bool):
        theta = carry["theta"]
        K = theta.shape[0]
        idx = jax.random.randint(key, (K, self.s["batch_size"]), 0, task.n_train)
        x, y = fetch_train(task, idx)
        g = self._grad(theta, x, y)
        total = g
        if prior and self.method == "ewc":
            total = g + hp["lam"] * ctx["prec"] * (theta - ctx["anchor"])
        elif prior and self.method == "si":
            total = g + 2.0 * hp["lam"] * ctx["big_omega"] * (theta - ctx["anchor"])
        m = B1 * carry["m"] + (1.0 - B1) * total
        v = B2 * carry["v"] + (1.0 - B2) * total * total
        step = carry["step"] + 1.0
        update = lr * (m / (1.0 - B1 ** step)) / (jnp.sqrt(v / (1.0 - B2 ** step)) + EPS)
        new = theta - (beta * update if self.method == "psb" else update)
        if prior and self.method == "psb":
            w = hp["lam"] * lr * (ctx["prec"] + hp["floor"])
            new = (new + w * ctx["anchor"]) / (1.0 + w)
        out = {"theta": new, "m": m, "v": v, "step": step}
        if self.method == "si":
            out["omega"] = carry["omega"] - g * (new - theta)
        return out

    # -- task 1 ------------------------------------------------------------------
    def first_task(self, key, hp, task):
        spec, s = self.spec, self.s
        laplace = s["init"] == "laplace"
        count = s["init_modes"] if laplace else self.K
        k_init, k_jitter, k_train, k_fisher, k_sample = jax.random.split(key, 5)
        start = init_params(spec, k_init)
        theta0 = start[None] + hp["jitter"] * jax.random.normal(k_jitter, (count, self.P)) \
            * block_rms(spec, start)[None]
        steps = s["first_task_steps"]
        lrs = hp["lr"] * 0.5 * (1.0 + jnp.cos(math.pi * jnp.arange(steps) / max(steps - 1, 1)))

        def body(carry, inputs):
            k, lr = inputs
            return self._move(carry, k, 1.0, lr, hp, task, None, prior=False), None

        carry, _ = jax.lax.scan(body, self._carry(theta0), (jax.random.split(k_train, steps), lrs))
        theta = carry["theta"]
        anchor = theta0
        if laplace:
            raw = self.fisher(theta, k_fisher, task, normalise=False)          # [modes, P]
            precision = task.n_train.astype(jnp.float32) * raw + fan_in_vector(spec)[None]
            reps = self.K // count
            base = jnp.repeat(theta, reps, axis=0)
            scale = hp["laplace_temperature"] / jnp.sqrt(jnp.repeat(precision, reps, axis=0))
            theta = base + scale * jax.random.normal(k_sample, (self.K, self.P))
            anchor = theta
        state = {"theta": theta, "anchor": anchor, "prec": jnp.zeros(self.P)}
        if self.method == "si":
            state["omega"] = carry["omega"] if not laplace else jnp.zeros_like(theta)
            state["big_omega"] = jnp.zeros_like(theta)
        return state

    # -- a task boundary ---------------------------------------------------------
    def boundary(self, state, hp, betas, key, task):
        s = self.s
        ctx = {k: state[k] for k in ("anchor", "prec", "big_omega") if k in state}
        carry = self._carry(state["theta"])
        if self.method == "si":
            carry["omega"] = state["omega"]

        moves = s["stages"] * s["moves_per_stage"]
        progress = jnp.arange(moves, dtype=jnp.float32) / max(moves - 1, 1)
        cosine = 0.5 * (1.0 + jnp.cos(math.pi * progress))
        rates = hp["lr"] * ((1.0 - hp["lr_decay"]) + hp["lr_decay"] * cosine)
        rates = rates.reshape(s["stages"], s["moves_per_stage"])

        def stage(carry, inputs):
            beta, k, stage_rates = inputs
            k_probe, k_moves = jax.random.split(k)
            pidx = jax.random.randint(k_probe, (s["probe_batch_size"],), 0, task.n_train)
            px, py = fetch_train(task, pidx)
            variance = jnp.var(self._shared_losses(carry["theta"], px, py))

            def move(c, inputs):
                km, rate = inputs
                return self._move(c, km, beta, rate, hp, task, ctx, prior=True), None

            carry, _ = jax.lax.scan(move, carry,
                                    (jax.random.split(k_moves, s["moves_per_stage"]), stage_rates))
            return carry, variance

        carry, variance = jax.lax.scan(stage, carry,
                                       (betas, jax.random.split(key, s["stages"]), rates))
        out = dict(state)
        out["theta"] = carry["theta"]
        if self.method == "si":
            out["omega"] = carry["omega"]
        return out, variance

    # -- diagonal Fisher ---------------------------------------------------------
    def fisher(self, theta, key, task, normalise: bool = True):
        s = self.s
        size = s["fisher_batch_size"]
        batches = -(-s["fisher_examples"] // size)
        zero = jnp.zeros(self.P) if normalise else jnp.zeros_like(theta)

        def body(acc, k):
            idx = jax.random.randint(k, (size,), 0, task.n_train)
            x, y = fetch_train(task, idx)
            squared = self._shared_sq(theta, x, y)
            return acc + (squared.mean(axis=0) if normalise else squared), None

        acc, _ = jax.lax.scan(body, zero, jax.random.split(key, batches))
        fisher = acc / (batches * size)
        return fisher / jnp.maximum(fisher.mean(), 1e-30) if normalise else fisher

    # -- after a task ------------------------------------------------------------
    def absorb(self, state, fisher, hp):
        out = dict(state)
        theta = state["theta"]
        if self.method in ("psb", "ewc"):
            out["prec"] = hp["gamma"] * state["prec"] + fisher
        if self.method == "si":
            out["big_omega"] = state["big_omega"] + state["omega"] / (
                (theta - state["anchor"]) ** 2 + hp["si_xi"])
            out["omega"] = jnp.zeros_like(theta)
        out["anchor"] = theta
        return out

    # -- ensemble test accuracy ----------------------------------------------------
    def evaluate(self, theta, task):
        size = self.s["eval_batch_size"]
        chunks = task.test_x.shape[0] // size

        def body(total, i):
            x, y = fetch_test(task, i * size, size)
            probs = jax.nn.softmax(self._shared_logits(theta, x), axis=-1).mean(axis=0)
            valid = (i * size + jnp.arange(size)) < task.n_test
            return total + jnp.sum((jnp.argmax(probs, axis=-1) == y) & valid), None

        total, _ = jax.lax.scan(body, jnp.zeros((), jnp.int32), jnp.arange(chunks))
        return total.astype(jnp.float32) / task.n_test.astype(jnp.float32)


def make_group(spec: MLPSpec, static: dict, mesh) -> dict:
    """Jitted, run-vmapped functions; the run axis is sharded over ``mesh``."""
    f = RunFunctions(spec, static)
    runs = NamedSharding(mesh, PartitionSpec("r"))

    def jit(fn, donate=()):
        return jax.jit(fn, out_shardings=runs, donate_argnums=donate)

    def derive(keys, task_index, purpose):
        return jax.vmap(lambda k: jax.random.fold_in(jax.random.fold_in(k, task_index), purpose))(keys)

    return {
        "first_task": jit(jax.vmap(f.first_task, in_axes=(0, 0, None))),
        "boundary": jit(jax.vmap(f.boundary, in_axes=(0, 0, 0, 0, None)), donate=(0,)),
        "fisher": jit(jax.vmap(f.fisher, in_axes=(0, 0, None))),
        "absorb": jit(jax.vmap(f.absorb, in_axes=(0, 0, 0)), donate=(0,)),
        "evaluate": jit(jax.vmap(f.evaluate, in_axes=(0, None))),
        "derive": jax.jit(derive, out_shardings=runs),
        "functions": f,
    }
