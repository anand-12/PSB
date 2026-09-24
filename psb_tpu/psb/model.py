"""A ReLU MLP over a flat parameter vector, written for a single network.

Everything here acts on one network; the particle and run axes are added with
``jax.vmap`` by the caller. The flat layout matches the PyTorch reference
(``bwd_cl/model.py``): per layer, the [out, in] weight then the [out] bias.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class MLPSpec:
    dims: tuple[int, ...]

    @property
    def layers(self) -> int:
        return len(self.dims) - 1

    @property
    def blocks(self) -> list[tuple[int, int, tuple[int, ...]]]:
        out, cursor = [], 0
        for layer in range(self.layers):
            fan_in, fan_out = self.dims[layer], self.dims[layer + 1]
            out.append((cursor, cursor + fan_out * fan_in, (fan_out, fan_in)))
            cursor += fan_out * fan_in
            out.append((cursor, cursor + fan_out, (fan_out,)))
            cursor += fan_out
        return out

    @property
    def num_parameters(self) -> int:
        return self.blocks[-1][1]


def make_spec(input_dim: int, width: int, depth: int, output_dim: int) -> MLPSpec:
    return MLPSpec(tuple([input_dim] + [width] * depth + [output_dim]))


def unflatten(spec: MLPSpec, theta: jnp.ndarray) -> list[jnp.ndarray]:
    return [theta[start:stop].reshape(shape) for start, stop, shape in spec.blocks]


def init_params(spec: MLPSpec, key: jax.Array) -> jnp.ndarray:
    pieces = []
    keys = jax.random.split(key, spec.layers)
    for layer in range(spec.layers):
        fan_in, fan_out = spec.dims[layer], spec.dims[layer + 1]
        bound = math.sqrt(6.0 / fan_in)
        weight = jax.random.uniform(keys[layer], (fan_out, fan_in), minval=-bound, maxval=bound)
        pieces += [weight.reshape(-1), jnp.zeros(fan_out)]
    return jnp.concatenate(pieces)


def block_rms(spec: MLPSpec, theta: jnp.ndarray) -> jnp.ndarray:
    pieces = []
    for start, stop, _ in spec.blocks:
        values = theta[start:stop]
        rms = jnp.maximum(jnp.sqrt(jnp.mean(values ** 2)), 1e-8)
        pieces.append(jnp.full(stop - start, rms))
    return jnp.concatenate(pieces)


def fan_in_vector(spec: MLPSpec) -> jnp.ndarray:
    """Per-coordinate fan-in, used as the prior precision of the Laplace init."""
    pieces = []
    for layer in range(spec.layers):
        fan_in, fan_out = spec.dims[layer], spec.dims[layer + 1]
        pieces += [jnp.full(fan_out * fan_in, float(fan_in)), jnp.full(fan_out, float(fan_in))]
    return jnp.concatenate(pieces)


def logits(spec: MLPSpec, theta: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    params = unflatten(spec, theta)
    hidden = x
    for layer in range(spec.layers):
        hidden = hidden @ params[2 * layer].T + params[2 * layer + 1]
        if layer < spec.layers - 1:
            hidden = jax.nn.relu(hidden)
    return hidden


def loss(spec: MLPSpec, theta: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    out = logits(spec, theta, x)
    picked = jnp.take_along_axis(out, y[:, None], axis=1)[:, 0]
    return jnp.mean(jax.nn.logsumexp(out, axis=1) - picked)


def per_example_squared_grads(spec: MLPSpec, theta: jnp.ndarray, x: jnp.ndarray,
                              y: jnp.ndarray) -> jnp.ndarray:
    """Sum over the batch of squared per-example gradients, shape [P].

    A linear layer's per-example weight gradient is delta outer activation, so
    the sum of squares is (delta^2)^T (a^2): one matmul per layer, and the
    [B, P] per-example gradient tensor is never materialised.
    """
    params = unflatten(spec, theta)
    activations, pre_activations = [], []
    hidden = x
    for layer in range(spec.layers):
        activations.append(hidden)
        pre = hidden @ params[2 * layer].T + params[2 * layer + 1]
        pre_activations.append(pre)
        hidden = jax.nn.relu(pre) if layer < spec.layers - 1 else pre
    delta = jax.nn.softmax(pre_activations[-1], axis=-1) - jax.nn.one_hot(y, spec.dims[-1])
    pieces: list = [None] * (2 * spec.layers)
    for layer in reversed(range(spec.layers)):
        squared = delta ** 2
        pieces[2 * layer] = (squared.T @ activations[layer] ** 2).reshape(-1)
        pieces[2 * layer + 1] = squared.sum(axis=0)
        if layer > 0:
            delta = (delta @ params[2 * layer]) * (pre_activations[layer - 1] > 0)
    return jnp.concatenate(pieces)
