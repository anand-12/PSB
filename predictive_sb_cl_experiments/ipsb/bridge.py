from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .model import centered_logits


@dataclass
class PredictiveEmbedding:
    mean: torch.Tensor
    components: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, source: torch.Tensor, target: torch.Tensor, rank: int = 0) -> "PredictiveEmbedding":
        values = torch.cat([source.flatten(1), target.flatten(1)], dim=0)
        mean = values.mean(dim=0)
        centered = values - mean
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        maximum = min(values.shape[0] - 1, vh.shape[0])
        selected = maximum if rank <= 0 else min(rank, maximum)
        components = vh[:selected]
        coordinates = centered @ components.T
        # Do not whiten every empirical PC independently.  With 2K samples and
        # rank 2K-1, exact whitening maps the endpoints to a regular simplex and
        # makes every cross-distance equal, erasing precisely the geometry that
        # Sinkhorn is meant to use.  A single RMS scale is unit-invariant while
        # retaining the relative predictive distances.
        global_scale = coordinates.square().mean().sqrt().clamp_min(1e-5)
        scale = torch.ones_like(coordinates[0]) * global_scale
        return cls(mean, components, scale)

    def encode(self, logits: torch.Tensor) -> torch.Tensor:
        return ((logits.flatten(1) - self.mean) @ self.components.T) / self.scale

    def decode(self, latent: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        flat = (latent * self.scale) @ self.components + self.mean
        return centered_logits(flat.reshape(latent.shape[0], *shape))

    def decode_delta(self, latent: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        """Map a latent displacement to centered-logit space."""
        flat = (latent * self.scale) @ self.components
        return centered_logits(flat.reshape(latent.shape[0], *shape))


@dataclass
class SinkhornResult:
    coupling: torch.Tensor
    cost: torch.Tensor
    epsilon: float
    marginal_error: float
    normalized_entropy: float


def sinkhorn_coupling(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon_ratio: float,
    iterations: int = 300,
) -> SinkhornResult:
    if source.shape[0] != target.shape[0]:
        raise ValueError("the basic experiment requires equal endpoint population sizes")
    count = source.shape[0]
    cost = 0.5 * torch.cdist(source, target).square()
    positive = cost[cost > 1e-12]
    median = positive.median() if positive.numel() else torch.tensor(1.0, device=cost.device)
    epsilon = float((epsilon_ratio * median).clamp_min(1e-6).item())
    log_kernel = -cost / epsilon
    log_mass = -math.log(count)
    log_u = torch.zeros(count, device=cost.device, dtype=cost.dtype)
    log_v = torch.zeros_like(log_u)
    for _ in range(iterations):
        log_u = log_mass - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_mass - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    coupling = (log_u[:, None] + log_kernel + log_v[None, :]).exp()
    coupling = coupling / coupling.sum()
    row_error = (coupling.sum(dim=1) - 1.0 / count).abs().max()
    col_error = (coupling.sum(dim=0) - 1.0 / count).abs().max()
    entropy = -(coupling * coupling.clamp_min(1e-30).log()).sum()
    return SinkhornResult(
        coupling=coupling,
        cost=cost,
        epsilon=epsilon,
        marginal_error=float(torch.maximum(row_error, col_error).item()),
        normalized_entropy=float((entropy / math.log(count * count)).item()),
    )


def optimal_permutation(cost: torch.Tensor) -> torch.Tensor:
    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
    permutation = np.empty(cost.shape[0], dtype=np.int64)
    permutation[rows] = columns
    return torch.as_tensor(permutation, device=cost.device)


def birkhoff_decomposition(coupling: torch.Tensor, tolerance: float = 1e-7):
    """Decompose K times a uniform coupling into weighted permutations."""
    count = coupling.shape[0]
    matrix = (coupling * count).detach().double().cpu().numpy()
    matrix[matrix < tolerance] = 0.0
    components: list[tuple[float, np.ndarray]] = []
    for _ in range(count * count + 1):
        if matrix.max() <= tolerance:
            break
        rows, columns = linear_sum_assignment(-matrix)
        selected = matrix[rows, columns]
        weight = float(selected.min())
        if weight <= tolerance:
            # Numerical Sinkhorn residuals can destroy an exact support matching.
            break
        permutation = np.empty(count, dtype=np.int64)
        permutation[rows] = columns
        components.append((weight, permutation))
        matrix[rows, columns] -= weight
        matrix[matrix < tolerance] = 0.0
    if not components:
        raise RuntimeError("could not decompose the Sinkhorn coupling")
    total = sum(weight for weight, _ in components)
    return [(weight / total, permutation) for weight, permutation in components]


def sample_birkhoff_permutation(
    coupling: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, int]:
    decomposition = birkhoff_decomposition(coupling)
    weights = torch.tensor([item[0] for item in decomposition], device=coupling.device)
    selected = int(torch.multinomial(weights, 1, generator=generator).item())
    permutation = torch.as_tensor(decomposition[selected][1], device=coupling.device)
    return permutation, len(decomposition)


@torch.no_grad()
def brownian_bridge_targets(
    source: torch.Tensor,
    target: torch.Tensor,
    permutation: torch.Tensor,
    embedding: PredictiveEmbedding,
    *,
    stages: int,
    epsilon: float,
    schedule: str,
    schedule_power: float,
    stochastic: bool,
    generator: torch.Generator,
) -> tuple[list[torch.Tensor], list[float]]:
    x0 = embedding.encode(source)
    x1 = embedding.encode(target)[permutation]
    fraction = torch.linspace(0.0, 1.0, stages + 1, device=source.device)[1:]
    if schedule == "linear":
        clock = fraction
    elif schedule == "cosine":
        clock = 0.5 * (1.0 - torch.cos(math.pi * fraction))
    elif schedule == "power":
        clock = fraction.pow(schedule_power)
    else:
        raise ValueError(f"unknown bridge schedule: {schedule}")

    if stochastic:
        times = torch.cat([torch.zeros(1, device=source.device), clock])
        increments = []
        for delta in times.diff():
            noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
            increments.append(delta.sqrt() * noise)
        motion = torch.stack(increments).cumsum(dim=0)
        terminal = motion[-1]
        bridge_noise = motion - clock[:, None, None] * terminal[None, :, :]
    else:
        bridge_noise = torch.zeros(
            (stages, *x0.shape), device=x0.device, dtype=x0.dtype
        )

    outputs = []
    shape = source.shape[1:]
    for index, beta in enumerate(clock):
        latent = (
            (1.0 - beta) * x0
            + beta * x1
            + math.sqrt(epsilon) * bridge_noise[index]
        )
        outputs.append(embedding.decode(latent, shape))
    return outputs, [float(value.item()) for value in clock]


@torch.no_grad()
def drifted_bridge_targets(
    reference_path: list[torch.Tensor],
    target: torch.Tensor,
    permutation: torch.Tensor,
    embedding: PredictiveEmbedding,
    *,
    epsilon: float,
    schedule: str,
    schedule_power: float,
    stochastic: bool,
    generator: torch.Generator,
) -> tuple[list[torch.Tensor], list[float]]:
    """Bridge around a particle-wise predictive reference trajectory.

    Let r_i(t) be the prediction path generated by ordinary minibatch SGD for
    source particle i.  The reference diffusion is

        dZ_t = d r_i(t) + sqrt(epsilon) dW_t.

    Conditional on terminal atom y_j, its bridge mean is
    r_i(t) + beta(t) (y_j - r_i(1)).  Sinkhorn pairing outside this function
    gives the empirical Schrödinger bridge for this drifted Gaussian reference.
    """
    stages = len(reference_path)
    if stages == 0:
        raise ValueError("reference_path must contain at least one stage")
    device = target.device
    fraction = torch.linspace(0.0, 1.0, stages + 1, device=device)[1:]
    if schedule == "linear":
        clock = fraction
    elif schedule == "cosine":
        clock = 0.5 * (1.0 - torch.cos(math.pi * fraction))
    elif schedule == "power":
        clock = fraction.pow(schedule_power)
    else:
        raise ValueError(f"unknown bridge schedule: {schedule}")

    terminal = reference_path[-1]
    assigned_target = target[permutation]
    if stochastic:
        times = torch.cat([torch.zeros(1, device=device), clock])
        increments = []
        latent_shape = embedding.encode(terminal).shape
        for delta in times.diff():
            noise = torch.randn(
                latent_shape,
                device=device,
                dtype=target.dtype,
                generator=generator,
            )
            increments.append(delta.sqrt() * noise)
        motion = torch.stack(increments).cumsum(dim=0)
        bridge_noise = motion - clock[:, None, None] * motion[-1][None, :, :]
    else:
        latent_shape = embedding.encode(terminal).shape
        bridge_noise = torch.zeros(
            (stages, *latent_shape), device=device, dtype=target.dtype
        )

    outputs = []
    shape = target.shape[1:]
    correction = assigned_target - terminal
    for index, beta in enumerate(clock):
        mean = reference_path[index] + beta * correction
        perturbation = math.sqrt(epsilon) * embedding.decode_delta(
            bridge_noise[index], shape
        )
        outputs.append(centered_logits(mean + perturbation))
    return outputs, [float(value.item()) for value in clock]


@torch.no_grad()
def residual_bridge_targets(
    source: torch.Tensor,
    target: torch.Tensor,
    permutation: torch.Tensor,
    embedding: PredictiveEmbedding,
    *,
    stages: int,
    epsilon: float,
    schedule: str,
    schedule_power: float,
    stochastic: bool,
    direct_mean: bool,
    generator: torch.Generator,
) -> tuple[list[torch.Tensor], list[float]]:
    """Transport only the centered ensemble residual distribution.

    The predictive mean is handled deterministically while Brownian motion is
    projected onto the zero-population-mean subspace.  Consequently stochastic
    uncertainty transport cannot perturb the ensemble mean used for
    classification, and the terminal empirical predictive marginal is exact.
    """
    source_mean = source.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    source_residual = source - source_mean
    target_residual = target - target_mean
    residual_path, clock = brownian_bridge_targets(
        source_residual,
        target_residual,
        permutation,
        embedding,
        stages=stages,
        epsilon=epsilon,
        schedule=schedule,
        schedule_power=schedule_power,
        stochastic=stochastic,
        generator=generator,
    )
    outputs = []
    for residual, beta in zip(residual_path, clock):
        # Projection makes this Brownian motion on the residual subspace and
        # prevents finite-particle noise from moving the predictive mean.
        residual = residual - residual.mean(dim=0, keepdim=True)
        if direct_mean:
            mean = target_mean
        else:
            mean = (1.0 - beta) * source_mean + beta * target_mean
        outputs.append(centered_logits(mean + residual))
    return outputs, clock


def functional_posterior_targets(
    source: torch.Tensor,
    current_labels: torch.Tensor,
    old_count: int,
    *,
    prior_std: float,
    likelihood_scale: float,
    steps: int,
    step_size: float,
    temperature: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """ULA samples from likelihood(z_new) times a Gaussian-mixture component.

    One chain starts from each source atom. Shuffling the returned atoms removes
    their arbitrary component labels before the transport problem is solved.
    """
    value = source.detach().clone()
    count = source.shape[0]
    expanded_labels = current_labels.unsqueeze(0).expand(count, -1)
    for _ in range(steps):
        differentiable = value.detach().requires_grad_(True)
        current = differentiable[:, old_count:]
        negative_log_likelihood = F.cross_entropy(
            current.reshape(-1, current.shape[-1]),
            expanded_labels.reshape(-1),
            reduction="none",
        ).reshape(count, -1).sum(dim=1)
        prior = 0.5 * (differentiable - source).square().flatten(1).sum(dim=1) / prior_std**2
        energy = (likelihood_scale * negative_log_likelihood + prior).sum()
        gradient = torch.autograd.grad(energy, differentiable)[0]
        noise = torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)
        value = value - step_size * gradient + math.sqrt(2.0 * step_size * temperature) * noise
        value = centered_logits(value)
    shuffle = torch.randperm(count, device=value.device, generator=generator)
    return value.detach()[shuffle]
