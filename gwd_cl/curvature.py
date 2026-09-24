from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .model import FlatMLP


@dataclass
class FisherReport:
    batches: int
    raw_mean: float
    raw_max: float


def estimate_diagonal_fisher(
    model: FlatMLP,
    networks: torch.Tensor,
    loader: DataLoader,
    *,
    maximum_batches: int,
    device: torch.device,
) -> tuple[torch.Tensor, FisherReport]:
    """Estimate one diagonal empirical Fisher from the current task only.

    Gradients are computed for every retained network and averaged over the
    fixed population. The returned statistic is normalized to unit mean so
    each task contributes comparable mass to the online precision.
    """
    parameters = networks.detach().clone().requires_grad_(True)
    fisher = torch.zeros(networks.shape[1], device=device, dtype=networks.dtype)
    batches = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        losses = model.losses(parameters, inputs, targets)
        gradient = torch.autograd.grad(losses.sum(), parameters)[0]
        fisher.add_(gradient.detach().square().mean(dim=0))
        batches += 1
        if batches >= maximum_batches:
            break
    if batches == 0:
        raise ValueError("cannot estimate Fisher from an empty loader")
    fisher.div_(batches)
    raw_mean = float(fisher.mean().item())
    raw_max = float(fisher.max().item())
    fisher.div_(fisher.mean().clamp_min(1e-20))
    return fisher.detach(), FisherReport(batches=batches, raw_mean=raw_mean, raw_max=raw_max)


def curvature_mobility(
    precision: torch.Tensor,
    parameter_scale: torch.Tensor,
    *,
    strength: float,
    damping: float,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """Return the inverse-Fisher mobility in normalized parameter coordinates.

    If theta = mean + scale*u, the quadratic precision in u coordinates is
    diag(scale^2 * precision). Mobility is RMS-normalized so curvature changes
    the direction of a data step without silently changing its global size.
    """
    if strength == 0.0:
        return torch.ones_like(precision)
    normalized_precision = precision * parameter_scale.square()
    normalized_precision = normalized_precision / normalized_precision.mean().clamp_min(1e-20)
    mobility = (normalized_precision + damping).pow(-0.5 * strength)
    mobility = mobility / mobility.square().mean().sqrt().clamp_min(1e-20)
    return mobility.clamp(minimum, maximum)
