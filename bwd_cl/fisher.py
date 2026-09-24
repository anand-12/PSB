from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import TaskData
from .model import FlatMLP


@dataclass
class FisherReport:
    examples: int
    raw_mean: float
    raw_max: float
    participation_ratio: float


@torch.no_grad()
def estimate_diagonal_fisher(
    model: FlatMLP,
    networks: torch.Tensor,
    task: TaskData,
    *,
    examples: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, FisherReport]:
    """Diagonal empirical Fisher for the current task, averaged over particles.

    The statistic is rescaled to unit mean so that each task contributes
    comparable mass to the accumulated precision regardless of how sharp its
    loss surface happens to be.
    """
    total = torch.zeros(networks.shape[1], device=networks.device, dtype=networks.dtype)
    seen = 0
    stream = task.stream(batch_size, generator)
    while seen < examples:
        inputs, targets = next(stream)
        total += model.per_example_squared_gradients(networks, inputs, targets).mean(dim=0)
        seen += inputs.shape[0]
    fisher = total / seen
    raw_mean = float(fisher.mean().item())
    raw_max = float(fisher.max().item())
    # A low participation ratio means the estimate is dominated by a few
    # coordinates, which is the regime where an isotropic anchor is worst.
    participation = float(
        (fisher.sum().square() / (fisher.numel() * fisher.square().sum().clamp_min(1e-30))).item()
    )
    fisher = fisher / fisher.mean().clamp_min(1e-30)
    return fisher, FisherReport(seen, raw_mean, raw_max, participation)
