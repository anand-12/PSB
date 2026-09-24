from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ParameterBlock:
    start: int
    stop: int
    shape: tuple[int, ...]


class FlatMLP:
    """A population of MLPs represented by a single [K,P] tensor."""

    def __init__(self, width: int = 100, depth: int = 2, classes: int = 10) -> None:
        dims = [784] + [width] * depth + [classes]
        blocks: list[ParameterBlock] = []
        cursor = 0
        for input_dim, output_dim in zip(dims[:-1], dims[1:]):
            size = input_dim * output_dim
            blocks.append(ParameterBlock(cursor, cursor + size, (output_dim, input_dim)))
            cursor += size
            blocks.append(ParameterBlock(cursor, cursor + output_dim, (output_dim,)))
            cursor += output_dim
        self.dims = dims
        self.blocks = blocks
        self.layers = len(dims) - 1
        self.num_parameters = cursor
        self.classes = classes

    def initialize(
        self, particles: int, *, device: torch.device, generator: torch.Generator, jitter: float
    ) -> torch.Tensor:
        values: list[torch.Tensor] = []
        for layer in range(self.layers):
            input_dim, output_dim = self.dims[layer], self.dims[layer + 1]
            bound = math.sqrt(6.0 / input_dim)
            weight = torch.empty(output_dim, input_dim, device=device)
            weight.uniform_(-bound, bound, generator=generator)
            values.extend([weight.reshape(-1), torch.zeros(output_dim, device=device)])
        centre = torch.cat(values)
        population = centre.unsqueeze(0).repeat(particles, 1)
        if jitter > 0.0:
            noise = torch.randn(
                population.shape, device=device, dtype=population.dtype, generator=generator
            )
            population = population + jitter * noise * self.block_rms(centre).unsqueeze(0)
        return population

    def _parameters(self, theta: torch.Tensor) -> list[torch.Tensor]:
        return [
            theta[..., block.start : block.stop].reshape(theta.shape[:-1] + block.shape)
            for block in self.blocks
        ]

    def logits(self, theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        parameters = self._parameters(theta)
        if theta.ndim == 1:
            hidden = inputs
            for layer in range(self.layers):
                hidden = F.linear(hidden, parameters[2 * layer], parameters[2 * layer + 1])
                if layer + 1 < self.layers:
                    hidden = F.relu(hidden)
            return hidden
        if inputs.ndim == 2:
            hidden = inputs.unsqueeze(0).expand(theta.shape[0], -1, -1)
        elif inputs.ndim == 3:
            hidden = inputs
        else:
            raise ValueError(f"inputs must be [B,D] or [K,B,D], got {tuple(inputs.shape)}")
        for layer in range(self.layers):
            weight, bias = parameters[2 * layer], parameters[2 * layer + 1]
            hidden = torch.einsum("kbi,koi->kbo", hidden, weight) + bias[:, None, :]
            if layer + 1 < self.layers:
                hidden = F.relu(hidden)
        return hidden

    def losses(self, theta: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.logits(theta, inputs)
        if targets.ndim == 1:
            targets = targets.unsqueeze(0).expand(theta.shape[0], -1)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        ).reshape(theta.shape[0], -1).mean(dim=1)

    def block_rms(self, theta: torch.Tensor) -> torch.Tensor:
        scale = torch.empty_like(theta)
        for block in self.blocks:
            value = theta[..., block.start : block.stop]
            rms = value.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
            scale[..., block.start : block.stop] = rms
        return scale

    @torch.no_grad()
    def ensemble_probabilities(self, theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits(theta, inputs).softmax(dim=-1).mean(dim=0)


def centered_logits(logits: torch.Tensor) -> torch.Tensor:
    """Remove the softmax-invariant additive logit direction."""
    return logits - logits.mean(dim=-1, keepdim=True)

