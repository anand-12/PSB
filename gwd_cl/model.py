from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ParameterBlock:
    name: str
    start: int
    stop: int
    shape: tuple[int, ...]


class FlatMLP:
    """An MLP whose complete parameter vector is an explicit tensor.

    Keeping the weights explicit is useful here: diffusion operates directly in
    parameter space, and a leading dimension can represent K networks without
    constructing K Python modules.
    """

    def __init__(
        self,
        input_dim: int = 784,
        width: int = 100,
        depth: int = 2,
        output_dim: int = 10,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be at least one")
        dims = [input_dim] + [width] * depth + [output_dim]
        blocks: list[ParameterBlock] = []
        cursor = 0
        for layer in range(len(dims) - 1):
            out_dim, in_dim = dims[layer + 1], dims[layer]
            count = out_dim * in_dim
            blocks.append(
                ParameterBlock(f"layer{layer}.weight", cursor, cursor + count, (out_dim, in_dim))
            )
            cursor += count
            blocks.append(ParameterBlock(f"layer{layer}.bias", cursor, cursor + out_dim, (out_dim,)))
            cursor += out_dim
        self.dims = dims
        self.blocks = blocks
        self.num_parameters = cursor

    def initialize(self, *, device: torch.device, generator: torch.Generator) -> torch.Tensor:
        values: list[torch.Tensor] = []
        for layer in range(len(self.dims) - 1):
            in_dim = self.dims[layer]
            out_dim = self.dims[layer + 1]
            bound = math.sqrt(6.0 / in_dim)
            weight = torch.empty(out_dim, in_dim, device=device)
            weight.uniform_(-bound, bound, generator=generator)
            bias = torch.zeros(out_dim, device=device)
            values.extend([weight.reshape(-1), bias])
        return torch.cat(values)

    def _parameters(self, theta: torch.Tensor) -> list[torch.Tensor]:
        return [theta[..., block.start : block.stop].reshape(theta.shape[:-1] + block.shape) for block in self.blocks]

    def logits(self, theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Return BxC logits for one network or KxBxC logits for K networks."""
        parameters = self._parameters(theta)
        if theta.ndim == 1:
            hidden = inputs
            for layer in range(len(self.dims) - 1):
                weight, bias = parameters[2 * layer], parameters[2 * layer + 1]
                hidden = F.linear(hidden, weight, bias)
                if layer < len(self.dims) - 2:
                    hidden = F.relu(hidden)
            return hidden
        if theta.ndim != 2:
            raise ValueError(f"theta must have shape [P] or [K,P], received {tuple(theta.shape)}")
        hidden = inputs.unsqueeze(0).expand(theta.shape[0], -1, -1)
        for layer in range(len(self.dims) - 1):
            weight, bias = parameters[2 * layer], parameters[2 * layer + 1]
            hidden = torch.einsum("kbi,koi->kbo", hidden, weight) + bias[:, None, :]
            if layer < len(self.dims) - 2:
                hidden = F.relu(hidden)
        return hidden

    def losses(self, theta: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.logits(theta, inputs)
        if theta.ndim == 1:
            return F.cross_entropy(logits, targets)
        repeated_targets = targets.unsqueeze(0).expand(theta.shape[0], -1)
        flat_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            repeated_targets.reshape(-1),
            reduction="none",
        )
        return flat_loss.reshape(theta.shape[0], -1).mean(dim=1)

    @torch.no_grad()
    def ensemble_probabilities(self, theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.logits(theta, inputs)
        if theta.ndim == 1:
            return logits.softmax(dim=-1)
        return logits.softmax(dim=-1).mean(dim=0)

