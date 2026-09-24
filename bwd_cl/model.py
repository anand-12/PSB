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
    """An MLP whose full parameter vector is an explicit tensor.

    A leading dimension represents K networks, so the whole particle population
    runs as one batched forward pass rather than K modules.
    """

    def __init__(self, input_dim: int = 784, width: int = 100, depth: int = 2, output_dim: int = 10) -> None:
        if depth < 1:
            raise ValueError("depth must be at least one")
        dims = [input_dim] + [width] * depth + [output_dim]
        blocks: list[ParameterBlock] = []
        cursor = 0
        for layer in range(len(dims) - 1):
            out_dim, in_dim = dims[layer + 1], dims[layer]
            blocks.append(ParameterBlock(f"layer{layer}.weight", cursor, cursor + out_dim * in_dim, (out_dim, in_dim)))
            cursor += out_dim * in_dim
            blocks.append(ParameterBlock(f"layer{layer}.bias", cursor, cursor + out_dim, (out_dim,)))
            cursor += out_dim
        self.dims = dims
        self.blocks = blocks
        self.layers = len(dims) - 1
        self.num_parameters = cursor

    def initialize(self, *, device: torch.device, generator: torch.Generator) -> torch.Tensor:
        values: list[torch.Tensor] = []
        for layer in range(self.layers):
            in_dim, out_dim = self.dims[layer], self.dims[layer + 1]
            bound = math.sqrt(6.0 / in_dim)
            weight = torch.empty(out_dim, in_dim, device=device)
            weight.uniform_(-bound, bound, generator=generator)
            values.extend([weight.reshape(-1), torch.zeros(out_dim, device=device)])
        return torch.cat(values)

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
                if layer < self.layers - 1:
                    hidden = F.relu(hidden)
            return hidden
        if theta.ndim != 2:
            raise ValueError(f"theta must have shape [P] or [K,P], received {tuple(theta.shape)}")
        hidden = inputs if inputs.ndim == 3 else inputs.unsqueeze(0).expand(theta.shape[0], -1, -1)
        for layer in range(self.layers):
            weight, bias = parameters[2 * layer], parameters[2 * layer + 1]
            hidden = torch.einsum("kbi,koi->kbo", hidden, weight) + bias[:, None, :]
            if layer < self.layers - 1:
                hidden = F.relu(hidden)
        return hidden

    def losses(self, theta: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.logits(theta, inputs)
        if theta.ndim == 1:
            return F.cross_entropy(logits, targets)
        expanded = targets if targets.ndim == 2 else targets.unsqueeze(0).expand(theta.shape[0], -1)
        flat = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), expanded.reshape(-1), reduction="none"
        )
        return flat.reshape(theta.shape[0], -1).mean(dim=1)

    @torch.no_grad()
    def per_example_squared_gradients(
        self, theta: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Sum of per-example squared log-likelihood gradients, shape [K,P].

        Averaging gradients over a minibatch before squaring destroys most of
        the Fisher signal, because independent per-example gradients cancel.
        For a ReLU MLP the per-example gradient of a weight matrix factorises as
        delta outer activation, so the exact per-example squared sum is one
        matrix product per layer and never materialises a [K,B,P] tensor.
        """
        parameters = self._parameters(theta)
        activations: list[torch.Tensor] = []
        pre_activations: list[torch.Tensor] = []
        hidden = inputs if inputs.ndim == 3 else inputs.unsqueeze(0).expand(theta.shape[0], -1, -1)
        for layer in range(self.layers):
            activations.append(hidden)
            weight, bias = parameters[2 * layer], parameters[2 * layer + 1]
            pre = torch.einsum("kbi,koi->kbo", hidden, weight) + bias[:, None, :]
            pre_activations.append(pre)
            hidden = F.relu(pre) if layer < self.layers - 1 else pre

        one_hot = F.one_hot(targets, self.dims[-1]).to(hidden.dtype)
        if one_hot.ndim == 2:
            one_hot = one_hot.unsqueeze(0)
        delta = pre_activations[-1].softmax(dim=-1) - one_hot

        squared = torch.zeros_like(theta)
        for layer in reversed(range(self.layers)):
            block_weight, block_bias = self.blocks[2 * layer], self.blocks[2 * layer + 1]
            delta_squared = delta.square()
            weight_term = torch.einsum("kbo,kbi->koi", delta_squared, activations[layer].square())
            squared[:, block_weight.start : block_weight.stop] = weight_term.reshape(theta.shape[0], -1)
            squared[:, block_bias.start : block_bias.stop] = delta_squared.sum(dim=1)
            if layer > 0:
                propagated = torch.einsum("kbo,koi->kbi", delta, parameters[2 * layer])
                delta = propagated * (pre_activations[layer - 1] > 0).to(propagated.dtype)
        return squared

    @torch.no_grad()
    def ensemble_probabilities(self, theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.logits(theta, inputs)
        if theta.ndim == 1:
            return logits.softmax(dim=-1)
        return logits.softmax(dim=-1).mean(dim=0)

    def block_rms(self, theta: torch.Tensor) -> torch.Tensor:
        """Per-tensor RMS magnitude broadcast back over the flat vector."""
        scale = torch.empty_like(theta)
        for block in self.blocks:
            values = theta[..., block.start : block.stop]
            rms = values.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
            scale[..., block.start : block.stop] = rms
        return scale
