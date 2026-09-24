from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import torch
from torch import nn

from .model import FlatMLP


@dataclass
class LayerTokens:
    """One layer's parameters viewed as one token per output neuron."""

    layer: int
    neurons: int
    dimension: int
    weight_start: int
    weight_stop: int
    bias_start: int
    bias_stop: int
    scale: torch.Tensor  # [dimension], per-coordinate spread across all tokens
    centre: torch.Tensor  # [dimension]


class TokenSpace:
    """Flat parameter vectors as sets of per-neuron tokens.

    A network is one sample in 89,610 dimensions, which is hopeless to model
    from K=256 networks. The same network is also 210 neurons, each a vector of
    at most 785 numbers. Scored that way, K networks supply 210*K training
    tokens and the sample-to-dimension ratio rises by four orders of magnitude.
    Denoising is per-neuron, which is what the reverse process actually needs:
    the task is restoring a corrupted network, not inventing one.
    """

    def __init__(self, model: FlatMLP, networks: torch.Tensor) -> None:
        self.model = model
        self.layers: list[LayerTokens] = []
        for layer in range(model.layers):
            weight_block = model.blocks[2 * layer]
            bias_block = model.blocks[2 * layer + 1]
            neurons, fan_in = weight_block.shape
            tokens = self._gather(networks, weight_block, bias_block, neurons, fan_in)
            flat = tokens.reshape(-1, fan_in + 1)
            centre = flat.mean(dim=0)
            scale = (flat - centre).square().mean(dim=0).sqrt().clamp_min(1e-8)
            self.layers.append(
                LayerTokens(
                    layer=layer,
                    neurons=neurons,
                    dimension=fan_in + 1,
                    weight_start=weight_block.start,
                    weight_stop=weight_block.stop,
                    bias_start=bias_block.start,
                    bias_stop=bias_block.stop,
                    scale=scale.detach(),
                    centre=centre.detach(),
                )
            )

    @staticmethod
    def _gather(networks, weight_block, bias_block, neurons, fan_in) -> torch.Tensor:
        weights = networks[:, weight_block.start : weight_block.stop].reshape(-1, neurons, fan_in)
        biases = networks[:, bias_block.start : bias_block.stop].reshape(-1, neurons, 1)
        return torch.cat([weights, biases], dim=-1)

    def encode(self, networks: torch.Tensor, layer: int) -> torch.Tensor:
        """[K,P] -> [K, neurons, dimension], normalised."""
        spec = self.layers[layer]
        weight_block = self.model.blocks[2 * layer]
        bias_block = self.model.blocks[2 * layer + 1]
        tokens = self._gather(networks, weight_block, bias_block, spec.neurons, spec.dimension - 1)
        return (tokens - spec.centre) / spec.scale

    def encode_raw(self, flat: torch.Tensor, layer: int) -> torch.Tensor:
        """Tokenise a flat *direction* (e.g. a gradient): scale only, no centring.

        Subtracting the token mean is right for a parameter vector and wrong for
        a gradient, which has no meaningful offset.
        """
        spec = self.layers[layer]
        weight_block = self.model.blocks[2 * layer]
        bias_block = self.model.blocks[2 * layer + 1]
        tokens = self._gather(flat, weight_block, bias_block, spec.neurons, spec.dimension - 1)
        return tokens / spec.scale

    def decode_into(self, networks: torch.Tensor, layer: int, tokens: torch.Tensor) -> torch.Tensor:
        """Write normalised tokens back into a copy of the flat vectors."""
        spec = self.layers[layer]
        raw = tokens * spec.scale + spec.centre
        updated = networks.clone()
        updated[:, spec.weight_start : spec.weight_stop] = raw[..., :-1].reshape(networks.shape[0], -1)
        updated[:, spec.bias_start : spec.bias_stop] = raw[..., -1]
        return updated

    def perturb(self, networks: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
        """Add sigma of noise in normalised token units across every layer."""
        perturbed = networks
        for layer in range(len(self.layers)):
            tokens = self.encode(perturbed, layer)
            noise = torch.randn(
                tokens.shape, device=tokens.device, dtype=tokens.dtype, generator=generator
            )
            perturbed = self.decode_into(perturbed, layer, tokens + sigma * noise)
        return perturbed

    def denoise(self, networks, denoisers, sigma: float, weight: float) -> torch.Tensor:
        """Apply each layer's denoiser to its own neuron tokens."""
        updated = networks
        for layer, denoiser in enumerate(denoisers):
            tokens = self.encode(updated, layer)
            level = torch.full((tokens.shape[0],), sigma, device=tokens.device, dtype=tokens.dtype)
            with torch.no_grad():
                predicted = denoiser(tokens, level)
            updated = self.decode_into(updated, layer, tokens + weight * (predicted - tokens))
        return updated

    def training_tokens(self, networks: torch.Tensor, layer: int) -> torch.Tensor:
        return self.encode(networks, layer).reshape(-1, self.layers[layer].dimension)


class TokenDenoiser(nn.Module):
    """Noise-conditional denoiser acting on a single neuron's weight vector."""

    def __init__(self, dimension: int, hidden: int = 512, noise_dimension: int = 32) -> None:
        super().__init__()
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), noise_dimension // 2))
        self.register_buffer("frequencies", frequencies)
        self.network = nn.Sequential(
            nn.Linear(dimension + noise_dimension, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dimension),
        )

    def forward(self, noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        angles = sigma.clamp_min(1e-8).log().unsqueeze(-1) * self.frequencies.unsqueeze(0)
        conditioning = torch.cat([angles.sin(), angles.cos()], dim=-1)
        while conditioning.ndim < noisy.ndim:
            conditioning = conditioning.unsqueeze(1)
        conditioning = conditioning.expand(*noisy.shape[:-1], conditioning.shape[-1])
        return noisy + self.network(torch.cat([noisy, conditioning], dim=-1))


@dataclass
class TokenScoreReport:
    initial_validation_loss: float
    final_validation_loss: float
    tokens: int
    dimension: int


def train_token_denoiser(
    tokens: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    sigma_min: float,
    sigma_max: float,
    hidden: int,
    seed: int,
    ema_decay: float = 0.995,
) -> tuple[TokenDenoiser, TokenScoreReport]:
    device = tokens.device
    generator = torch.Generator(device=device).manual_seed(seed)
    model = TokenDenoiser(tokens.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    ema_state = copy.deepcopy(model.state_dict())

    def sample_sigma(count: int, gen: torch.Generator) -> torch.Tensor:
        uniform = torch.rand(count, device=device, generator=gen)
        return (math.log(sigma_min) + uniform * (math.log(sigma_max) - math.log(sigma_min))).exp()

    holdout = torch.Generator(device=device).manual_seed(seed + 7919)
    count = min(2048, tokens.shape[0])
    validation_clean = tokens[torch.randperm(tokens.shape[0], device=device, generator=holdout)[:count]]
    validation_sigma = sample_sigma(count, holdout)
    validation_noisy = validation_clean + validation_sigma[:, None] * torch.randn(
        validation_clean.shape, device=device, generator=holdout, dtype=tokens.dtype
    )

    @torch.no_grad()
    def validation_loss() -> float:
        return float((model(validation_noisy, validation_sigma) - validation_clean).square().mean().item())

    initial = validation_loss()
    model.train()
    for _ in range(steps):
        index = torch.randint(tokens.shape[0], (batch_size,), device=device, generator=generator)
        clean = tokens[index]
        sigma = sample_sigma(batch_size, generator)
        noisy = clean + sigma[:, None] * torch.randn(
            clean.shape, device=device, generator=generator, dtype=clean.dtype
        )
        loss = (model(noisy, sigma) - clean).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if value.is_floating_point():
                    ema_state[name].mul_(ema_decay).add_(value, alpha=1.0 - ema_decay)
                else:
                    ema_state[name].copy_(value)

    model.load_state_dict(ema_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, TokenScoreReport(initial, validation_loss(), tokens.shape[0], tokens.shape[1])
