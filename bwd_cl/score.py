from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch
from torch import nn


class NoiseEmbedding(nn.Module):
    def __init__(self, dimension: int = 32) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("noise embedding dimension must be even")
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), dimension // 2))
        self.register_buffer("frequencies", frequencies)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        angles = sigma.clamp_min(1e-8).log().unsqueeze(-1) * self.frequencies.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class WeightDenoiser(nn.Module):
    """Predict clean latent coordinates from their Gaussian-corrupted values."""

    def __init__(self, latent_dimension: int, hidden_dimension: int = 256, noise_dimension: int = 32) -> None:
        super().__init__()
        self.noise_embedding = NoiseEmbedding(noise_dimension)
        self.network = nn.Sequential(
            nn.Linear(latent_dimension + noise_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, latent_dimension),
        )

    def forward(self, noisy_latent: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 0:
            sigma = sigma.expand(noisy_latent.shape[0])
        conditioning = self.noise_embedding(sigma)
        return noisy_latent + self.network(torch.cat([noisy_latent, conditioning], dim=-1))


@dataclass
class ScoreReport:
    initial_validation_loss: float
    final_validation_loss: float


def train_denoiser(
    clean_latents: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    sigma_min: float,
    sigma_max: float,
    hidden_dimension: int,
    seed: int,
    ema_decay: float = 0.995,
) -> tuple[WeightDenoiser, ScoreReport]:
    device = clean_latents.device
    generator = torch.Generator(device=device).manual_seed(seed)
    model = WeightDenoiser(clean_latents.shape[1], hidden_dimension).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    ema_state = copy.deepcopy(model.state_dict())
    coordinate_scale = clean_latents.std(dim=0, unbiased=False).square().clamp_min(1e-4)

    def sample_sigma(count: int, gen: torch.Generator) -> torch.Tensor:
        uniform = torch.rand(count, device=device, generator=gen)
        return (math.log(sigma_min) + uniform * (math.log(sigma_max) - math.log(sigma_min))).exp()

    holdout = torch.Generator(device=device).manual_seed(seed + 1_000_003)
    validation_clean = clean_latents[torch.arange(256, device=device) % clean_latents.shape[0]]
    validation_sigma = sample_sigma(validation_clean.shape[0], holdout)
    validation_noisy = validation_clean + validation_sigma[:, None] * torch.randn(
        validation_clean.shape, device=device, generator=holdout, dtype=clean_latents.dtype
    )

    @torch.no_grad()
    def validation_loss() -> float:
        return float(((model(validation_noisy, validation_sigma) - validation_clean).square()
                      / coordinate_scale).mean().item())

    initial = validation_loss()
    model.train()
    for _ in range(steps):
        index = torch.randint(clean_latents.shape[0], (batch_size,), device=device, generator=generator)
        clean = clean_latents[index]
        sigma = sample_sigma(batch_size, generator)
        noisy = clean + sigma[:, None] * torch.randn(
            clean.shape, device=device, generator=generator, dtype=clean.dtype
        )
        loss = ((model(noisy, sigma) - clean).square() / coordinate_scale).mean()
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
    return model, ScoreReport(initial, validation_loss())
