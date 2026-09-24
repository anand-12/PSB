from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import ParameterBlock


@dataclass
class ParameterDistribution:
    """A local coordinate system for a set of nearby trained networks.

    Parameter tensors are first normalized block-by-block. PCA is then fitted to
    the K normalized networks. The learned score acts in the PCA span. Under
    Gaussian smoothing, the score outside that span is available analytically,
    so new-task gradients may still move in the full parameter space.
    """

    mean: torch.Tensor
    scale: torch.Tensor
    basis: torch.Tensor
    clean_latents: torch.Tensor

    @classmethod
    def fit(
        cls,
        networks: torch.Tensor,
        blocks: list[ParameterBlock],
        *,
        rank: int | None = None,
        relative_scale_floor: float = 0.05,
        absolute_scale_floor: float = 1e-3,
        singular_tolerance: float = 1e-7,
    ) -> "ParameterDistribution":
        if networks.ndim != 2 or networks.shape[0] < 2:
            raise ValueError("networks must have shape [K,P] with K >= 2")
        mean = networks.mean(dim=0)
        centered = networks - mean
        scale = torch.empty_like(mean)
        for block in blocks:
            reference = mean[block.start : block.stop]
            variation = centered[:, block.start : block.stop]
            reference_rms = reference.square().mean().sqrt()
            variation_rms = variation.square().mean().sqrt()
            block_scale = torch.maximum(
                variation_rms,
                torch.maximum(
                    relative_scale_floor * reference_rms,
                    reference_rms.new_tensor(absolute_scale_floor),
                ),
            )
            scale[block.start : block.stop] = block_scale

        normalized = centered / scale
        _, singular_values, vh = torch.linalg.svd(normalized, full_matrices=False)
        maximum_rank = min(networks.shape[0] - 1, vh.shape[0])
        requested_rank = maximum_rank if rank is None else min(rank, maximum_rank)
        threshold = singular_tolerance * singular_values.max().clamp_min(1.0)
        numerical_rank = int((singular_values > threshold).sum().item())
        retained_rank = max(1, min(requested_rank, numerical_rank))
        basis = vh[:retained_rank]
        clean_latents = normalized @ basis.T
        return cls(mean=mean.detach(), scale=scale.detach(), basis=basis.detach(), clean_latents=clean_latents.detach())

    def encode(self, parameters: torch.Tensor) -> torch.Tensor:
        normalized = (parameters - self.mean) / self.scale
        return normalized @ self.basis.T

    def normalize(self, parameters: torch.Tensor) -> torch.Tensor:
        return (parameters - self.mean) / self.scale

    def decode(self, normalized: torch.Tensor) -> torch.Tensor:
        return self.mean + self.scale * normalized

    def parallel_component(self, normalized: torch.Tensor) -> torch.Tensor:
        return (normalized @ self.basis.T) @ self.basis

    def prior_score(
        self,
        normalized: torch.Tensor,
        sigma: torch.Tensor,
        denoiser: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = normalized @ self.basis.T
        denoised_latent = denoiser(latent, sigma)
        sigma_squared = sigma.square().reshape(-1, 1).clamp_min(1e-12)
        latent_score = (denoised_latent - latent) / sigma_squared
        parallel = latent @ self.basis
        orthogonal = normalized - parallel
        full_score = latent_score @ self.basis - orthogonal / sigma_squared
        return full_score, denoised_latent

    def analytic_prior_score(
        self,
        normalized: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact score of the Gaussian-smoothed empirical latent distribution."""
        latent = normalized @ self.basis.T
        sigma_squared = sigma.square().reshape(-1, 1).clamp_min(1e-12)
        squared_distance = (latent[:, None, :] - self.clean_latents[None, :, :]).square().sum(dim=-1)
        weights = torch.softmax(-squared_distance / (2.0 * sigma_squared), dim=1)
        denoised_latent = weights @ self.clean_latents
        latent_score = (denoised_latent - latent) / sigma_squared
        parallel = latent @ self.basis
        orthogonal = normalized - parallel
        full_score = latent_score @ self.basis - orthogonal / sigma_squared
        return full_score, denoised_latent

