from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import ParameterBlock


@dataclass
class ParticleLatent:
    """Low-dimensional coordinates for the non-Gaussian part of the posterior.

    The Gaussian envelope already accounts for scale and support in every
    direction. What it cannot represent is the curvature and correlation of the
    set of solutions, and that is what this latent plus its score model carry.
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
        relative_floor: float = 1e-3,
    ) -> "ParticleLatent":
        if networks.ndim != 2 or networks.shape[0] < 2:
            raise ValueError("networks must have shape [K,P] with K >= 2")
        mean = networks.mean(dim=0)
        centered = networks - mean
        scale = torch.empty_like(mean)
        for block in blocks:
            spread = centered[:, block.start : block.stop].square().mean().sqrt()
            reference = mean[block.start : block.stop].square().mean().sqrt()
            scale[block.start : block.stop] = torch.maximum(
                spread, (relative_floor * reference).clamp_min(1e-8)
            )
        normalized = centered / scale
        _, singular, vh = torch.linalg.svd(normalized, full_matrices=False)
        maximum = min(networks.shape[0] - 1, vh.shape[0])
        requested = maximum if rank is None else min(rank, maximum)
        keep = max(1, min(requested, int((singular > 1e-7 * singular.max().clamp_min(1.0)).sum().item())))
        basis = vh[:keep]
        return cls(
            mean=mean.detach(),
            scale=scale.detach(),
            basis=basis.detach(),
            clean_latents=(normalized @ basis.T).detach(),
        )

    @property
    def rank(self) -> int:
        return self.basis.shape[0]

    def encode(self, theta: torch.Tensor) -> torch.Tensor:
        return ((theta - self.mean) / self.scale) @ self.basis.T

    def project(self, delta: torch.Tensor) -> torch.Tensor:
        """Component of a weight-space displacement lying inside the span.

        Forward noise is projected through this before it is applied. Isotropic
        noise in the ambient space puts 1 - R/P of its energy in directions no
        score model fit to K samples has ever seen, and nothing in the reverse
        process can put that information back.
        """
        normalized = delta / self.scale
        return self.scale * ((normalized @ self.basis.T) @ self.basis)

    def span_noise(self, count: int, generator: torch.Tensor) -> torch.Tensor:
        """Full-magnitude Gaussian noise drawn *inside* the span.

        Projecting ambient noise would retain only R/P of its energy, leaving
        nothing to denoise. Drawing R coordinates and mapping them up keeps the
        corruption at full strength while confining it to directions the score
        model can actually represent. The basis is orthonormal in normalised
        coordinates, so latent-unit noise maps to comparable weight-unit noise.
        """
        latent_noise = torch.randn(
            (count, self.basis.shape[0]),
            device=self.basis.device,
            dtype=self.basis.dtype,
            generator=generator,
        )
        return self.scale * (latent_noise @ self.basis)

    def manifold_pull(
        self, theta: torch.Tensor, sigma: torch.Tensor, denoiser: torch.nn.Module
    ) -> torch.Tensor:
        """Displacement toward the learned solution manifold, in weight units.

        Only the component inside the latent span is returned. The orthogonal
        direction is deliberately left alone: shrinking it is what the earlier
        isotropic construction did, and it suppresses exactly the new-task
        structure the likelihood operator is trying to create.
        """
        latent = self.encode(theta)
        with torch.no_grad():
            denoised = denoiser(latent, sigma)
        return self.scale * ((denoised - latent) @ self.basis)
