from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GaussianEnvelope:
    """Diagonal Gaussian summary of the posterior over solutions.

    Carries an anchor point and an accumulated diagonal precision. The anchor is
    either the population mean, shape [P], or one point per particle, shape
    [K,P]. Per-particle anchoring keeps the population from collapsing onto a
    single trajectory, which is what makes the ensemble and the non-Gaussian
    structure worth carrying at all. This is the part of
    the representation that has full support in every parameter direction, so
    it is what lets a new task move mass where the finite particle population
    has none. Its size is O(P) and independent of the number of tasks.
    """

    mean: torch.Tensor
    precision: torch.Tensor

    @classmethod
    def initial(cls, mean: torch.Tensor) -> "GaussianEnvelope":
        precision = torch.zeros(mean.shape[-1], device=mean.device, dtype=mean.dtype)
        return cls(mean=mean.detach().clone(), precision=precision)

    def absorb(
        self, mean: torch.Tensor, fisher: torch.Tensor, *, decay: float, normalize: bool = True
    ) -> "GaussianEnvelope":
        """Fold one task's Fisher into the running precision and recentre.

        Accumulating raw Fisher mass makes the anchor uniformly stiffer at every
        later task, which starves acquisition near the end of a long stream.
        Renormalising to unit mean keeps the *shape* of the accumulated
        curvature -- which is the part that matters -- while leaving the overall
        anchor scale as one stream-wide constant.
        """
        precision = decay * self.precision + fisher
        if normalize:
            precision = precision / precision.mean().clamp_min(1e-30)
        return GaussianEnvelope(mean=mean.detach().clone(), precision=precision.detach())

    @property
    def per_particle(self) -> bool:
        return self.mean.ndim == 2

    @property
    def centre(self) -> torch.Tensor:
        """The single population mean, whatever the anchor's shape."""
        return self.mean.mean(dim=0) if self.mean.ndim == 2 else self.mean

    def proximal_step(self, theta: torch.Tensor, strength: float, floor: float) -> torch.Tensor:
        """Exact proximal operator of the Gaussian log-density.

        Solves argmin_x  0.5*||x-theta||^2/strength + 0.5*(x-mu)^T L (x-mu)
        coordinate-wise, which is unconditionally stable for any strength and,
        unlike a gradient step, preserves the Fisher geometry exactly: stiff
        directions are pulled hard toward the mean while flat ones barely move.
        """
        if strength <= 0.0:
            return theta
        weight = strength * (self.precision + floor)
        return (theta + weight * self.mean) / (1.0 + weight)

    def exploration_scale(self, floor: float) -> torch.Tensor:
        """Posterior standard deviation directions, normalised to unit RMS."""
        scale = (self.precision + floor).rsqrt()
        return scale / scale.square().mean().sqrt().clamp_min(1e-20)

    def sample(self, count: int, temperature: float, floor: float, generator: torch.Generator) -> torch.Tensor:
        centre = self.mean.mean(dim=0) if self.per_particle else self.mean
        noise = torch.randn(
            (count, self.precision.numel()), device=centre.device, dtype=centre.dtype, generator=generator
        )
        return centre.unsqueeze(0) + temperature * noise * self.exploration_scale(floor).unsqueeze(0)

    def stiffness_report(self) -> dict[str, float]:
        precision = self.precision
        return {
            "precision_mean": float(precision.mean().item()),
            "precision_max": float(precision.max().item()),
            "precision_median": float(precision.median().item()),
        }
