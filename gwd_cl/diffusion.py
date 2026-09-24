from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from .curvature import curvature_mobility
from .model import FlatMLP
from .parameter_space import ParameterDistribution


class NoiseEmbedding(nn.Module):
    def __init__(self, dimension: int = 32) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("noise embedding dimension must be even")
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), dimension // 2))
        self.register_buffer("frequencies", frequencies)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        log_sigma = sigma.clamp_min(1e-8).log().unsqueeze(-1)
        angles = log_sigma * self.frequencies.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class WeightDenoiser(nn.Module):
    """Predict clean PCA coordinates from their Gaussian-corrupted values."""

    def __init__(self, latent_dimension: int, hidden_dimension: int = 128, noise_dimension: int = 32) -> None:
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
        conditioning = self.noise_embedding(sigma)
        # Residual prediction makes the zero-noise boundary an easy solution.
        return noisy_latent + self.network(torch.cat([noisy_latent, conditioning], dim=-1))


@dataclass
class ScoreTrainingReport:
    initial_validation_loss: float
    final_validation_loss: float
    final_training_loss: float


@dataclass
class ScoreDistillationReport:
    final_loss: float
    final_denoising_loss: float
    final_posterior_loss: float
    teacher_score_rms: float
    likelihood_score_rms: float


def _sample_sigmas(
    count: int,
    sigma_min: float,
    sigma_max: float,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    uniform = torch.rand(count, device=device, generator=generator)
    return (math.log(sigma_min) + uniform * (math.log(sigma_max) - math.log(sigma_min))).exp()


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
) -> tuple[WeightDenoiser, ScoreTrainingReport]:
    device = clean_latents.device
    generator = torch.Generator(device=device).manual_seed(seed)
    model = WeightDenoiser(clean_latents.shape[1], hidden_dimension).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    ema_state = copy.deepcopy(model.state_dict())

    validation_generator = torch.Generator(device=device).manual_seed(seed + 1_000_003)
    validation_indices = torch.arange(max(64, clean_latents.shape[0]), device=device) % clean_latents.shape[0]
    validation_clean = clean_latents[validation_indices]
    validation_sigma = _sample_sigmas(
        validation_clean.shape[0], sigma_min, sigma_max, device=device, generator=validation_generator
    )
    validation_noise = torch.randn(
        validation_clean.shape, device=device, generator=validation_generator, dtype=validation_clean.dtype
    )
    validation_noisy = validation_clean + validation_sigma[:, None] * validation_noise

    @torch.no_grad()
    def validation_loss() -> float:
        prediction = model(validation_noisy, validation_sigma)
        return float((prediction - validation_clean).square().mean().item())

    initial_loss = validation_loss()
    final_training_loss = float("nan")
    model.train()
    for _ in range(steps):
        indices = torch.randint(
            clean_latents.shape[0], (batch_size,), device=device, generator=generator
        )
        clean = clean_latents[indices]
        sigma = _sample_sigmas(batch_size, sigma_min, sigma_max, device=device, generator=generator)
        noise = torch.randn(clean.shape, device=device, generator=generator, dtype=clean.dtype)
        noisy = clean + sigma[:, None] * noise
        prediction = model(noisy, sigma)
        # The denominator prevents large-magnitude PCA directions dominating.
        coordinate_scale = clean_latents.std(dim=0, unbiased=False).square().clamp_min(1e-4)
        loss = ((prediction - clean).square() / coordinate_scale).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        final_training_loss = float(loss.detach().item())
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
    final_loss = validation_loss()
    return model, ScoreTrainingReport(initial_loss, final_loss, final_training_loss)


def distill_sequential_posterior(
    networks: torch.Tensor,
    *,
    old_space: ParameterDistribution,
    old_denoiser: WeightDenoiser,
    new_space: ParameterDistribution,
    classifier: FlatMLP,
    train_loader: DataLoader,
    steps: int,
    batch_size: int,
    learning_rate: float,
    sigma_min: float,
    sigma_max: float,
    hidden_dimension: int,
    posterior_weight: float,
    likelihood_strength: float,
    maximum_correction_rms: float,
    seed: int,
    ema_decay: float = 0.995,
) -> tuple[WeightDenoiser, ScoreDistillationReport]:
    """Compress the sequential posterior score into one new score network.

    The teacher is the previous posterior score plus the current log-likelihood
    score.  Its score is transformed through the old and new parameter
    coordinate systems by the chain rule.  A denoising-score-matching term on
    the updated networks anchors the student to solutions of the current task.
    Only the student is retained after this function returns.
    """
    if not 0.0 <= posterior_weight <= 1.0:
        raise ValueError("posterior_weight must lie in [0,1]")
    if steps < 1 or batch_size < 1:
        raise ValueError("distillation steps and batch size must be positive")
    device = networks.device
    generator = torch.Generator(device=device).manual_seed(seed)
    student = WeightDenoiser(new_space.clean_latents.shape[1], hidden_dimension).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=1e-5)
    ema_state = copy.deepcopy(student.state_dict())
    batches = infinite_batches(train_loader)
    new_clean_normalized = new_space.normalize(networks)
    coordinate_scale = new_space.clean_latents.std(dim=0, unbiased=False).square().clamp_min(1e-4)

    # A scalar conversion is necessarily approximate because blockwise
    # normalization makes an isotropic new-coordinate corruption anisotropic
    # in old coordinates. RMS matching preserves its expected parameter-space
    # magnitude and is stable across task boundaries.
    old_noise_conversion = (new_space.scale / old_space.scale).square().mean().sqrt()
    final_values = [float("nan")] * 5
    student.train()
    for _ in range(steps):
        indices = torch.randint(networks.shape[0], (batch_size,), device=device, generator=generator)
        clean_u = new_clean_normalized[indices]
        clean_z = new_space.clean_latents[indices]
        sigma = _sample_sigmas(batch_size, sigma_min, sigma_max, device=device, generator=generator)
        noise = torch.randn(clean_u.shape, device=device, generator=generator, dtype=clean_u.dtype)
        noisy_u = clean_u + sigma[:, None] * noise
        noisy_z = noisy_u @ new_space.basis.T

        # Evaluate the carried posterior score at exactly the same parameter
        # vectors, then transport it into the student's coordinates.
        with torch.no_grad():
            theta = new_space.decode(noisy_u)
            old_u = old_space.normalize(theta)
            old_sigma = (sigma * old_noise_conversion).clamp(sigma_min, sigma_max)
            old_score_u, _ = old_space.prior_score(old_u, old_sigma, old_denoiser)
            old_score_theta = old_score_u / old_space.scale
            old_score_new_u = old_score_theta * new_space.scale

        inputs, targets = next(batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        differentiable_u = noisy_u.detach().requires_grad_(True)
        current_loss = classifier.losses(new_space.decode(differentiable_u), inputs, targets)
        likelihood_gradient = torch.autograd.grad(current_loss.sum(), differentiable_u)[0]
        likelihood_rms = likelihood_gradient.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
        normalized_likelihood_score = likelihood_gradient / likelihood_rms

        # Likelihood guidance is least reliable at high noise and becomes exact
        # at the clean boundary. The target is the sequential Bayes score
        # s_t = s_{t-1} + grad log p(D_t|theta).
        ramp = (1.0 - sigma / sigma_max).clamp_min(0.0)
        teacher_score_u = old_score_new_u - likelihood_strength * ramp[:, None] * normalized_likelihood_score
        teacher_score_z = teacher_score_u @ new_space.basis.T
        correction = sigma[:, None].square() * teacher_score_z
        correction_rms = correction.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
        correction = correction * (maximum_correction_rms / correction_rms).clamp(max=1.0)
        posterior_clean_target = (noisy_z + correction).detach()

        prediction = student(noisy_z, sigma)
        denoising_loss = ((prediction - clean_z).square() / coordinate_scale).mean()
        posterior_loss = ((prediction - posterior_clean_target).square() / coordinate_scale).mean()
        loss = (1.0 - posterior_weight) * denoising_loss + posterior_weight * posterior_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
        optimizer.step()
        with torch.no_grad():
            for name, value in student.state_dict().items():
                if value.is_floating_point():
                    ema_state[name].mul_(ema_decay).add_(value, alpha=1.0 - ema_decay)
                else:
                    ema_state[name].copy_(value)
        final_values = [
            float(loss.detach().item()),
            float(denoising_loss.detach().item()),
            float(posterior_loss.detach().item()),
            float(old_score_new_u.square().mean().sqrt().item()),
            float(normalized_likelihood_score.square().mean().sqrt().item()),
        ]

    student.load_state_dict(ema_state)
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    return student, ScoreDistillationReport(*final_values)


def antithetic_noise(
    shape: tuple[int, int], *, device: torch.device, generator: torch.Generator, dtype: torch.dtype
) -> torch.Tensor:
    count, dimension = shape
    pairs = count // 2
    positive = torch.randn((pairs, dimension), device=device, generator=generator, dtype=dtype)
    pieces = [positive, -positive]
    if count % 2:
        pieces.append(torch.randn((1, dimension), device=device, generator=generator, dtype=dtype))
    return torch.cat(pieces, dim=0)


def infinite_batches(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


@dataclass
class ReverseDiffusionReport:
    forward_noise_rms: float
    final_likelihood_loss: float
    mean_likelihood_gradient_rms: float
    mean_prior_score_rms: float
    mean_data_step_rms: float
    mobility_min: float
    mobility_max: float


def likelihood_guided_reverse_diffusion(
    networks: torch.Tensor,
    *,
    space: ParameterDistribution,
    denoiser: WeightDenoiser,
    model: FlatMLP,
    train_loader: DataLoader,
    steps: int,
    sigma_min: float,
    sigma_max: float,
    solution_kernel_std: float,
    guidance_scale: float,
    guidance_ramp_power: float,
    gradient_clip: float,
    guidance_mode: str,
    posterior_learning_rate: float,
    posterior_max_step: float,
    previous_precision: torch.Tensor | None,
    curvature_strength: float,
    curvature_damping: float,
    mobility_minimum: float,
    mobility_maximum: float,
    seed: int,
    analytic_score: bool = False,
) -> tuple[torch.Tensor, ReverseDiffusionReport]:
    """Sample the next sequential posterior with guided VE reverse diffusion.

    For the forward process, u_sigma = u_0 + sqrt(tau^2+sigma^2)*epsilon.
    During reversal,
    the old-task score is combined with the current likelihood. In ``score``
    mode the scores are added directly. In ``proximal`` mode, the default, an
    old-posterior probability-flow step alternates with an adaptive current-data
    step. The latter is a preconditioned forward-backward splitting update.

        score_t(u,sigma) ~= score_{1:t-1}(u,sigma) - c(sigma) grad_u L_t(u).

    The update below is the deterministic probability-flow/DDIM discretization.
    Randomness enters through the explicit forward noising operation.
    """
    if steps < 2:
        raise ValueError("reverse diffusion needs at least two steps")
    if guidance_mode not in {"score", "proximal"}:
        raise ValueError(f"unknown guidance mode: {guidance_mode}")
    device = networks.device
    generator = torch.Generator(device=device).manual_seed(seed)
    clean_normalized = space.normalize(networks)
    forward_noise = antithetic_noise(
        tuple(clean_normalized.shape), device=device, generator=generator, dtype=clean_normalized.dtype
    )
    # A finite-width kernel gives the old distribution full support. Without
    # it, an empirical distribution of K point masses cannot acquire durable
    # probability mass in genuinely new parameter directions.
    effective_sigma_max = math.sqrt(solution_kernel_std**2 + sigma_max**2)
    normalized = clean_normalized + effective_sigma_max * forward_noise
    sigmas = torch.exp(
        torch.linspace(math.log(sigma_max), math.log(sigma_min), steps, device=device)
    )
    batches = infinite_batches(train_loader)
    likelihood_gradient_rms: list[float] = []
    prior_score_rms: list[float] = []
    data_step_rms: list[float] = []
    first_moment = torch.zeros_like(normalized)
    second_moment = torch.zeros_like(normalized)
    beta1, beta2 = 0.9, 0.999
    if previous_precision is None:
        mobility = torch.ones_like(space.scale)
    else:
        mobility = curvature_mobility(
            previous_precision,
            space.scale,
            strength=curvature_strength,
            damping=curvature_damping,
            minimum=mobility_minimum,
            maximum=mobility_maximum,
        )

    for index in range(steps - 1):
        sigma_value = sigmas[index]
        next_sigma = sigmas[index + 1]
        effective_sigma_value = torch.sqrt(sigma_value.square() + solution_kernel_std**2)
        effective_next_sigma = torch.sqrt(next_sigma.square() + solution_kernel_std**2)
        sigma = effective_sigma_value.expand(networks.shape[0])
        with torch.no_grad():
            if analytic_score:
                old_score, _ = space.analytic_prior_score(normalized, sigma)
            else:
                old_score, _ = space.prior_score(normalized, sigma, denoiser)

        progress = 1.0 - float(sigma_value.item() / sigma_max)
        ratio = effective_next_sigma / effective_sigma_value
        ramp = max(progress, 0.0) ** guidance_ramp_power

        if guidance_mode == "proximal":
            # First apply the old-posterior probability-flow operator.
            old_denoised = normalized + effective_sigma_value.square() * old_score
            prior_updated = (old_denoised + ratio * (normalized - old_denoised)).detach()
            likelihood_point = prior_updated
        else:
            likelihood_point = normalized

        inputs, targets = next(batches)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        differentiable_normalized = likelihood_point.detach().requires_grad_(True)
        candidate_parameters = space.decode(differentiable_normalized)
        individual_losses = model.losses(candidate_parameters, inputs, targets)
        likelihood_gradient = torch.autograd.grad(individual_losses.sum(), differentiable_normalized)[0]
        gradient_rms = likelihood_gradient.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-12)

        if guidance_mode == "score":
            normalized_gradient = (likelihood_gradient / gradient_rms).clamp(-gradient_clip, gradient_clip)
            posterior_score = old_score - guidance_scale * ramp * normalized_gradient
            denoised_estimate = normalized + effective_sigma_value.square() * posterior_score
            normalized = (denoised_estimate + ratio * (normalized - denoised_estimate)).detach()
            step_rms = (normalized - likelihood_point).square().mean(dim=1).sqrt()
        else:
            # An Adam-preconditioned likelihood step is the forward/data part of
            # the operator split. Its explicit RMS bound prevents a corrupted
            # high-noise minibatch from overwhelming the posterior denoiser.
            first_moment.mul_(beta1).add_(likelihood_gradient, alpha=1.0 - beta1)
            second_moment.mul_(beta2).addcmul_(
                likelihood_gradient, likelihood_gradient, value=1.0 - beta2
            )
            time_index = index + 1
            corrected_first = first_moment / (1.0 - beta1**time_index)
            corrected_second = second_moment / (1.0 - beta2**time_index)
            data_step = posterior_learning_rate * ramp * corrected_first / (corrected_second.sqrt() + 1e-8)
            data_step = data_step * mobility.unsqueeze(0)
            step_rms = data_step.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
            limiter = (posterior_max_step / step_rms).clamp(max=1.0)
            data_step = data_step * limiter
            normalized = (prior_updated - data_step).detach()
            step_rms = data_step.square().mean(dim=1).sqrt()

        likelihood_gradient_rms.append(float(gradient_rms.mean().detach().item()))
        prior_score_rms.append(float(old_score.square().mean().sqrt().detach().item()))
        data_step_rms.append(float(step_rms.mean().detach().item()))

    # We intentionally retain the finite posterior width rather than collapsing
    # back to a PCA center; this is what preserves newly discovered directions.
    inputs, targets = next(batches)
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    with torch.no_grad():
        result = space.decode(normalized).detach()
        individual_losses = model.losses(result, inputs, targets)

    report = ReverseDiffusionReport(
        forward_noise_rms=float((effective_sigma_max * forward_noise).square().mean().sqrt().item()),
        final_likelihood_loss=float(individual_losses.mean().detach().item()),
        mean_likelihood_gradient_rms=float(sum(likelihood_gradient_rms) / len(likelihood_gradient_rms)),
        mean_prior_score_rms=float(sum(prior_score_rms) / len(prior_score_rms)),
        mean_data_step_rms=float(sum(data_step_rms) / len(data_step_rms)),
        mobility_min=float(mobility.min().item()),
        mobility_max=float(mobility.max().item()),
    )
    return result, report
