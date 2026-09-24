from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .data import TaskData
from .latent import ParticleLatent
from .model import FlatMLP
from .bridge import apply_drift
from .posterior import GaussianEnvelope
from .score import WeightDenoiser


@dataclass
class TransitionReport:
    final_loss: float
    mean_data_step_rms: float
    mean_anchor_pull_rms: float
    mean_correction_rms: float
    particle_spread: float


def guided_posterior_transition(
    networks: torch.Tensor,
    *,
    model: FlatMLP,
    envelope: GaussianEnvelope,
    latent: ParticleLatent | None,
    denoiser: WeightDenoiser | None,
    task: TaskData,
    steps: int,
    batch_size: int,
    learning_rate: float,
    anchor_strength: float,
    precision_floor: float,
    correction_weight: float,
    sigma_max: float,
    sigma_min: float,
    exploration_noise: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, TransitionReport]:
    """Move the particle population to the posterior including the new task.

    Every step applies three operators in sequence:

      1. a data operator, one Adam step on the current task's loss;
      2. a prior operator, the exact proximal step of the Gaussian envelope,
         which holds stiff directions near the old mean and lets flat ones go;
      3. a correction operator, a pull toward the learned non-Gaussian manifold
         of past solutions, annealed from sigma_max down to sigma_min.

    Splitting the prior out of the gradient matters. Folded into the Adam
    gradient it would be rescaled by Adam's per-coordinate normalisation, which
    is precisely the Fisher weighting the anchor exists to apply.
    """
    if steps < 1:
        raise ValueError("a transition needs at least one step")
    theta = networks.detach().clone()
    first_moment = torch.zeros_like(theta)
    second_moment = torch.zeros_like(theta)
    beta1, beta2 = 0.9, 0.999
    exploration_scale = envelope.exploration_scale(precision_floor)
    stream = task.particle_stream(theta.shape[0], batch_size, generator)

    data_rms: list[float] = []
    anchor_rms: list[float] = []
    correction_rms: list[float] = []
    final_loss = float("nan")

    for index in range(steps):
        progress = index / max(steps - 1, 1)
        rate = learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        sigma = math.exp(math.log(sigma_max) + progress * (math.log(sigma_min) - math.log(sigma_max)))

        inputs, targets = next(stream)
        differentiable = theta.detach().requires_grad_(True)
        losses = model.losses(differentiable, inputs, targets)
        gradient = torch.autograd.grad(losses.sum(), differentiable)[0]
        final_loss = float(losses.mean().detach().item())

        first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        corrected_first = first_moment / (1.0 - beta1 ** (index + 1))
        corrected_second = second_moment / (1.0 - beta2 ** (index + 1))
        data_step = rate * corrected_first / (corrected_second.sqrt() + 1e-8)
        theta = (theta - data_step).detach()

        # Coupling the anchor to the current rate keeps the balance between the
        # two operators fixed as the schedule decays: the equilibrium offset in
        # a direction is 1/(anchor_strength * precision), independent of rate.
        anchored = envelope.proximal_step(theta, anchor_strength * rate, precision_floor)
        anchor_pull = anchored - theta
        theta = anchored

        correction = torch.zeros((), device=theta.device)
        if correction_weight > 0.0 and latent is not None and denoiser is not None:
            pull = latent.manifold_pull(
                theta, torch.full((theta.shape[0],), sigma, device=theta.device), denoiser
            )
            correction = correction_weight * rate * pull
            theta = theta + correction

        if exploration_noise > 0.0:
            noise = torch.randn(theta.shape, device=theta.device, dtype=theta.dtype, generator=generator)
            theta = theta + exploration_noise * rate * exploration_scale * noise

        if index % 25 == 0:
            data_rms.append(float(data_step.square().mean().sqrt().item()))
            anchor_rms.append(float(anchor_pull.square().mean().sqrt().item()))
            correction_rms.append(float(correction.square().mean().sqrt().item()) if correction.ndim else 0.0)

    spread = float((networks.shape[0] and (theta - theta.mean(dim=0)).square().mean().sqrt().item()) or 0.0)
    report = TransitionReport(
        final_loss=final_loss,
        mean_data_step_rms=sum(data_rms) / max(len(data_rms), 1),
        mean_anchor_pull_rms=sum(anchor_rms) / max(len(anchor_rms), 1),
        mean_correction_rms=sum(correction_rms) / max(len(correction_rms), 1),
        particle_spread=spread,
    )
    return theta.detach(), report


def optimize_particles(
    networks: torch.Tensor,
    *,
    model: FlatMLP,
    task: TaskData,
    steps: int,
    batch_size: int,
    learning_rate: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Plain Adam on every particle, each with its own data order."""
    theta = networks.detach().clone()
    first_moment = torch.zeros_like(theta)
    second_moment = torch.zeros_like(theta)
    beta1, beta2 = 0.9, 0.999
    stream = task.particle_stream(theta.shape[0], batch_size, generator)
    for index in range(steps):
        rate = learning_rate * 0.5 * (1.0 + math.cos(math.pi * index / max(steps - 1, 1)))
        inputs, targets = next(stream)
        differentiable = theta.detach().requires_grad_(True)
        losses = model.losses(differentiable, inputs, targets)
        gradient = torch.autograd.grad(losses.sum(), differentiable)[0]
        first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        corrected_first = first_moment / (1.0 - beta1 ** (index + 1))
        corrected_second = second_moment / (1.0 - beta2 ** (index + 1))
        theta = (theta - rate * corrected_first / (corrected_second.sqrt() + 1e-8)).detach()
    return theta


@dataclass
class DiffusionReport:
    final_loss: float
    forward_noise_rms: float
    mean_envelope_shrink_rms: float
    mean_correction_rms: float
    mean_guidance_rms: float
    particle_spread: float
    span_fraction: float


def diffusion_transition(
    networks: torch.Tensor,
    *,
    model: FlatMLP,
    envelope: GaussianEnvelope,
    latent: ParticleLatent | None,
    denoiser: WeightDenoiser | None,
    task: TaskData,
    steps: int,
    batch_size: int,
    sigma_max: float,
    sigma_min: float,
    posterior_width: float,
    guidance_rate: float,
    guidance_ramp: float,
    anchor_strength: float,
    precision_floor: float,
    correction_weight: float,
    latent_sigma_max: float,
    latent_sigma_min: float,
    block_scale: torch.Tensor,
    generator: torch.Generator,
    subspace_noise: bool = True,
) -> tuple[torch.Tensor, DiffusionReport]:
    """Reverse diffusion confined to the learnable subspace, plus a full-space anchor.

    Corruption is projected onto the rank-R span before it is applied. Ambient
    noise puts 1 - R/P of its energy in directions no score fit to K samples can
    represent (here P/K is about 350), and nothing in the reverse process can
    restore it -- which is why unrestricted weight-space diffusion loses the old
    solution outright.

    Three operators run per step. Inside the span: Tweedie denoising under the
    Gaussian envelope plus the learned non-Gaussian correction, advanced by the
    deterministic probability flow. Over the full vector: the proximal anchor,
    which supplies retention in the 1 - R/P directions the diffusion never
    touches. Then likelihood guidance, taken at the denoised estimate.
    """
    if steps < 2:
        raise ValueError("reverse diffusion needs at least two steps")
    theta_clean = networks.detach()
    if latent is not None and subspace_noise:
        forward = sigma_max * latent.span_noise(theta_clean.shape[0], generator)
        span_fraction = 1.0
    else:
        raw = torch.randn(
            theta_clean.shape, device=theta_clean.device, dtype=theta_clean.dtype, generator=generator
        )
        forward = sigma_max * block_scale.unsqueeze(0) * raw
        span_fraction = float(latent.project(forward).square().sum().item()
                              / forward.square().sum().clamp_min(1e-30).item()) if latent is not None else 0.0
    theta = theta_clean + forward

    sigmas = torch.exp(
        torch.linspace(math.log(sigma_max), math.log(sigma_min), steps, device=theta.device)
    )
    first_moment = torch.zeros_like(theta)
    second_moment = torch.zeros_like(theta)
    beta1, beta2 = 0.9, 0.999
    stream = task.particle_stream(theta.shape[0], batch_size, generator)

    shrink_rms: list[float] = []
    correction_rms: list[float] = []
    guidance_rms: list[float] = []
    final_loss = float("nan")

    for index in range(steps - 1):
        sigma = float(sigmas[index].item())
        next_sigma = float(sigmas[index + 1].item())
        progress = index / max(steps - 2, 1)
        rate = guidance_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        latent_sigma = math.exp(
            math.log(latent_sigma_max)
            + progress * (math.log(latent_sigma_min) - math.log(latent_sigma_max))
        )

        # --- prior operator inside the span: Tweedie + learned correction ---
        shrink = (sigma / posterior_width) ** 2
        envelope_shift = envelope.proximal_step(theta, shrink, precision_floor) - theta
        if latent is not None and subspace_noise:
            envelope_shift = latent.project(envelope_shift)
        denoised = theta + envelope_shift

        correction = torch.zeros((), device=theta.device)
        if correction_weight > 0.0 and latent is not None and denoiser is not None:
            correction = correction_weight * latent.manifold_pull(
                denoised, torch.full((theta.shape[0],), latent_sigma, device=theta.device), denoiser
            )
            denoised = denoised + correction

        # --- deterministic probability flow toward the next noise level ---
        ratio = next_sigma / sigma
        theta = (denoised + ratio * (theta - denoised)).detach()

        # --- retention anchor over the full parameter vector ---
        if anchor_strength > 0.0:
            theta = envelope.proximal_step(theta, anchor_strength * rate, precision_floor)

        # --- likelihood guidance, evaluated at the denoised estimate (DPS) ---
        inputs, targets = next(stream)
        differentiable = denoised.detach().requires_grad_(True)
        losses = model.losses(differentiable, inputs, targets)
        gradient = torch.autograd.grad(losses.sum(), differentiable)[0]
        final_loss = float(losses.mean().detach().item())

        ramp = (1.0 - sigma / sigma_max) ** guidance_ramp
        first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        corrected_first = first_moment / (1.0 - beta1 ** (index + 1))
        corrected_second = second_moment / (1.0 - beta2 ** (index + 1))
        guidance = rate * ramp * corrected_first / (corrected_second.sqrt() + 1e-8)
        theta = (theta - guidance * block_scale.unsqueeze(0)).detach()

        if index % 25 == 0:
            shrink_rms.append(float(envelope_shift.square().mean().sqrt().item()))
            correction_rms.append(float(correction.square().mean().sqrt().item()) if correction.ndim else 0.0)
            guidance_rms.append(float((guidance * block_scale).square().mean().sqrt().item()))

    return theta.detach(), DiffusionReport(
        final_loss=final_loss,
        forward_noise_rms=float(forward.square().mean().sqrt().item()),
        mean_envelope_shrink_rms=sum(shrink_rms) / max(len(shrink_rms), 1),
        mean_correction_rms=sum(correction_rms) / max(len(correction_rms), 1),
        mean_guidance_rms=sum(guidance_rms) / max(len(guidance_rms), 1),
        particle_spread=float((theta - theta.mean(dim=0)).square().mean().sqrt().item()),
        span_fraction=span_fraction,
    )


@dataclass
class TokenDiffusionReport:
    final_loss: float
    forward_noise_rms: float
    mean_denoise_rms: float
    mean_guidance_rms: float
    particle_spread: float


def token_diffusion_transition(
    networks: torch.Tensor,
    *,
    model: FlatMLP,
    envelope: GaussianEnvelope,
    token_space,
    denoisers,
    task: TaskData,
    steps: int,
    batch_size: int,
    sigma_max: float,
    sigma_min: float,
    guidance_rate: float,
    guidance_ramp: float,
    anchor_strength: float,
    precision_floor: float,
    correction_weight: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, TokenDiffusionReport]:
    """Reverse diffusion where the score acts on per-neuron weight tokens.

    The forward process corrupts the whole parameter vector, but the score is
    evaluated one neuron at a time by a denoiser trained on 210*K tokens rather
    than K whole networks. That is what makes denoising well posed here: the
    ambient view has K/P ~ 0.003 samples per dimension, the token view ~33.
    """
    theta_clean = networks.detach()
    theta = token_space.perturb(theta_clean, sigma_max, generator)
    forward_rms = float((theta - theta_clean).square().mean().sqrt().item())

    sigmas = torch.exp(
        torch.linspace(math.log(sigma_max), math.log(sigma_min), steps, device=theta.device)
    )
    first_moment = torch.zeros_like(theta)
    second_moment = torch.zeros_like(theta)
    beta1, beta2 = 0.9, 0.999
    stream = task.particle_stream(theta.shape[0], batch_size, generator)
    block_scale = model.block_rms(envelope.centre)
    denoise_rms: list[float] = []
    guidance_rms: list[float] = []
    final_loss = float("nan")

    for index in range(steps - 1):
        sigma = float(sigmas[index].item())
        next_sigma = float(sigmas[index + 1].item())
        progress = index / max(steps - 2, 1)
        rate = guidance_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

        if correction_weight > 0.0 and denoisers is not None:
            denoised = token_space.denoise(theta, denoisers, sigma, correction_weight)
        else:
            denoised = theta
        shift = denoised - theta

        ratio = next_sigma / sigma
        theta = (denoised + ratio * (theta - denoised)).detach()

        if anchor_strength > 0.0:
            theta = envelope.proximal_step(theta, anchor_strength * rate, precision_floor)

        inputs, targets = next(stream)
        differentiable = denoised.detach().requires_grad_(True)
        losses = model.losses(differentiable, inputs, targets)
        gradient = torch.autograd.grad(losses.sum(), differentiable)[0]
        final_loss = float(losses.mean().detach().item())

        ramp = (1.0 - sigma / sigma_max) ** guidance_ramp
        first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        corrected_first = first_moment / (1.0 - beta1 ** (index + 1))
        corrected_second = second_moment / (1.0 - beta2 ** (index + 1))
        guidance = rate * ramp * corrected_first / (corrected_second.sqrt() + 1e-8)
        theta = (theta - guidance * block_scale.unsqueeze(0)).detach()

        if index % 25 == 0:
            denoise_rms.append(float(shift.square().mean().sqrt().item()))
            guidance_rms.append(float((guidance * block_scale).square().mean().sqrt().item()))

    return theta.detach(), TokenDiffusionReport(
        final_loss=final_loss,
        forward_noise_rms=forward_rms,
        mean_denoise_rms=sum(denoise_rms) / max(len(denoise_rms), 1),
        mean_guidance_rms=sum(guidance_rms) / max(len(guidance_rms), 1),
        particle_spread=float((theta - theta.mean(dim=0)).square().mean().sqrt().item()),
    )


@dataclass
class MergeReport:
    fresh_loss: float
    merge_displacement_rms: float
    fresh_displacement_rms: float
    particle_spread: float


def merge_transition(
    networks: torch.Tensor,
    *,
    model: FlatMLP,
    envelope: GaussianEnvelope,
    task: TaskData,
    steps: int,
    batch_size: int,
    learning_rate: float,
    precision_floor: float,
    new_precision: torch.Tensor,
    fresh: torch.Tensor,
) -> tuple[torch.Tensor, MergeReport]:
    """Product-of-experts composition of the running posterior with a new one.

    Since D_1..t and D_{t+1} are conditionally independent given theta,

        p(theta | D_1..t+1)  proportional to  p(theta | D_1..t) p(theta | D_{t+1}) / prior

    so for Gaussian experts the composition is a precision-weighted mean. The
    new task is fitted with no constraint at all, which is the point: the
    stability/plasticity tradeoff that limits anchored training does not apply,
    because retention is restored by the merge rather than defended during it.
    """
    prior_precision = envelope.precision + precision_floor
    task_precision = new_precision + precision_floor
    prior_mean = envelope.mean
    merged = (prior_precision * prior_mean + task_precision * fresh) / (prior_precision + task_precision)
    return merged.detach(), MergeReport(
        fresh_loss=float("nan"),
        merge_displacement_rms=float((merged - fresh).square().mean().sqrt().item()),
        fresh_displacement_rms=float((fresh - networks).square().mean().sqrt().item()),
        particle_spread=float((merged - merged.mean(dim=0)).square().mean().sqrt().item()),
    )


@dataclass
class TransportReport:
    final_loss: float
    resamples: int
    mean_ess: float
    min_ess: float
    unique_particles: int
    mean_data_step_rms: float
    mean_anchor_pull_rms: float
    particle_spread: float


def _systematic_resample(log_weights: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    weights = torch.softmax(log_weights, dim=0)
    count = weights.shape[0]
    offset = torch.rand(1, device=weights.device, generator=generator)
    positions = (offset + torch.arange(count, device=weights.device, dtype=weights.dtype)) / count
    return torch.searchsorted(weights.cumsum(0), positions).clamp(max=count - 1)


def annealed_transport_transition(
    networks: torch.Tensor,
    *,
    model: FlatMLP,
    envelope: GaussianEnvelope,
    task: TaskData,
    anneal_steps: int,
    move_steps: int,
    batch_size: int,
    weight_batch_size: int,
    learning_rate: float,
    likelihood_scale: float,
    anchor_strength: float,
    precision_floor: float,
    noise_scale: float,
    resample_threshold: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, TransportReport]:
    """Transport the particle population along pi_beta ~ L^beta q_t, beta 0 -> 1.

    The path is built by tempering the likelihood rather than by corrupting the
    parameters. Weight-space noise at this scale is unrecoverable (P/K is about
    350 here), but an annealed likelihood needs no restoration at all: every
    intermediate distribution is a valid posterior over the same uncorrupted
    weights.

    Each stage reweights particles by their incremental log-likelihood and
    resamples when the effective sample size collapses, then moves them with a
    kernel targeting pi_beta. The reweighting is the capability no anchored
    optimiser has: it can kill a particle that the new task rules out, rather
    than dragging it somewhere it does not belong.
    """
    theta = networks.detach().clone()
    count = theta.shape[0]
    log_weights = torch.zeros(count, device=theta.device, dtype=theta.dtype)
    first_moment = torch.zeros_like(theta)
    second_moment = torch.zeros_like(theta)
    beta1, beta2 = 0.9, 0.999
    exploration = envelope.exploration_scale(precision_floor)
    stream = task.particle_stream(count, batch_size, generator)
    weight_stream = task.stream(weight_batch_size, generator)

    resamples = 0
    ess_history: list[float] = []
    data_rms: list[float] = []
    anchor_rms: list[float] = []
    final_loss = float("nan")
    step_index = 0

    for stage in range(anneal_steps):
        beta_previous = stage / anneal_steps
        beta = (stage + 1) / anneal_steps
        delta = beta - beta_previous

        # --- reweight on a batch shared by every particle ---
        weight_inputs, weight_targets = next(weight_stream)
        with torch.no_grad():
            stage_loss = model.losses(theta, weight_inputs, weight_targets)
        log_weights = log_weights - delta * likelihood_scale * stage_loss
        log_weights = log_weights - log_weights.max()
        normalised = torch.softmax(log_weights, dim=0)
        ess = float((1.0 / normalised.square().sum()).item())
        ess_history.append(ess)

        if resample_threshold > 0.0 and ess < resample_threshold * count:
            index = _systematic_resample(log_weights, generator)
            theta = theta[index].contiguous()
            first_moment = first_moment[index].contiguous()
            second_moment = second_moment[index].contiguous()
            log_weights = torch.zeros_like(log_weights)
            resamples += 1

        # --- move under pi_beta ---
        for _ in range(move_steps):
            step_index += 1
            inputs, targets = next(stream)
            differentiable = theta.detach().requires_grad_(True)
            losses = model.losses(differentiable, inputs, targets)
            gradient = torch.autograd.grad(losses.sum(), differentiable)[0]
            final_loss = float(losses.mean().detach().item())

            first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            corrected_first = first_moment / (1.0 - beta1 ** step_index)
            corrected_second = second_moment / (1.0 - beta2 ** step_index)
            data_step = beta * learning_rate * corrected_first / (corrected_second.sqrt() + 1e-8)
            theta = (theta - data_step).detach()

            anchored = envelope.proximal_step(theta, anchor_strength * learning_rate, precision_floor)
            anchor_pull = anchored - theta
            theta = anchored

            if noise_scale > 0.0:
                noise = torch.randn(
                    theta.shape, device=theta.device, dtype=theta.dtype, generator=generator
                )
                theta = theta + noise_scale * learning_rate * exploration * noise

            if step_index % 25 == 0:
                data_rms.append(float(data_step.square().mean().sqrt().item()))
                anchor_rms.append(float(anchor_pull.square().mean().sqrt().item()))

    unique = int(torch.unique(theta[:, :64], dim=0).shape[0])
    return theta.detach(), TransportReport(
        final_loss=final_loss,
        resamples=resamples,
        mean_ess=sum(ess_history) / max(len(ess_history), 1),
        min_ess=min(ess_history) if ess_history else float("nan"),
        unique_particles=unique,
        mean_data_step_rms=sum(data_rms) / max(len(data_rms), 1),
        mean_anchor_pull_rms=sum(anchor_rms) / max(len(anchor_rms), 1),
        particle_spread=float((theta - theta.mean(dim=0)).square().mean().sqrt().item()),
    )


@dataclass
class BridgeReport:
    final_loss: float
    drift_step_rms: float
    kernel_step_rms: float
    anchor_pull_rms: float
    particle_spread: float
    recorded_tokens: int
    used_drift: bool


def bridge_transition(
    networks: torch.Tensor,
    *,
    model: FlatMLP,
    envelope: GaussianEnvelope,
    token_space,
    drifts,
    buffer,
    task: TaskData,
    stages: int,
    move_steps: int,
    batch_size: int,
    learning_rate: float,
    anchor_strength: float,
    precision_floor: float,
    drift_scale: float,
    record_particles: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, BridgeReport]:
    """Traverse the posterior bridge, optionally with a learned drift.

    Each stage advances beta and moves the population under pi_beta. When a
    trained drift is supplied it takes one large learned step per stage, which
    is meant to reproduce what many heuristic steps would have achieved; the
    remaining kernel steps then correct it. When a buffer is supplied the stage
    displacement produced by the reference kernel is recorded as the flow
    matching target for fitting that drift.
    """
    theta = networks.detach().clone()
    count = theta.shape[0]
    first_moment = torch.zeros_like(theta)
    second_moment = torch.zeros_like(theta)
    beta1, beta2 = 0.9, 0.999
    stream = task.particle_stream(count, batch_size, generator)
    probe_stream = task.stream(1024, generator)

    drift_rms: list[float] = []
    kernel_rms: list[float] = []
    anchor_rms: list[float] = []
    final_loss = float("nan")
    step_index = 0
    recorded = 0
    keep = min(record_particles, count)

    for stage in range(stages):
        beta = (stage + 1) / stages
        start = theta.detach().clone()

        # Gradient at the stage start, on one batch shared by all particles, so
        # the drift's conditioning signal is consistent with what was recorded.
        probe_inputs, probe_targets = next(probe_stream)
        probed = theta.detach().requires_grad_(True)
        probe_losses = model.losses(probed, probe_inputs, probe_targets)
        stage_gradient = torch.autograd.grad(probe_losses.sum(), probed)[0].detach()

        # The prior force is as much a part of the stage displacement as the
        # data force, so the drift is conditioned on both.
        stage_anchor = (
            envelope.proximal_step(theta, anchor_strength * learning_rate, precision_floor) - theta
        ).detach()

        if drifts is not None:
            before = theta
            theta = apply_drift(
                theta,
                token_space=token_space,
                drifts=drifts,
                gradient=stage_gradient,
                anchor=stage_anchor,
                beta=beta,
                scale=drift_scale,
            )
            drift_rms.append(float((theta - before).square().mean().sqrt().item()))

        kernel_start = theta.detach().clone()
        for _ in range(move_steps):
            step_index += 1
            inputs, targets = next(stream)
            differentiable = theta.detach().requires_grad_(True)
            losses = model.losses(differentiable, inputs, targets)
            gradient = torch.autograd.grad(losses.sum(), differentiable)[0]
            final_loss = float(losses.mean().detach().item())

            first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            corrected_first = first_moment / (1.0 - beta1 ** step_index)
            corrected_second = second_moment / (1.0 - beta2 ** step_index)
            data_step = beta * learning_rate * corrected_first / (corrected_second.sqrt() + 1e-8)
            theta = (theta - data_step).detach()

            anchored = envelope.proximal_step(theta, anchor_strength * learning_rate, precision_floor)
            anchor_rms.append(float((anchored - theta).square().mean().sqrt().item()))
            theta = anchored

        kernel_rms.append(float((theta - kernel_start).square().mean().sqrt().item()))

        if buffer is not None:
            for layer in range(model.layers):
                before_tokens = token_space.encode(start[:keep], layer)
                after_tokens = token_space.encode(theta[:keep], layer)
                gradient_tokens = token_space.encode_raw(stage_gradient[:keep], layer)
                anchor_tokens = token_space.encode_raw(stage_anchor[:keep], layer)
                buffer.add(
                    layer, before_tokens, gradient_tokens, anchor_tokens, beta,
                    after_tokens - before_tokens,
                )
                recorded += before_tokens.shape[0] * before_tokens.shape[1]

    return theta.detach(), BridgeReport(
        final_loss=final_loss,
        drift_step_rms=sum(drift_rms) / max(len(drift_rms), 1),
        kernel_step_rms=sum(kernel_rms) / max(len(kernel_rms), 1),
        anchor_pull_rms=sum(anchor_rms) / max(len(anchor_rms), 1),
        particle_spread=float((theta - theta.mean(dim=0)).square().mean().sqrt().item()),
        recorded_tokens=recorded,
        used_drift=drifts is not None,
    )
