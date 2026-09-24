from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from .bridge import (
    PredictiveEmbedding,
    brownian_bridge_targets,
    drifted_bridge_targets,
    functional_posterior_targets,
    optimal_permutation,
    residual_bridge_targets,
    sample_birkhoff_permutation,
    sinkhorn_coupling,
)
from .data import InducingMemory, PermutedMNIST, TaskData, synthesize_inducing
from .model import FlatMLP, centered_logits


@dataclass
class InducingMeasure:
    """A distribution over inducing inputs, resampled during the transition.

    A fixed inducing set constrains the old predictive only at its points. Fresh
    draws at every stage turn the same storage into an inducing *measure*, whose
    coverage is limited by compute rather than by memory. The targets are the
    frozen boundary model's own predictive: on old probes the functional
    posterior leaves the source predictive essentially untouched (measured
    target_old_mean_kl is ~1e-9), so no posterior step is needed for them.
    """

    memory: InducingMemory
    source_theta: torch.Tensor
    kind: str
    count: int
    strength: float

    def draw(
        self, model: FlatMLP, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.memory.inputs is not None and self.memory.task_ids is not None
        inputs = synthesize_inducing(
            self.memory.inputs,
            self.memory.task_ids,
            kind=self.kind,
            count=self.count,
            strength=self.strength,
            generator=generator,
        )
        with torch.no_grad():
            targets = centered_logits(model.logits(self.source_theta, inputs)).softmax(dim=-1)
        return inputs, targets


@dataclass
class TaskMetrics:
    task: int
    average_accuracy: float
    forgetting: float
    acquisition: float
    accuracies: list[float]
    nlls: list[float]
    eces: list[float]
    diagnostics: dict[str, float | int | str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inducing Predictive Schrodinger Bridge CL")
    parser.add_argument(
        "--method",
        choices=[
            "finetune",
            "direct",
            "random_path",
            "ot_path",
            "sb",
            "aligned_direct",
            "aligned_path",
            "aligned_sb",
            "residual_path",
            "residual_sb",
        ],
        default="sb",
    )
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=2000)
    parser.add_argument("--first-steps", type=int, default=1000)
    parser.add_argument("--stages", type=int, default=10)
    parser.add_argument("--steps-per-stage", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--reference-steps-per-stage",
        type=int,
        default=10,
        help="current-task SGD steps used to construct each reference stage",
    )
    parser.add_argument(
        "--reference-learning-rate",
        type=float,
        default=0.0,
        help="zero uses --learning-rate",
    )
    parser.add_argument("--particle-jitter", type=float, default=0.02)
    parser.add_argument("--memory-size", type=int, default=256)
    parser.add_argument("--current-probes", type=int, default=64)
    parser.add_argument(
        "--inducing-augment",
        choices=["none", "mixup", "noise"],
        default="none",
        help="synthesise extra inducing inputs from the stored ones, at no storage cost",
    )
    parser.add_argument("--inducing-augment-count", type=int, default=0)
    parser.add_argument(
        "--inducing-resample",
        action="store_true",
        help="redraw the synthetic inducing inputs at every stage (inducing measure)",
    )
    parser.add_argument(
        "--inducing-augment-strength",
        type=float,
        default=0.0,
        help="noise standard deviation, or the mixup weight floor (0 gives U[0.5,1])",
    )
    parser.add_argument("--old-distill-weight", type=float, default=5.0)
    parser.add_argument("--current-distill-weight", type=float, default=1.0)
    parser.add_argument("--predictive-rank", type=int, default=0)
    parser.add_argument("--target-prior-std", type=float, default=0.25)
    parser.add_argument("--target-likelihood-scale", type=float, default=40.0)
    parser.add_argument("--target-steps", type=int, default=200)
    parser.add_argument("--target-step-size", type=float, default=5e-4)
    parser.add_argument("--target-temperature", type=float, default=0.05)
    parser.add_argument("--sinkhorn-epsilon-ratio", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iterations", type=int, default=300)
    parser.add_argument(
        "--bridge-mc-samples",
        type=int,
        default=1,
        help="Rao-Blackwellize stochastic residual targets over this many paths",
    )
    parser.add_argument("--schedule", choices=["linear", "cosine", "power"], default="linear")
    parser.add_argument("--schedule-power", type=float, default=0.5)
    parser.add_argument(
        "--residual-mean-schedule",
        choices=["direct", "bridge"],
        default="direct",
        help="update the predictive mean immediately or along the bridge clock",
    )
    parser.add_argument(
        "--endpoint-mix",
        type=float,
        default=0.0,
        help="fraction of distillation weight reserved for the terminal marginal at every stage",
    )
    parser.add_argument("--no-bridge-noise", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data", default="../data")
    parser.add_argument("--output", default="runs")
    parser.add_argument("--tag", default="experiment")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optimize_first_task(
    theta: torch.Tensor,
    model: FlatMLP,
    task: TaskData,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    generator: torch.Generator,
) -> torch.Tensor:
    parameter = torch.nn.Parameter(theta)
    optimizer = torch.optim.Adam([parameter], lr=learning_rate)
    stream = task.particle_stream(theta.shape[0], batch_size, generator)
    for _ in range(steps):
        inputs, labels = next(stream)
        loss = model.losses(parameter, inputs, labels).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return parameter.detach()


def train_transition(
    theta: torch.Tensor,
    model: FlatMLP,
    task: TaskData,
    probes: torch.Tensor,
    old_count: int,
    stage_targets: list[torch.Tensor] | None,
    *,
    stages: int,
    steps_per_stage: int,
    batch_size: int,
    learning_rate: float,
    old_weight: float,
    current_weight: float,
    endpoint_mix: float,
    generator: torch.Generator,
    measure: "InducingMeasure | None" = None,
) -> torch.Tensor:
    parameter = torch.nn.Parameter(theta)
    optimizer = torch.optim.Adam([parameter], lr=learning_rate)
    stream = task.particle_stream(theta.shape[0], batch_size, generator)
    for stage in range(stages):
        fresh_inputs = None
        fresh_targets = None
        if measure is not None:
            fresh_inputs, fresh_targets = measure.draw(model, generator)
        target_probabilities = None
        if stage_targets is not None:
            target_probabilities = stage_targets[stage].softmax(dim=-1)
            if endpoint_mix > 0.0:
                endpoint_probabilities = stage_targets[-1].softmax(dim=-1)
                target_probabilities = (
                    (1.0 - endpoint_mix) * target_probabilities
                    + endpoint_mix * endpoint_probabilities
                )
        for _ in range(steps_per_stage):
            inputs, labels = next(stream)
            classification = model.losses(parameter, inputs, labels).mean()
            loss = classification
            if target_probabilities is not None:
                prediction = model.logits(parameter, probes).log_softmax(dim=-1)
                pointwise = (
                    target_probabilities * (target_probabilities.clamp_min(1e-12).log() - prediction)
                ).sum(dim=-1)
                if old_count:
                    old_term = pointwise[:, :old_count].mean()
                    if fresh_inputs is not None:
                        fresh_prediction = model.logits(parameter, fresh_inputs).log_softmax(dim=-1)
                        fresh_pointwise = (
                            fresh_targets
                            * (fresh_targets.clamp_min(1e-12).log() - fresh_prediction)
                        ).sum(dim=-1)
                        # Average over the union, so the total weight on the old
                        # predictive does not depend on how many points are drawn.
                        fresh_count = fresh_inputs.shape[0]
                        old_term = (
                            old_count * old_term + fresh_count * fresh_pointwise.mean()
                        ) / (old_count + fresh_count)
                    loss = loss + old_weight * old_term
                if old_count < probes.shape[0]:
                    loss = loss + current_weight * pointwise[:, old_count:].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter], 10.0)
            optimizer.step()
    return parameter.detach()


def optimization_reference_path(
    theta: torch.Tensor,
    model: FlatMLP,
    task: TaskData,
    probes: torch.Tensor,
    *,
    stages: int,
    steps_per_stage: int,
    batch_size: int,
    learning_rate: float,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    """Push particles through a private current-task SGD reference process."""
    parameter = torch.nn.Parameter(theta.detach().clone())
    optimizer = torch.optim.Adam([parameter], lr=learning_rate)
    stream = task.particle_stream(theta.shape[0], batch_size, generator)
    path: list[torch.Tensor] = []
    for _ in range(stages):
        for _ in range(steps_per_stage):
            inputs, labels = next(stream)
            loss = model.losses(parameter, inputs, labels).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter], 10.0)
            optimizer.step()
        with torch.no_grad():
            path.append(centered_logits(model.logits(parameter, probes)).detach())
    return path


@torch.no_grad()
def evaluate(
    model: FlatMLP, theta: torch.Tensor, task: TaskData, batch_size: int, bins: int = 15
) -> tuple[float, float, float]:
    correct = 0
    count = 0
    nll = 0.0
    confidences: list[torch.Tensor] = []
    outcomes: list[torch.Tensor] = []
    for inputs, labels in task.evaluation_batches(batch_size):
        probabilities = model.ensemble_probabilities(theta, inputs)
        confidence, prediction = probabilities.max(dim=-1)
        correct += int((prediction == labels).sum().item())
        count += labels.numel()
        nll += float(-probabilities.gather(1, labels[:, None]).clamp_min(1e-12).log().sum().item())
        confidences.append(confidence)
        outcomes.append((prediction == labels).float())
    confidence = torch.cat(confidences)
    outcome = torch.cat(outcomes)
    ece = torch.zeros((), device=confidence.device)
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=confidence.device)
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        selected = (confidence > low) & (confidence <= high)
        if selected.any():
            ece += selected.float().mean() * (confidence[selected].mean() - outcome[selected].mean()).abs()
    return correct / count, nll / count, float(ece.item())


def continual_summary(matrix: torch.Tensor, current: int) -> tuple[float, float, float]:
    average = float(matrix[current, : current + 1].mean().item())
    acquisition = float(torch.diagonal(matrix)[: current + 1].nanmean().item())
    if current == 0:
        return average, 0.0, acquisition
    forgetting = []
    for task in range(current):
        history = matrix[task:current, task]
        forgetting.append(history[~torch.isnan(history)].max() - matrix[current, task])
    return average, float(torch.stack(forgetting).mean().item()), acquisition


def transition_targets(
    theta: torch.Tensor,
    model: FlatMLP,
    task: TaskData,
    memory: InducingMemory,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, float | int | str], torch.Tensor]:
    order = torch.randperm(task.train_x.shape[0], device=theta.device, generator=generator)
    probe_index = order[: args.current_probes]
    current_x = task.train_x[probe_index]
    current_y = task.train_y[probe_index]
    old_x = memory.inputs
    stored_count = 0 if old_x is None else old_x.shape[0]
    if old_x is not None and args.inducing_augment_count > 0 and not args.inducing_resample:
        assert memory.task_ids is not None
        extra = synthesize_inducing(
            old_x,
            memory.task_ids,
            kind=args.inducing_augment,
            count=args.inducing_augment_count,
            strength=args.inducing_augment_strength,
            generator=generator,
        )
        old_x = torch.cat([old_x, extra], dim=0)
    old_count = 0 if old_x is None else old_x.shape[0]
    probes = current_x if old_x is None else torch.cat([old_x, current_x], dim=0)
    source = centered_logits(model.logits(theta, probes).detach())
    target = functional_posterior_targets(
        source,
        current_y,
        old_count,
        prior_std=args.target_prior_std,
        likelihood_scale=args.target_likelihood_scale,
        steps=args.target_steps,
        step_size=args.target_step_size,
        temperature=args.target_temperature,
        generator=generator,
    )
    # Method-specific transport sampling must not alter later minibatches,
    # memory selection, or future probes.  This gives every method common
    # optimization randomness after constructing the same posterior endpoint.
    transport_generator = torch.Generator(device=theta.device)
    transport_generator.set_state(generator.get_state())
    aligned = args.method.startswith("aligned_")
    residual = args.method.startswith("residual_")
    reference_path: list[torch.Tensor] | None = None
    transport_source = source
    if aligned:
        reference_generator = torch.Generator(device=theta.device)
        reference_generator.set_state(generator.get_state())
        reference_path = optimization_reference_path(
            theta,
            model,
            task,
            probes,
            stages=args.stages,
            steps_per_stage=args.reference_steps_per_stage,
            batch_size=args.batch_size,
            learning_rate=(
                args.learning_rate
                if args.reference_learning_rate <= 0.0
                else args.reference_learning_rate
            ),
            generator=reference_generator,
        )
        transport_source = reference_path[-1]
    transport_target = target
    if residual:
        transport_source = source - source.mean(dim=0, keepdim=True)
        transport_target = target - target.mean(dim=0, keepdim=True)
    embedding = PredictiveEmbedding.fit(
        transport_source, transport_target, rank=args.predictive_rank
    )
    latent_source = embedding.encode(transport_source)
    latent_target = embedding.encode(transport_target)
    cost = 0.5 * torch.cdist(latent_source, latent_target).square()
    diagnostics: dict[str, float | int | str] = {
        "old_probes": old_count,
        "stored_probes": stored_count,
        "current_probes": current_x.shape[0],
        "predictive_rank": latent_source.shape[1],
        "source_current_accuracy": float(
            (source[:, old_count:].softmax(-1).mean(0).argmax(-1) == current_y).float().mean().item()
        ),
        "target_current_accuracy": float(
            (target[:, old_count:].softmax(-1).mean(0).argmax(-1) == current_y).float().mean().item()
        ),
    }
    if reference_path is not None:
        diagnostics.update(
            reference_steps=args.stages * args.reference_steps_per_stage,
            reference_terminal_rms=float(
                (reference_path[-1] - source).square().mean().sqrt().item()
            ),
            reference_current_accuracy=float(
                (
                    reference_path[-1][:, old_count:]
                    .softmax(-1)
                    .mean(0)
                    .argmax(-1)
                    == current_y
                )
                .float()
                .mean()
                .item()
            ),
        )
    if old_count:
        source_prob = source[:, :old_count].softmax(-1).mean(dim=0)
        target_prob = target[:, :old_count].softmax(-1).mean(dim=0)
        diagnostics["target_old_mean_kl"] = float(
            (source_prob * (source_prob.clamp_min(1e-12).log() - target_prob.clamp_min(1e-12).log()))
            .sum(-1)
            .mean()
            .item()
        )

    if args.method == "random_path":
        permutation = torch.randperm(
            theta.shape[0], device=theta.device, generator=transport_generator
        )
        epsilon = 0.0
        stochastic = False
        diagnostics["coupling"] = "random_permutation"
    elif args.method in {
        "direct",
        "ot_path",
        "aligned_direct",
        "aligned_path",
        "residual_path",
    }:
        permutation = optimal_permutation(cost)
        epsilon = 0.0
        stochastic = False
        diagnostics["coupling"] = "optimal_assignment"
    elif args.method in {"sb", "aligned_sb", "residual_sb"}:
        result = sinkhorn_coupling(
            latent_source,
            latent_target,
            epsilon_ratio=args.sinkhorn_epsilon_ratio,
            iterations=args.sinkhorn_iterations,
        )
        permutation, components = sample_birkhoff_permutation(
            result.coupling, transport_generator
        )
        epsilon = result.epsilon
        stochastic = not args.no_bridge_noise
        diagnostics.update(
            coupling="sinkhorn_sb",
            sinkhorn_epsilon=epsilon,
            sinkhorn_marginal_error=result.marginal_error,
            coupling_entropy=result.normalized_entropy,
            birkhoff_components=components,
        )
    else:
        raise ValueError(f"targets are not used by method {args.method}")

    if args.method in {"direct", "aligned_direct"}:
        final = target[permutation]
        stage_targets = [final for _ in range(args.stages)]
        diagnostics["path"] = (
            "aligned_direct_endpoint" if aligned else "direct_endpoint"
        )
    elif aligned:
        assert reference_path is not None
        stage_targets, clock = drifted_bridge_targets(
            reference_path,
            target,
            permutation,
            embedding,
            epsilon=epsilon,
            schedule=args.schedule,
            schedule_power=args.schedule_power,
            stochastic=stochastic,
            generator=transport_generator,
        )
        diagnostics["path"] = (
            "stochastic_optimization_aligned_bridge"
            if stochastic
            else "optimization_aligned_conditional_mean"
        )
        diagnostics["clock_first"] = clock[0]
        diagnostics["clock_penultimate"] = clock[-2] if len(clock) > 1 else clock[-1]
        endpoint_error = (stage_targets[-1] - target[permutation]).square().mean().sqrt()
        diagnostics["endpoint_rms_error"] = float(endpoint_error.item())
    elif residual:
        sampled_paths = []
        samples = args.bridge_mc_samples if stochastic else 1
        for _ in range(samples):
            sampled, clock = residual_bridge_targets(
                source,
                target,
                permutation,
                embedding,
                stages=args.stages,
                epsilon=epsilon,
                schedule=args.schedule,
                schedule_power=args.schedule_power,
                stochastic=stochastic,
                direct_mean=args.residual_mean_schedule == "direct",
                generator=transport_generator,
            )
            sampled_paths.append(sampled)
        if samples == 1:
            stage_targets = sampled_paths[0]
        else:
            stage_targets = []
            for stage in range(args.stages):
                probabilities = torch.stack(
                    [path[stage].softmax(-1) for path in sampled_paths]
                ).mean(dim=0)
                stage_targets.append(centered_logits(probabilities.clamp_min(1e-12).log()))
        diagnostics["path"] = (
            "rao_blackwellized_residual_bridge"
            if stochastic and samples > 1
            else "stochastic_residual_bridge"
            if stochastic
            else "residual_conditional_mean"
        )
        diagnostics["bridge_mc_samples"] = samples
        diagnostics["residual_mean_schedule"] = args.residual_mean_schedule
        diagnostics["source_residual_rms"] = float(
            transport_source.square().mean().sqrt().item()
        )
        diagnostics["target_residual_rms"] = float(
            transport_target.square().mean().sqrt().item()
        )
        diagnostics["clock_first"] = clock[0]
        diagnostics["clock_penultimate"] = clock[-2] if len(clock) > 1 else clock[-1]
        endpoint_error = (stage_targets[-1] - target[permutation]).square().mean().sqrt()
        diagnostics["endpoint_rms_error"] = float(endpoint_error.item())
    else:
        stage_targets, clock = brownian_bridge_targets(
            source,
            target,
            permutation,
            embedding,
            stages=args.stages,
            epsilon=epsilon,
            schedule=args.schedule,
            schedule_power=args.schedule_power,
            stochastic=stochastic,
            generator=transport_generator,
        )
        diagnostics["path"] = "stochastic_bridge" if stochastic else "conditional_mean"
        diagnostics["clock_first"] = clock[0]
        diagnostics["clock_penultimate"] = clock[-2] if len(clock) > 1 else clock[-1]
        endpoint_error = (stage_targets[-1] - target[permutation]).square().mean().sqrt()
        diagnostics["endpoint_rms_error"] = float(endpoint_error.item())
    return probes, stage_targets, diagnostics, current_x


def run(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output) / f"{args.tag}_{args.method}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text("")

    benchmark = PermutedMNIST(args.data, args.tasks, args.permutation_seed, device)
    model = FlatMLP(args.width, args.depth)
    generator = torch.Generator(device=device).manual_seed(args.seed * 100_003 + 17)
    theta = model.initialize(
        args.particles, device=device, generator=generator, jitter=args.particle_jitter
    )
    theta = optimize_first_task(
        theta,
        model,
        benchmark.task(0),
        steps=args.first_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        generator=generator,
    )
    memory = InducingMemory()
    accuracy_matrix = torch.full((args.tasks, args.tasks), float("nan"), device=device)
    history: list[TaskMetrics] = []

    for current in range(args.tasks):
        diagnostics: dict[str, float | int | str] = {}
        task = benchmark.task(current)
        if current > 0:
            if args.method == "finetune":
                theta = train_transition(
                    theta,
                    model,
                    task,
                    torch.empty(0, 784, device=device),
                    0,
                    None,
                    stages=args.stages,
                    steps_per_stage=args.steps_per_stage,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    old_weight=0.0,
                    current_weight=0.0,
                    endpoint_mix=0.0,
                    generator=generator,
                )
                current_candidates = task.train_x[
                    torch.randperm(task.train_x.shape[0], device=device, generator=generator)[
                        : args.current_probes
                    ]
                ]
            else:
                probes, targets, diagnostics, current_candidates = transition_targets(
                    theta, model, task, memory, args, generator
                )
                measure = None
                if args.inducing_resample and args.inducing_augment_count > 0:
                    measure = InducingMeasure(
                        memory=memory,
                        source_theta=theta.detach().clone(),
                        kind=args.inducing_augment,
                        count=args.inducing_augment_count,
                        strength=args.inducing_augment_strength,
                    )
                    diagnostics["inducing_measure_draws"] = args.stages * args.inducing_augment_count
                theta = train_transition(
                    theta,
                    model,
                    task,
                    probes,
                    memory.size,
                    targets,
                    stages=args.stages,
                    steps_per_stage=args.steps_per_stage,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    old_weight=args.old_distill_weight,
                    current_weight=args.current_distill_weight,
                    endpoint_mix=args.endpoint_mix,
                    generator=generator,
                    measure=measure,
                )
                with torch.no_grad():
                    achieved = centered_logits(model.logits(theta, probes))
                    diagnostics["lift_endpoint_kl"] = float(
                        F.kl_div(
                            achieved.log_softmax(-1),
                            targets[-1].softmax(-1),
                            reduction="batchmean",
                        ).item()
                        / probes.shape[0]
                    )
        else:
            current_candidates = task.train_x[
                torch.randperm(task.train_x.shape[0], device=device, generator=generator)[
                    : args.current_probes
                ]
            ]
        memory.update(current_candidates, current, args.memory_size, generator)

        accuracies: list[float] = []
        nlls: list[float] = []
        eces: list[float] = []
        for seen in range(current + 1):
            accuracy, nll, ece = evaluate(model, theta, benchmark.task(seen), args.eval_batch_size)
            accuracy_matrix[current, seen] = accuracy
            accuracies.append(accuracy)
            nlls.append(nll)
            eces.append(ece)
        average, forgetting, acquisition = continual_summary(accuracy_matrix, current)
        record = TaskMetrics(
            task=current + 1,
            average_accuracy=average,
            forgetting=forgetting,
            acquisition=acquisition,
            accuracies=accuracies,
            nlls=nlls,
            eces=eces,
            diagnostics=diagnostics,
        )
        history.append(record)
        line = json.dumps(asdict(record))
        with metrics_path.open("a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    summary = {
        "method": args.method,
        "seed": args.seed,
        "final": asdict(history[-1]),
        "accuracy_matrix": accuracy_matrix.detach().cpu().tolist(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output


def main() -> None:
    args = parse_args()
    path = run(args)
    print(json.dumps({"output": str(path)}), flush=True)


if __name__ == "__main__":
    main()
