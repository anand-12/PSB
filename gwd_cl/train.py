from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .curvature import estimate_diagonal_fisher
from .data import PermutedMNIST
from .diffusion import (
    antithetic_noise,
    distill_sequential_posterior,
    likelihood_guided_reverse_diffusion,
    train_denoiser,
)
from .metrics import continual_metrics, evaluate
from .model import FlatMLP
from .parameter_space import ParameterDistribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Likelihood-guided weight diffusion on Permuted MNIST")
    parser.add_argument(
        "--method", choices=["gwd", "spsd"], default="spsd",
        help="spsd carries the sequential posterior score; gwd refits it from K networks",
    )
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--particles", type=int, default=8, help="number K of nearby trained networks")
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--task1-epochs", type=int, default=15)
    parser.add_argument("--particle-refine-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--particle-jitter",
        type=float,
        default=0.03,
        help="Task-1 perturbation as a fraction of each parameter tensor's RMS",
    )
    parser.add_argument("--pca-rank", type=int, default=0, help="0 uses the maximum rank K-1")
    parser.add_argument("--score-steps", type=int, default=1500)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--score-learning-rate", type=float, default=2e-3)
    parser.add_argument("--score-hidden", type=int, default=128)
    parser.add_argument("--distill-steps", type=int, default=1000)
    parser.add_argument("--distill-batch-size", type=int, default=16)
    parser.add_argument(
        "--distill-posterior-weight", type=float, default=0.65,
        help="weight on the carried posterior-score target; the remainder is ordinary denoising",
    )
    parser.add_argument("--distill-likelihood-strength", type=float, default=4.0)
    parser.add_argument("--distill-max-correction-rms", type=float, default=3.0)
    parser.add_argument("--sigma-max", type=float, default=0.50)
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument(
        "--solution-kernel-std",
        type=float,
        default=0.15,
        help="task-independent width of the full-support old solution distribution",
    )
    parser.add_argument("--reverse-steps", type=int, default=1200)
    parser.add_argument("--guidance-scale", type=float, default=12.0)
    parser.add_argument("--guidance-ramp-power", type=float, default=1.0)
    parser.add_argument("--guidance-gradient-clip", type=float, default=5.0)
    parser.add_argument(
        "--guidance-mode",
        choices=["proximal", "score"],
        default="proximal",
        help="proximal alternates prior denoising and current-data updates; score is the original ablation",
    )
    parser.add_argument(
        "--posterior-learning-rate",
        type=float,
        default=0.05,
        help="Adam step size in normalized parameter coordinates for proximal posterior guidance",
    )
    parser.add_argument(
        "--posterior-max-step",
        type=float,
        default=0.05,
        help="maximum RMS normalized-parameter displacement in one data-consistency step",
    )
    parser.add_argument(
        "--posterior-task-decay",
        type=float,
        default=1.0,
        help="decay exponent: the Task-t posterior step is base_lr/(t-1)^exponent",
    )
    parser.add_argument(
        "--posterior-min-learning-rate",
        type=float,
        default=0.0125,
        help="task-independent floor preventing late tasks from becoming underfit",
    )
    parser.add_argument(
        "--curvature-strength",
        type=float,
        default=1.0,
        help="0 disables the online-Fisher mobility; 1 uses inverse square-root Fisher geometry",
    )
    parser.add_argument("--curvature-damping", type=float, default=0.1)
    parser.add_argument("--mobility-minimum", type=float, default=0.05)
    parser.add_argument("--mobility-maximum", type=float, default=3.0)
    parser.add_argument("--fisher-batches", type=int, default=64)
    parser.add_argument("--fisher-batch-size", type=int, default=64)
    parser.add_argument(
        "--precision-decay",
        type=float,
        default=1.0,
        help="online precision retention; 1 accumulates all tasks",
    )
    parser.add_argument(
        "--score",
        choices=["learned", "analytic"],
        default="learned",
        help="learned is the proposed method; analytic is a Gaussian-mixture score ablation",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=0,
        help="fixed across model seeds so every run sees the same task sequence",
    )
    parser.add_argument("--all-tasks-permuted", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="runs")
    parser.add_argument("--tag", default="default")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optimize_one_network(
    model: FlatMLP,
    initial: torch.Tensor,
    loader,
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> torch.Tensor:
    parameters = torch.nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.Adam(params=[parameters], lr=learning_rate)
    for _ in range(epochs):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = model.losses(parameters, inputs, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return parameters.detach()


def task1_parameter_scale(model: FlatMLP, parameters: torch.Tensor) -> torch.Tensor:
    scale = torch.empty_like(parameters)
    for block in model.blocks:
        block_values = parameters[block.start : block.stop]
        block_rms = block_values.square().mean().sqrt().clamp_min(1e-2)
        scale[block.start : block.stop] = block_rms
    return scale


def initialize_task1_ensemble(
    model: FlatMLP,
    optimum: torch.Tensor,
    data: PermutedMNIST,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(args.seed + 20_011)
    perturbations = antithetic_noise(
        (args.particles, model.num_parameters),
        device=device,
        generator=generator,
        dtype=optimum.dtype,
    )
    perturbations *= args.particle_jitter * task1_parameter_scale(model, optimum)
    initial_copies = optimum.unsqueeze(0) + perturbations
    refined = []
    # Independent optimization gives each perturbed copy a distinct minibatch path.
    for particle in range(args.particles):
        loader = data.loader(
            0,
            train=True,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=args.seed * 100_000 + particle + 101,
            pin_memory=device.type == "cuda",
        )
        refined.append(
            optimize_one_network(
                model,
                initial_copies[particle],
                loader,
                epochs=args.particle_refine_epochs,
                learning_rate=args.learning_rate,
                device=device,
            )
        )
    return torch.stack(refined)


def json_safe_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: value for key, value in vars(args).items()}


def run(args: argparse.Namespace) -> Path:
    if args.particles < 2:
        raise ValueError("--particles must be at least 2")
    if not 0.0 < args.sigma_min < args.sigma_max:
        raise ValueError("require 0 < sigma-min < sigma-max")
    if args.solution_kernel_std <= 0.0:
        raise ValueError("--solution-kernel-std must be positive")
    if args.posterior_learning_rate <= 0.0 or args.posterior_max_step <= 0.0:
        raise ValueError("posterior learning rate and maximum step must be positive")
    if args.posterior_min_learning_rate < 0.0:
        raise ValueError("--posterior-min-learning-rate must be nonnegative")
    if args.posterior_task_decay < 0.0:
        raise ValueError("--posterior-task-decay must be nonnegative")
    if not 0.0 <= args.curvature_strength <= 2.0:
        raise ValueError("--curvature-strength must lie in [0,2]")
    if args.curvature_damping <= 0.0 or args.mobility_minimum <= 0.0:
        raise ValueError("curvature damping and mobility bounds must be positive")
    if args.mobility_maximum < args.mobility_minimum:
        raise ValueError("mobility maximum must be at least its minimum")
    if args.fisher_batches < 1 or args.fisher_batch_size < 1:
        raise ValueError("Fisher batch counts and sizes must be positive")
    if not 0.0 < args.precision_decay <= 1.0:
        raise ValueError("--precision-decay must lie in (0,1]")
    if not 0.0 <= args.distill_posterior_weight <= 1.0:
        raise ValueError("--distill-posterior-weight must lie in [0,1]")
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu for a CPU run")
    device = torch.device(args.device)
    output_directory = Path(args.output) / f"{args.tag}_seed{args.seed}"
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "config.json").write_text(json.dumps(json_safe_args(args), indent=2) + "\n")
    log_path = output_directory / "metrics.jsonl"
    log_path.write_text("")

    benchmark = PermutedMNIST(
        args.data,
        tasks=args.tasks,
        seed=args.permutation_seed,
        identity_first_task=not args.all_tasks_permuted,
    )
    model = FlatMLP(width=args.width, depth=args.depth)
    initialization_generator = torch.Generator(device=device).manual_seed(args.seed + 7)
    initial = model.initialize(device=device, generator=initialization_generator)
    task1_loader = benchmark.loader(
        0,
        train=True,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed * 100_000 + 1,
        pin_memory=device.type == "cuda",
    )
    optimum = optimize_one_network(
        model,
        initial,
        task1_loader,
        epochs=args.task1_epochs,
        learning_rate=args.learning_rate,
        device=device,
    )
    networks = initialize_task1_ensemble(model, optimum, benchmark, args, device)

    accuracy_matrix = torch.full((args.tasks, args.tasks), float("nan"))
    histories: list[dict[str, object]] = []
    online_precision = None
    carried_space = None
    carried_denoiser = None
    if args.method == "spsd":
        rank = None if args.pca_rank == 0 else args.pca_rank
        carried_space = ParameterDistribution.fit(networks, model.blocks, rank=rank)
        carried_denoiser, _ = train_denoiser(
            carried_space.clean_latents,
            steps=args.score_steps,
            batch_size=args.score_batch_size,
            learning_rate=args.score_learning_rate,
            sigma_min=(args.solution_kernel_std**2 + args.sigma_min**2) ** 0.5,
            sigma_max=(args.solution_kernel_std**2 + args.sigma_max**2) ** 0.5,
            hidden_dimension=args.score_hidden,
            seed=args.seed * 1_000_000 + 17,
        )
    started = time.time()
    for current_task in range(args.tasks):
        score_report = None
        reverse_report = None
        distillation_report = None
        retained_rank = None
        effective_posterior_learning_rate = None
        if current_task > 0:
            rank = None if args.pca_rank == 0 else args.pca_rank
            if args.method == "spsd":
                assert carried_space is not None and carried_denoiser is not None
                space, denoiser = carried_space, carried_denoiser
            else:
                space = ParameterDistribution.fit(networks, model.blocks, rank=rank)
                denoiser, score_report = train_denoiser(
                    space.clean_latents,
                    steps=args.score_steps,
                    batch_size=args.score_batch_size,
                    learning_rate=args.score_learning_rate,
                    sigma_min=(args.solution_kernel_std**2 + args.sigma_min**2) ** 0.5,
                    sigma_max=(args.solution_kernel_std**2 + args.sigma_max**2) ** 0.5,
                    hidden_dimension=args.score_hidden,
                    seed=args.seed * 1_000_000 + current_task * 10_000 + 17,
                )
            retained_rank = space.basis.shape[0]
            current_loader = benchmark.loader(
                current_task,
                train=True,
                batch_size=args.batch_size,
                workers=args.workers,
                seed=args.seed * 100_000 + current_task * 1_000 + 29,
                pin_memory=device.type == "cuda",
            )
            effective_posterior_learning_rate = max(
                args.posterior_min_learning_rate,
                args.posterior_learning_rate / (current_task**args.posterior_task_decay),
            )
            networks, reverse_report = likelihood_guided_reverse_diffusion(
                networks,
                space=space,
                denoiser=denoiser,
                model=model,
                train_loader=current_loader,
                steps=args.reverse_steps,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                solution_kernel_std=args.solution_kernel_std,
                guidance_scale=args.guidance_scale,
                guidance_ramp_power=args.guidance_ramp_power,
                gradient_clip=args.guidance_gradient_clip,
                guidance_mode=args.guidance_mode,
                posterior_learning_rate=effective_posterior_learning_rate,
                posterior_max_step=args.posterior_max_step,
                previous_precision=online_precision,
                curvature_strength=args.curvature_strength,
                curvature_damping=args.curvature_damping,
                mobility_minimum=args.mobility_minimum,
                mobility_maximum=args.mobility_maximum,
                seed=args.seed * 1_000_000 + current_task * 10_000 + 31,
                analytic_score=args.score == "analytic",
            )

            # Unlike GWD, SPSD never reconstructs the complete history from K
            # point estimates alone. It transfers the previous score function
            # into a single new student before discarding the teacher.
            if args.method == "spsd" and current_task < args.tasks - 1:
                new_space = ParameterDistribution.fit(networks, model.blocks, rank=rank)
                distill_loader = benchmark.loader(
                    current_task,
                    train=True,
                    batch_size=args.fisher_batch_size,
                    workers=args.workers,
                    seed=args.seed * 100_000 + current_task * 1_000 + 37,
                    pin_memory=device.type == "cuda",
                )
                new_denoiser, distillation_report = distill_sequential_posterior(
                    networks,
                    old_space=space,
                    old_denoiser=denoiser,
                    new_space=new_space,
                    classifier=model,
                    train_loader=distill_loader,
                    steps=args.distill_steps,
                    batch_size=args.distill_batch_size,
                    learning_rate=args.score_learning_rate,
                    sigma_min=(args.solution_kernel_std**2 + args.sigma_min**2) ** 0.5,
                    sigma_max=(args.solution_kernel_std**2 + args.sigma_max**2) ** 0.5,
                    hidden_dimension=args.score_hidden,
                    posterior_weight=args.distill_posterior_weight,
                    likelihood_strength=args.distill_likelihood_strength,
                    maximum_correction_rms=args.distill_max_correction_rms,
                    seed=args.seed * 1_000_000 + current_task * 10_000 + 53,
                )
                carried_space, carried_denoiser = new_space, new_denoiser

        predictive_losses = []
        task_accuracies = []
        for evaluation_task in range(current_task + 1):
            test_loader = benchmark.loader(
                evaluation_task,
                train=False,
                batch_size=args.eval_batch_size,
                workers=args.workers,
                seed=args.seed + evaluation_task,
                pin_memory=device.type == "cuda",
            )
            accuracy, predictive_loss = evaluate(model, networks, test_loader, device)
            accuracy_matrix[current_task, evaluation_task] = accuracy
            task_accuracies.append(accuracy)
            predictive_losses.append(predictive_loss)

        average_accuracy, forgetting, backward_transfer = continual_metrics(accuracy_matrix, current_task)
        fisher_loader = benchmark.loader(
            current_task,
            train=True,
            batch_size=args.fisher_batch_size,
            workers=args.workers,
            seed=args.seed * 100_000 + current_task * 1_000 + 43,
            pin_memory=device.type == "cuda",
        )
        current_fisher, fisher_report = estimate_diagonal_fisher(
            model,
            networks,
            fisher_loader,
            maximum_batches=args.fisher_batches,
            device=device,
        )
        if online_precision is None:
            online_precision = current_fisher
        else:
            online_precision = args.precision_decay * online_precision + current_fisher
        record: dict[str, object] = {
            "method": args.method if args.score == "learned" else f"{args.method}_analytic_score",
            "tag": args.tag,
            "seed": args.seed,
            "task": current_task + 1,
            "average_accuracy": average_accuracy,
            "forgetting": forgetting,
            "backward_transfer": backward_transfer,
            "predictive_loss": float(sum(predictive_losses) / len(predictive_losses)),
            "task_accuracies": task_accuracies,
            "particles": args.particles,
            "guidance_mode": args.guidance_mode,
            "effective_posterior_learning_rate": effective_posterior_learning_rate,
            "pca_rank": retained_rank,
            "curvature_strength": args.curvature_strength,
            "precision_mean": float(online_precision.mean().item()),
            "elapsed_seconds": time.time() - started,
        }
        record.update({f"fisher_{key}": value for key, value in asdict(fisher_report).items()})
        if score_report is not None:
            record.update({f"score_{key}": value for key, value in asdict(score_report).items()})
        if reverse_report is not None:
            record.update({f"reverse_{key}": value for key, value in asdict(reverse_report).items()})
        if distillation_report is not None:
            record.update({f"distill_{key}": value for key, value in asdict(distillation_report).items()})
        histories.append(record)
        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        torch.save(
            {
                "task": current_task + 1,
                "networks": networks.detach().cpu(),
                "accuracy_matrix": accuracy_matrix,
                "online_precision": online_precision.detach().cpu(),
                "args": json_safe_args(args),
            },
            output_directory / "latest.pt",
        )

    summary = {
        "config": json_safe_args(args),
        "history": histories,
        "accuracy_matrix": accuracy_matrix.tolist(),
        "parameter_count_per_network": model.num_parameters,
    }
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path


def main() -> None:
    summary_path = run(parse_args())
    print(json.dumps({"completed": True, "summary": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()
