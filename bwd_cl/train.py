from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .data import build_benchmark
from .fisher import estimate_diagonal_fisher
from .latent import ParticleLatent
from .metrics import continual_metrics, evaluate
from .model import FlatMLP
from .posterior import GaussianEnvelope
from .bridge import TrajectoryBuffer, bridge_consistency, train_bridge_drift
from .sampler import (
    annealed_transport_transition,
    bridge_transition,
    diffusion_transition,
    guided_posterior_transition,
    merge_transition,
    optimize_particles,
    token_diffusion_transition,
)
from .tokens import TokenSpace, train_token_denoiser
from .score import train_denoiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bayesian weight diffusion on Permuted MNIST")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--benchmark", choices=["permuted", "rotated", "split"], default="permuted")
    parser.add_argument("--split-mode", choices=["domain", "class"], default="domain",
                        help="domain shares a two-way label space; class is the class-incremental setting")
    parser.add_argument("--max-angle", type=float, default=180.0,
                        help="rotated benchmark: angle spanned by the task sequence")
    parser.add_argument("--particles", type=int, default=256,
                        help="K samples of the posterior; must exceed the latent rank for the score to be estimable")
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=5000)
    parser.add_argument("--first-task-steps", type=int, default=3000)
    parser.add_argument("--transition-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--particle-jitter", type=float, default=0.02)

    parser.add_argument("--anchor-strength", type=float, default=30.0,
                        help="proximal pull toward the posterior mean; equilibrium offset is 1/(strength*precision)")
    parser.add_argument("--precision-floor", type=float, default=1e-3,
                        help="isotropic floor added to the diagonal precision")
    parser.add_argument("--precision-decay", type=float, default=1.0,
                        help="retention of earlier tasks' Fisher mass; 1 accumulates all")
    parser.add_argument("--anchor-mode", choices=["mean", "particle"], default="mean",
                        help="anchor every particle to the population mean, or each to its own last position")
    parser.add_argument("--no-precision-normalize", action="store_true",
                        help="accumulate raw Fisher mass instead of renormalising each task")
    parser.add_argument("--fisher-examples", type=int, default=20000)
    parser.add_argument("--fisher-batch-size", type=int, default=256)

    parser.add_argument("--transition-mode", choices=["bridge", "transport", "merge", "token", "diffusion", "proximal"], default="bridge",
                        help="diffusion is the method; proximal is the envelope-only ablation")
    parser.add_argument("--latent-rank", type=int, default=32,
                        help="score-model latent dimension; keep well below --particles")
    parser.add_argument("--sigma-max-weight", type=float, default=0.5,
                        help="forward noise level as a multiple of each tensor's RMS")
    parser.add_argument("--sigma-min-weight", type=float, default=0.01)
    parser.add_argument("--posterior-width", type=float, default=0.3,
                        help="envelope standard deviation scale; sets how hard Tweedie shrinks toward the mean")
    parser.add_argument("--guidance-rate", type=float, default=0.01,
                        help="Adam step size for the likelihood operator, in units of tensor RMS")
    parser.add_argument("--guidance-ramp", type=float, default=1.0)
    parser.add_argument("--bridge-warmup-tasks", type=int, default=3,
                        help="transitions traversed with the reference kernel before the drift is fitted")
    parser.add_argument("--bridge-reference-moves", type=int, default=100,
                        help="kernel steps per bridge stage while recording reference trajectories")
    parser.add_argument("--bridge-deployed-moves", type=int, default=25,
                        help="kernel steps per bridge stage once the learned drift carries the transition")
    parser.add_argument("--drift-scale", type=float, default=1.0)
    parser.add_argument("--ipf-iterations", type=int, default=3,
                        help="Iterative Markovian Fitting rounds; 1 is plain bridge matching")
    parser.add_argument("--ipf-warm-start", action="store_true",
                        help="initialise each IMF round from the previous round's drift")
    parser.add_argument("--drift-steps", type=int, default=3000)
    parser.add_argument("--drift-hidden", type=int, default=512)
    parser.add_argument("--drift-batch", type=int, default=512)
    parser.add_argument("--record-particles", type=int, default=8,
                        help="particles sampled per stage for the trajectory buffer")
    parser.add_argument("--anneal-steps", type=int, default=25,
                        help="number of beta stages along L^beta q_t")
    parser.add_argument("--likelihood-scale", type=float, default=1000.0,
                        help="effective sample count multiplying the mean log-likelihood in the SMC weights")
    parser.add_argument("--weight-batch-size", type=int, default=2048,
                        help="shared batch used to compute importance weights")
    parser.add_argument("--resample-threshold", type=float, default=0.5,
                        help="resample when ESS falls below this fraction of K; 0 disables reweighting")
    parser.add_argument("--token-hidden", type=int, default=512)
    parser.add_argument("--token-score-steps", type=int, default=3000)
    parser.add_argument("--token-score-batch", type=int, default=512)
    parser.add_argument("--ambient-noise", action="store_true",
                        help="corrupt the full parameter vector instead of only the learned subspace")
    parser.add_argument("--correction-weight", type=float, default=0.0,
                        help="strength of the learned non-Gaussian manifold pull; 0 disables it")
    parser.add_argument("--score-steps", type=int, default=2000)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--score-learning-rate", type=float, default=2e-3)
    parser.add_argument("--score-hidden", type=int, default=256)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--exploration-noise", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--all-tasks-permuted", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="runs_bwd")
    parser.add_argument("--tag", default="default")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_first_population(
    model: FlatMLP, benchmark: PermutedMNIST, args: argparse.Namespace, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(args.seed + 7)
    start = model.initialize(device=device, generator=generator)
    jitter = torch.randn(
        (args.particles, model.num_parameters), device=device, dtype=start.dtype, generator=generator
    )
    population = start.unsqueeze(0) + args.particle_jitter * jitter * model.block_rms(start).unsqueeze(0)
    return optimize_particles(
        population,
        model=model,
        task=benchmark.task(0),
        steps=args.first_task_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        generator=generator,
    )


def run(args: argparse.Namespace) -> Path:
    if args.particles < 2:
        raise ValueError("--particles must be at least 2")
    if args.anchor_strength < 0.0:
        raise ValueError("--anchor-strength must be nonnegative")
    if not 0.0 < args.precision_decay <= 1.0:
        raise ValueError("--precision-decay must lie in (0,1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")

    seed_everything(args.seed)
    device = torch.device(args.device)
    output_directory = Path(args.output) / f"{args.tag}_seed{args.seed}"
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    log_path = output_directory / "metrics.jsonl"
    log_path.write_text("")

    benchmark, output_dim = build_benchmark(
        args.benchmark,
        args.data,
        tasks=args.tasks,
        seed=args.permutation_seed,
        device=device,
        identity_first_task=not args.all_tasks_permuted,
        split_mode=args.split_mode,
        max_angle=args.max_angle,
    )
    # Split MNIST fixes its own task count from the label pairs.
    args.tasks = benchmark.tasks
    model = FlatMLP(width=args.width, depth=args.depth, output_dim=output_dim)
    networks = build_first_population(model, benchmark, args, device)
    envelope = GaussianEnvelope.initial(
        networks if args.anchor_mode == "particle" else networks.mean(dim=0)
    )

    accuracy_matrix = torch.full((args.tasks, args.tasks), float("nan"))
    histories: list[dict[str, object]] = []
    trajectory_buffer = TrajectoryBuffer(model.layers) if args.transition_mode == "bridge" else None
    drifts = None
    backward_drifts = None
    drift_report = None
    backward_report = None
    consistency = None
    imf_iteration = 0
    started = time.time()

    for current_task in range(args.tasks):
        score_report = None
        transition_report = None
        latent_rank = None
        if current_task > 0:
            latent = None
            denoiser = None
            if args.transition_mode != "token" and (
                args.correction_weight > 0.0 or args.transition_mode == "diffusion"
            ):
                latent = ParticleLatent.fit(networks, model.blocks, rank=args.latent_rank)
                latent_rank = latent.rank
            if args.correction_weight > 0.0 and latent is not None:
                denoiser, score_report = train_denoiser(
                    latent.clean_latents,
                    steps=args.score_steps,
                    batch_size=args.score_batch_size,
                    learning_rate=args.score_learning_rate,
                    sigma_min=args.sigma_min,
                    sigma_max=args.sigma_max,
                    hidden_dimension=args.score_hidden,
                    seed=args.seed * 1_000_000 + current_task * 10_000 + 17,
                )
            transition_generator = torch.Generator(device=device).manual_seed(
                args.seed * 1_000_000 + current_task * 10_000 + 31
            )
            if args.transition_mode == "bridge":
                token_space = TokenSpace(model, networks)
                # Every boundary is recorded: each one supplies a fresh coupling
                # for the next Iterative Markovian Fitting round.
                recording = True
                fine = drifts is None
                networks, transition_report = bridge_transition(
                    networks,
                    model=model,
                    envelope=envelope,
                    token_space=token_space,
                    drifts=drifts,
                    buffer=trajectory_buffer,
                    task=benchmark.task(current_task),
                    stages=max(1, args.anneal_steps),
                    move_steps=(
                        args.bridge_reference_moves if fine else args.bridge_deployed_moves
                    ),
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    anchor_strength=args.anchor_strength,
                    precision_floor=args.precision_floor,
                    drift_scale=args.drift_scale,
                    record_particles=args.record_particles,
                    generator=transition_generator,
                )
                if (
                    current_task >= args.bridge_warmup_tasks
                    and imf_iteration < args.ipf_iterations
                    and trajectory_buffer.size > 0
                ):
                    # One IMF round: Markovian projection in each direction on
                    # the current coupling, then discard it so the next boundary
                    # supplies a fresh one generated by the drift just fitted.
                    imf_iteration += 1
                    common = dict(
                        steps=args.drift_steps,
                        batch_size=args.drift_batch,
                        learning_rate=args.score_learning_rate,
                        hidden=args.drift_hidden,
                        seed=args.seed * 1_000_000 + 7717 + 101 * imf_iteration,
                        iteration=imf_iteration,
                    )
                    drifts, drift_report = train_bridge_drift(
                        trajectory_buffer, model, direction="forward",
                        existing=drifts if args.ipf_warm_start else None, **common,
                    )
                    backward_drifts, backward_report = train_bridge_drift(
                        trajectory_buffer, model, direction="backward",
                        existing=backward_drifts if args.ipf_warm_start else None, **common,
                    )
                    consistency = bridge_consistency(
                        trajectory_buffer, drifts, backward_drifts, model
                    )
                    trajectory_buffer = TrajectoryBuffer(model.layers)
            elif args.transition_mode == "transport":
                stages = max(1, args.anneal_steps)
                networks, transition_report = annealed_transport_transition(
                    networks,
                    model=model,
                    envelope=envelope,
                    task=benchmark.task(current_task),
                    anneal_steps=stages,
                    move_steps=max(1, args.transition_steps // stages),
                    batch_size=args.batch_size,
                    weight_batch_size=args.weight_batch_size,
                    learning_rate=args.learning_rate,
                    likelihood_scale=args.likelihood_scale,
                    anchor_strength=args.anchor_strength,
                    precision_floor=args.precision_floor,
                    noise_scale=args.exploration_noise,
                    resample_threshold=args.resample_threshold,
                    generator=transition_generator,
                )
            elif args.transition_mode == "merge":
                if args.anchor_strength > 0.0:
                    # A mild anchor bounds how far the new expert drifts. The
                    # Gaussian product is only accurate for nearby experts, and
                    # unconstrained drift compounds over a long stream.
                    fresh, _ = guided_posterior_transition(
                        networks,
                        model=model,
                        envelope=envelope,
                        latent=None,
                        denoiser=None,
                        task=benchmark.task(current_task),
                        steps=args.transition_steps,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        anchor_strength=args.anchor_strength,
                        precision_floor=args.precision_floor,
                        correction_weight=0.0,
                        sigma_max=args.sigma_max,
                        sigma_min=args.sigma_min,
                        exploration_noise=args.exploration_noise,
                        generator=transition_generator,
                    )
                else:
                    fresh = optimize_particles(
                        networks,
                        model=model,
                        task=benchmark.task(current_task),
                        steps=args.transition_steps,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        generator=transition_generator,
                    )
                fresh_fisher, _ = estimate_diagonal_fisher(
                    model,
                    fresh,
                    benchmark.task(current_task),
                    examples=args.fisher_examples,
                    batch_size=args.fisher_batch_size,
                    generator=torch.Generator(device=device).manual_seed(
                        args.seed * 1_000_000 + current_task * 10_000 + 61
                    ),
                )
                networks, transition_report = merge_transition(
                    networks,
                    model=model,
                    envelope=envelope,
                    task=benchmark.task(current_task),
                    steps=args.transition_steps,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    precision_floor=args.precision_floor,
                    new_precision=fresh_fisher,
                    fresh=fresh,
                )
            elif args.transition_mode == "token":
                token_space = TokenSpace(model, networks)
                denoisers = None
                if args.correction_weight > 0.0:
                    denoisers = []
                    reports = []
                    for layer in range(model.layers):
                        denoiser_l, report_l = train_token_denoiser(
                            token_space.training_tokens(networks, layer),
                            steps=args.token_score_steps,
                            batch_size=args.token_score_batch,
                            learning_rate=args.score_learning_rate,
                            sigma_min=args.sigma_min,
                            sigma_max=args.sigma_max,
                            hidden=args.token_hidden,
                            seed=args.seed * 1_000_000 + current_task * 10_000 + 101 * layer + 5,
                        )
                        denoisers.append(denoiser_l)
                        reports.append(report_l)
                    score_report = reports[0]
                    latent_rank = sum(r.tokens for r in reports)
                networks, transition_report = token_diffusion_transition(
                    networks,
                    model=model,
                    envelope=envelope,
                    token_space=token_space,
                    denoisers=denoisers,
                    task=benchmark.task(current_task),
                    steps=args.transition_steps,
                    batch_size=args.batch_size,
                    sigma_max=args.sigma_max,
                    sigma_min=args.sigma_min,
                    guidance_rate=args.guidance_rate,
                    guidance_ramp=args.guidance_ramp,
                    anchor_strength=args.anchor_strength,
                    precision_floor=args.precision_floor,
                    correction_weight=args.correction_weight,
                    generator=transition_generator,
                )
            elif args.transition_mode == "diffusion":
                networks, transition_report = diffusion_transition(
                    networks,
                    model=model,
                    envelope=envelope,
                    latent=latent,
                    denoiser=denoiser,
                    task=benchmark.task(current_task),
                    steps=args.transition_steps,
                    batch_size=args.batch_size,
                    sigma_max=args.sigma_max_weight,
                    sigma_min=args.sigma_min_weight,
                    posterior_width=args.posterior_width,
                    guidance_rate=args.guidance_rate,
                    guidance_ramp=args.guidance_ramp,
                    anchor_strength=args.anchor_strength,
                    precision_floor=args.precision_floor,
                    correction_weight=args.correction_weight,
                    latent_sigma_max=args.sigma_max,
                    latent_sigma_min=args.sigma_min,
                    block_scale=model.block_rms(envelope.centre),
                    generator=transition_generator,
                    subspace_noise=not args.ambient_noise,
                )
            else:
                networks, transition_report = guided_posterior_transition(
                    networks,
                model=model,
                envelope=envelope,
                latent=latent,
                denoiser=denoiser,
                    task=benchmark.task(current_task),
                    steps=args.transition_steps,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    anchor_strength=args.anchor_strength,
                    precision_floor=args.precision_floor,
                    correction_weight=args.correction_weight,
                    sigma_max=args.sigma_max,
                    sigma_min=args.sigma_min,
                    exploration_noise=args.exploration_noise,
                    generator=transition_generator,
                )

        task_accuracies = []
        predictive_losses = []
        for evaluation_task in range(current_task + 1):
            accuracy, predictive_loss = evaluate(
                model, networks, benchmark.task(evaluation_task), args.eval_batch_size
            )
            accuracy_matrix[current_task, evaluation_task] = accuracy
            task_accuracies.append(accuracy)
            predictive_losses.append(predictive_loss)
        average_accuracy, forgetting, backward_transfer = continual_metrics(accuracy_matrix, current_task)

        fisher_generator = torch.Generator(device=device).manual_seed(
            args.seed * 1_000_000 + current_task * 10_000 + 43
        )
        fisher, fisher_report = estimate_diagonal_fisher(
            model,
            networks,
            benchmark.task(current_task),
            examples=args.fisher_examples,
            batch_size=args.fisher_batch_size,
            generator=fisher_generator,
        )
        envelope = envelope.absorb(
            networks if args.anchor_mode == "particle" else networks.mean(dim=0),
            fisher,
            decay=args.precision_decay,
            normalize=not args.no_precision_normalize,
        )

        record: dict[str, object] = {
            "method": "bwd",
            "benchmark": args.benchmark,
            "tag": args.tag,
            "seed": args.seed,
            "task": current_task + 1,
            "average_accuracy": average_accuracy,
            "forgetting": forgetting,
            "backward_transfer": backward_transfer,
            "predictive_loss": float(sum(predictive_losses) / len(predictive_losses)),
            "task_accuracies": task_accuracies,
            "particles": args.particles,
            "transition_mode": args.transition_mode,
            "anchor_strength": args.anchor_strength,
            "correction_weight": args.correction_weight,
            "latent_rank": latent_rank,
            "elapsed_seconds": time.time() - started,
        }
        record.update(envelope.stiffness_report())
        if drift_report is not None:
            record.update({f"drift_{k}": v for k, v in asdict(drift_report).items()})
        if backward_report is not None:
            record["drift_backward_explained"] = backward_report.explained
        if consistency is not None:
            record.update({f"bridge_{k}": v for k, v in asdict(consistency).items()})
        record["imf_iteration"] = imf_iteration
        record.update({f"fisher_{k}": v for k, v in asdict(fisher_report).items()})
        if score_report is not None:
            record.update({f"score_{k}": v for k, v in asdict(score_report).items()})
        if transition_report is not None:
            record.update({f"transition_{k}": v for k, v in asdict(transition_report).items()})
        histories.append(record)
        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    (output_directory / "summary.json").write_text(
        json.dumps(
            {
                "config": vars(args),
                "history": histories,
                "accuracy_matrix": accuracy_matrix.tolist(),
                "parameter_count_per_network": model.num_parameters,
            },
            indent=2,
        )
        + "\n"
    )
    torch.save(
        {
            "networks": networks.detach().cpu(),
            "envelope_mean": envelope.mean.detach().cpu(),
            "envelope_precision": envelope.precision.detach().cpu(),
            "accuracy_matrix": accuracy_matrix,
            "args": vars(args),
        },
        output_directory / "latest.pt",
    )
    return output_directory / "summary.json"


def main() -> None:
    summary_path = run(parse_args())
    print(json.dumps({"completed": True, "summary": str(summary_path)}), flush=True)


if __name__ == "__main__":
    main()
