import torch
from torch.utils.data import DataLoader, TensorDataset

from gwd_cl.curvature import curvature_mobility, estimate_diagonal_fisher
from gwd_cl.diffusion import (
    distill_sequential_posterior,
    likelihood_guided_reverse_diffusion,
    train_denoiser,
)
from gwd_cl.model import FlatMLP
from gwd_cl.parameter_space import ParameterDistribution
from gwd_cl.train import optimize_one_network


def test_batched_flat_mlp_matches_individual_networks() -> None:
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(3)
    model = FlatMLP(input_dim=4, width=5, depth=2, output_dim=3)
    networks = torch.stack([model.initialize(device=device, generator=generator) for _ in range(3)])
    inputs = torch.randn(7, 4, generator=generator)
    batched = model.logits(networks, inputs)
    individual = torch.stack([model.logits(network, inputs) for network in networks])
    torch.testing.assert_close(batched, individual)


def test_pca_coordinates_reconstruct_retained_networks() -> None:
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(7)
    model = FlatMLP(input_dim=4, width=4, depth=1, output_dim=2)
    center = model.initialize(device=device, generator=generator)
    networks = center + 0.01 * torch.randn(4, center.numel(), generator=generator)
    space = ParameterDistribution.fit(networks, model.blocks)
    normalized = space.normalize(networks)
    reconstruction = space.decode(space.parallel_component(normalized))
    torch.testing.assert_close(reconstruction, networks, atol=2e-5, rtol=2e-5)


def test_complete_guided_diffusion_transition_is_finite() -> None:
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(11)
    model = FlatMLP(input_dim=4, width=5, depth=1, output_dim=2)
    center = model.initialize(device=device, generator=generator)
    networks = center + 0.01 * torch.randn(4, center.numel(), generator=generator)
    space = ParameterDistribution.fit(networks, model.blocks)
    denoiser, report = train_denoiser(
        space.clean_latents,
        steps=8,
        batch_size=8,
        learning_rate=1e-3,
        sigma_min=0.01,
        sigma_max=0.1,
        hidden_dimension=16,
        seed=13,
    )
    inputs = torch.randn(32, 4, generator=generator)
    targets = torch.randint(0, 2, (32,), generator=generator)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8, shuffle=False)
    updated, reverse_report = likelihood_guided_reverse_diffusion(
        networks,
        space=space,
        denoiser=denoiser,
        model=model,
        train_loader=loader,
        steps=8,
        sigma_min=0.01,
        sigma_max=0.1,
        solution_kernel_std=0.05,
        guidance_scale=1.0,
        guidance_ramp_power=1.0,
        gradient_clip=5.0,
        guidance_mode="proximal",
        posterior_learning_rate=0.01,
        posterior_max_step=0.02,
        previous_precision=torch.ones(networks.shape[1]),
        curvature_strength=1.0,
        curvature_damping=0.1,
        mobility_minimum=0.05,
        mobility_maximum=3.0,
        seed=17,
    )
    assert updated.shape == networks.shape
    assert torch.isfinite(updated).all()
    assert report.final_validation_loss >= 0.0
    assert reverse_report.forward_noise_rms > 0.0


def test_task1_optimizer_executes() -> None:
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(23)
    model = FlatMLP(input_dim=4, width=5, depth=1, output_dim=2)
    initial = model.initialize(device=device, generator=generator)
    inputs = torch.randn(16, 4, generator=generator)
    targets = torch.randint(0, 2, (16,), generator=generator)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8, shuffle=False)
    updated = optimize_one_network(
        model,
        initial,
        loader,
        epochs=1,
        learning_rate=1e-3,
        device=device,
    )
    assert updated.shape == initial.shape
    assert torch.isfinite(updated).all()
    assert not torch.equal(updated, initial)


def test_online_fisher_and_mobility_are_finite() -> None:
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(31)
    model = FlatMLP(input_dim=4, width=5, depth=1, output_dim=2)
    networks = torch.stack([model.initialize(device=device, generator=generator) for _ in range(3)])
    inputs = torch.randn(24, 4, generator=generator)
    targets = torch.randint(0, 2, (24,), generator=generator)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    fisher, report = estimate_diagonal_fisher(
        model, networks, loader, maximum_batches=2, device=device
    )
    mobility = curvature_mobility(
        fisher,
        torch.ones_like(fisher),
        strength=1.0,
        damping=0.1,
        minimum=0.05,
        maximum=3.0,
    )
    assert report.batches == 2
    assert torch.isfinite(fisher).all()
    assert torch.isfinite(mobility).all()
    assert float(mobility.min()) >= 0.05
    assert float(mobility.max()) <= 3.0


def test_sequential_score_distillation_is_finite() -> None:
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(41)
    classifier = FlatMLP(input_dim=4, width=5, depth=1, output_dim=2)
    center = classifier.initialize(device=device, generator=generator)
    old_networks = center + 0.01 * torch.randn(4, center.numel(), generator=generator)
    new_networks = old_networks + 0.005 * torch.randn(4, center.numel(), generator=generator)
    old_space = ParameterDistribution.fit(old_networks, classifier.blocks)
    new_space = ParameterDistribution.fit(new_networks, classifier.blocks)
    teacher, _ = train_denoiser(
        old_space.clean_latents,
        steps=4,
        batch_size=4,
        learning_rate=1e-3,
        sigma_min=0.05,
        sigma_max=0.2,
        hidden_dimension=16,
        seed=43,
    )
    inputs = torch.randn(16, 4, generator=generator)
    targets = torch.randint(0, 2, (16,), generator=generator)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    student, report = distill_sequential_posterior(
        new_networks,
        old_space=old_space,
        old_denoiser=teacher,
        new_space=new_space,
        classifier=classifier,
        train_loader=loader,
        steps=4,
        batch_size=4,
        learning_rate=1e-3,
        sigma_min=0.05,
        sigma_max=0.2,
        hidden_dimension=16,
        posterior_weight=0.65,
        likelihood_strength=2.0,
        maximum_correction_rms=3.0,
        seed=47,
    )
    prediction = student(new_space.clean_latents, torch.full((4,), 0.1))
    assert torch.isfinite(prediction).all()
    assert report.final_loss >= 0.0
