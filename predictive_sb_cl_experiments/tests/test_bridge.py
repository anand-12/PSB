import torch

from ipsb.bridge import (
    PredictiveEmbedding,
    birkhoff_decomposition,
    brownian_bridge_targets,
    drifted_bridge_targets,
    residual_bridge_targets,
    sinkhorn_coupling,
)


def test_sinkhorn_has_uniform_marginals():
    torch.manual_seed(0)
    source = torch.randn(8, 5)
    target = torch.randn(8, 5)
    result = sinkhorn_coupling(source, target, epsilon_ratio=0.2, iterations=1000)
    expected = torch.full((8,), 1.0 / 8)
    torch.testing.assert_close(result.coupling.sum(0), expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.coupling.sum(1), expected, atol=1e-5, rtol=1e-5)


def test_birkhoff_reconstructs_uniform_coupling():
    coupling = torch.tensor(
        [[0.20, 0.05, 0.00], [0.00, 0.20, 0.05], [0.05, 0.00, 0.20]],
        dtype=torch.float64,
    )
    coupling = coupling / coupling.sum()
    decomposition = birkhoff_decomposition(coupling)
    reconstruction = torch.zeros_like(coupling)
    for weight, permutation in decomposition:
        reconstruction[torch.arange(3), torch.as_tensor(permutation)] += weight / 3
    torch.testing.assert_close(reconstruction, coupling, atol=1e-7, rtol=1e-7)


def test_bridge_hits_target_endpoint():
    torch.manual_seed(1)
    source = torch.randn(4, 6, 3)
    source = source - source.mean(-1, keepdim=True)
    target = torch.randn(4, 6, 3)
    target = target - target.mean(-1, keepdim=True)
    embedding = PredictiveEmbedding.fit(source, target)
    permutation = torch.tensor([2, 0, 3, 1])
    generator = torch.Generator().manual_seed(3)
    path, _ = brownian_bridge_targets(
        source,
        target,
        permutation,
        embedding,
        stages=7,
        epsilon=0.5,
        schedule="cosine",
        schedule_power=0.5,
        stochastic=True,
        generator=generator,
    )
    torch.testing.assert_close(path[-1], target[permutation], atol=2e-5, rtol=2e-5)


def test_drifted_bridge_hits_target_endpoint():
    torch.manual_seed(4)
    source = torch.randn(4, 6, 3)
    source = source - source.mean(-1, keepdim=True)
    reference = []
    for fraction in torch.linspace(0.2, 1.0, 5):
        value = source + fraction * torch.randn_like(source)
        reference.append(value - value.mean(-1, keepdim=True))
    target = torch.randn(4, 6, 3)
    target = target - target.mean(-1, keepdim=True)
    embedding = PredictiveEmbedding.fit(reference[-1], target)
    permutation = torch.tensor([1, 3, 0, 2])
    generator = torch.Generator().manual_seed(8)
    path, _ = drifted_bridge_targets(
        reference,
        target,
        permutation,
        embedding,
        epsilon=0.3,
        schedule="linear",
        schedule_power=1.0,
        stochastic=True,
        generator=generator,
    )
    torch.testing.assert_close(path[-1], target[permutation], atol=2e-5, rtol=2e-5)


def test_residual_bridge_preserves_mean_and_hits_endpoint():
    torch.manual_seed(9)
    source = torch.randn(6, 7, 4)
    source = source - source.mean(-1, keepdim=True)
    target = torch.randn(6, 7, 4)
    target = target - target.mean(-1, keepdim=True)
    source_residual = source - source.mean(0, keepdim=True)
    target_residual = target - target.mean(0, keepdim=True)
    embedding = PredictiveEmbedding.fit(source_residual, target_residual)
    permutation = torch.tensor([3, 5, 0, 4, 1, 2])
    generator = torch.Generator().manual_seed(10)
    path, _ = residual_bridge_targets(
        source,
        target,
        permutation,
        embedding,
        stages=6,
        epsilon=0.4,
        schedule="cosine",
        schedule_power=1.0,
        stochastic=True,
        direct_mean=True,
        generator=generator,
    )
    for value in path:
        torch.testing.assert_close(
            value.mean(0), target.mean(0), atol=2e-5, rtol=2e-5
        )
    torch.testing.assert_close(path[-1], target[permutation], atol=2e-5, rtol=2e-5)
