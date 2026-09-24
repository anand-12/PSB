from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch
from torch import nn

from .model import FlatMLP
from .tokens import TokenSpace


class StageEmbedding(nn.Module):
    """Fourier features of the bridge coordinate beta in [0,1]."""

    def __init__(self, dimension: int = 32) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("stage embedding dimension must be even")
        frequencies = torch.exp(torch.linspace(0.0, math.log(64.0), dimension // 2))
        self.register_buffer("frequencies", frequencies)

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        angles = beta.unsqueeze(-1) * self.frequencies.unsqueeze(0) * math.pi
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class BridgeDrift(nn.Module):
    """Displacement of one neuron's weight vector along the posterior bridge.

    Conditioned on the neuron's current weights, the current task's likelihood
    gradient for that neuron, and the bridge coordinate beta. The gradient
    conditioning is what makes the drift transferable: it is not a memorised
    displacement for one task but a rule for how far to move given the local
    evidence, so a drift fitted on early transitions applies to later ones.
    """

    def __init__(self, dimension: int, hidden: int = 512, stage_dimension: int = 32) -> None:
        super().__init__()
        self.stage_embedding = StageEmbedding(stage_dimension)
        self.network = nn.Sequential(
            nn.Linear(3 * dimension + stage_dimension + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dimension),
        )
        # Start as a no-op so an untrained drift degrades to the reference
        # kernel rather than to noise.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        tokens: torch.Tensor,
        gradients: torch.Tensor,
        anchors: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        gradient_scale = gradients.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
        token_scale = tokens.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
        anchor_scale = anchors.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
        features = [
            tokens / token_scale,
            gradients / gradient_scale,
            anchors / anchor_scale,
            gradient_scale.log(),
            token_scale.log(),
            anchor_scale.log(),
        ]
        stage = self.stage_embedding(beta)
        while stage.ndim < tokens.ndim:
            stage = stage.unsqueeze(1)
        features.append(stage.expand(*tokens.shape[:-1], stage.shape[-1]))
        prediction = self.network(torch.cat(features, dim=-1))
        # Predict displacement in units of the token's own scale.
        return prediction * token_scale


@dataclass
class TrajectorySample:
    layer: int
    tokens: torch.Tensor
    gradients: torch.Tensor
    anchors: torch.Tensor
    betas: torch.Tensor
    targets: torch.Tensor


class TrajectoryBuffer:
    """Paired (state, gradient, beta) -> displacement samples from reference runs.

    One buffer per layer, capped so that a long stream cannot grow the stored
    state: the drift is a fixed-size object like every other part of the method.
    """

    def __init__(self, layers: int, capacity: int = 200_000) -> None:
        self.capacity = capacity
        self.tokens: list[list[torch.Tensor]] = [[] for _ in range(layers)]
        self.gradients: list[list[torch.Tensor]] = [[] for _ in range(layers)]
        self.anchors: list[list[torch.Tensor]] = [[] for _ in range(layers)]
        self.betas: list[list[torch.Tensor]] = [[] for _ in range(layers)]
        self.targets: list[list[torch.Tensor]] = [[] for _ in range(layers)]

    def add(self, layer: int, tokens, gradients, anchors, beta, target) -> None:
        self.tokens[layer].append(tokens.detach().reshape(-1, tokens.shape[-1]))
        self.gradients[layer].append(gradients.detach().reshape(-1, gradients.shape[-1]))
        self.anchors[layer].append(anchors.detach().reshape(-1, anchors.shape[-1]))
        count = self.tokens[layer][-1].shape[0]
        self.betas[layer].append(torch.full((count,), float(beta), device=tokens.device))
        self.targets[layer].append(target.detach().reshape(-1, target.shape[-1]))

    def assemble(self, layer: int) -> TrajectorySample | None:
        if not self.tokens[layer]:
            return None
        tokens = torch.cat(self.tokens[layer], dim=0)
        gradients = torch.cat(self.gradients[layer], dim=0)
        anchors = torch.cat(self.anchors[layer], dim=0)
        betas = torch.cat(self.betas[layer], dim=0)
        targets = torch.cat(self.targets[layer], dim=0)
        if tokens.shape[0] > self.capacity:
            keep = torch.randperm(tokens.shape[0], device=tokens.device)[: self.capacity]
            tokens, gradients, anchors = tokens[keep], gradients[keep], anchors[keep]
            betas, targets = betas[keep], targets[keep]
        return TrajectorySample(layer, tokens, gradients, anchors, betas, targets)

    @property
    def size(self) -> int:
        return sum(t.shape[0] for layer in self.tokens for t in layer)


@dataclass
class DriftReport:
    layers: int
    samples: int
    initial_loss: float
    final_loss: float
    explained: float
    direction: str = "forward"
    iteration: int = 0


@dataclass
class BridgeConsistency:
    """Cycle error of the fitted half-bridges.

    Applying the forward drift and then the backward drift should return a token
    to where it started. The residual measures how far the pair is from a
    genuine Schrodinger bridge, and is the quantity Iterative Markovian Fitting
    drives down. Reported relative to the displacement magnitude so it is
    comparable across layers and iterations.
    """

    absolute: float
    relative: float


def train_bridge_drift(
    buffer: TrajectoryBuffer,
    model: FlatMLP,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    hidden: int,
    seed: int,
    existing: list[BridgeDrift] | None = None,
    direction: str = "forward",
    iteration: int = 0,
    ema_decay: float = 0.995,
) -> tuple[list[BridgeDrift], DriftReport]:
    """Fit one half-bridge by matching displacements along the reference path.

    ``direction="forward"`` regresses the displacement that carries a token from
    beta to beta+1; ``"backward"`` regresses the reverse, taking the arrived
    token back. One half-bridge alone is bridge matching. Alternating the two
    while re-coupling the endpoints is Iterative Markovian Fitting, whose fixed
    point is the Schrodinger bridge; ``iteration`` records which round this is.
    """
    if direction not in {"forward", "backward"}:
        raise ValueError(f"unknown half-bridge direction: {direction}")
    drifts: list[BridgeDrift] = []
    total = 0
    initial_total = 0.0
    final_total = 0.0
    variance_total = 0.0
    residual_total = 0.0

    for layer in range(model.layers):
        sample = buffer.assemble(layer)
        if sample is None:
            raise ValueError(f"no trajectory samples recorded for layer {layer}")
        device = sample.tokens.device
        generator = torch.Generator(device=device).manual_seed(seed + 17 * layer)
        # The backward half-bridge starts from the arrived token and undoes the
        # displacement, so its inputs and targets are the forward pair reversed.
        if direction == "forward":
            inputs, targets = sample.tokens, sample.targets
        else:
            inputs, targets = sample.tokens + sample.targets, -sample.targets
        drift = (
            copy.deepcopy(existing[layer])
            if existing is not None
            else BridgeDrift(inputs.shape[1], hidden).to(device)
        )
        # A previously fitted drift was frozen for inference; warm-starting the
        # next IMF round has to make it trainable again.
        for parameter in drift.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(drift.parameters(), lr=learning_rate, weight_decay=1e-6)
        ema_state = copy.deepcopy(drift.state_dict())

        holdout = torch.Generator(device=device).manual_seed(seed + 977 * (layer + 1))
        pick = torch.randperm(sample.tokens.shape[0], device=device, generator=holdout)
        validation = pick[: min(4096, pick.shape[0])]

        @torch.no_grad()
        def validation_loss() -> float:
            prediction = drift(
                inputs[validation],
                sample.gradients[validation],
                sample.anchors[validation],
                sample.betas[validation],
            )
            return float((prediction - targets[validation]).square().mean().item())

        initial = validation_loss()
        drift.train()
        for _ in range(steps):
            index = torch.randint(
                sample.tokens.shape[0], (batch_size,), device=device, generator=generator
            )
            prediction = drift(
                inputs[index], sample.gradients[index], sample.anchors[index], sample.betas[index]
            )
            loss = (prediction - targets[index]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(drift.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                for name, value in drift.state_dict().items():
                    if value.is_floating_point():
                        ema_state[name].mul_(ema_decay).add_(value, alpha=1.0 - ema_decay)
                    else:
                        ema_state[name].copy_(value)

        drift.load_state_dict(ema_state)
        drift.eval()
        for parameter in drift.parameters():
            parameter.requires_grad_(False)
        final = validation_loss()

        drifts.append(drift)
        total += sample.tokens.shape[0]
        initial_total += initial
        final_total += final
        variance_total += float(targets[validation].square().mean().item())
        residual_total += final

    explained = 1.0 - residual_total / max(variance_total, 1e-30)
    return drifts, DriftReport(
        layers=model.layers,
        samples=total,
        initial_loss=initial_total / model.layers,
        final_loss=final_total / model.layers,
        explained=explained,
        direction=direction,
        iteration=iteration,
    )


@torch.no_grad()
def bridge_consistency(
    buffer: TrajectoryBuffer,
    forward: list[BridgeDrift],
    backward: list[BridgeDrift],
    model: FlatMLP,
    *,
    samples: int = 4096,
) -> BridgeConsistency:
    """Measure how far the fitted half-bridge pair is from being a bridge."""
    residual = 0.0
    magnitude = 0.0
    for layer in range(model.layers):
        sample = buffer.assemble(layer)
        if sample is None:
            continue
        take = min(samples, sample.tokens.shape[0])
        z = sample.tokens[:take]
        g = sample.gradients[:take]
        a = sample.anchors[:take]
        b = sample.betas[:take]
        step = forward[layer](z, g, a, b)
        back = backward[layer](z + step, g, a, b)
        residual += float((step + back).square().mean().item())
        magnitude += float(step.square().mean().item())
    return BridgeConsistency(
        absolute=residual / model.layers,
        relative=residual / max(magnitude, 1e-30),
    )


@torch.no_grad()
def apply_drift(
    theta: torch.Tensor,
    *,
    token_space: TokenSpace,
    drifts: list[BridgeDrift],
    gradient: torch.Tensor,
    anchor: torch.Tensor,
    beta: float,
    scale: float,
) -> torch.Tensor:
    """One learned bridge step over every layer's neuron tokens."""
    updated = theta
    for layer, drift in enumerate(drifts):
        tokens = token_space.encode(updated, layer)
        gradient_tokens = token_space.encode_raw(gradient, layer)
        anchor_tokens = token_space.encode_raw(anchor, layer)
        level = torch.full((tokens.shape[0],), beta, device=tokens.device, dtype=tokens.dtype)
        step = drift(tokens, gradient_tokens, anchor_tokens, level)
        updated = token_space.decode_into(updated, layer, tokens + scale * step)
    return updated
