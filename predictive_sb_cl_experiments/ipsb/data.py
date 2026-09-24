from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torchvision.datasets import MNIST


@dataclass
class TaskData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor

    def particle_stream(
        self, particles: int, batch_size: int, generator: torch.Generator
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            index = torch.randint(
                self.train_x.shape[0],
                (particles, batch_size),
                device=self.train_x.device,
                generator=generator,
            )
            yield self.train_x[index], self.train_y[index]

    def evaluation_batches(self, batch_size: int):
        for start in range(0, self.test_x.shape[0], batch_size):
            yield self.test_x[start : start + batch_size], self.test_y[start : start + batch_size]


class PermutedMNIST:
    def __init__(self, root: str, tasks: int, seed: int, device: torch.device) -> None:
        train = MNIST(root, train=True, download=False)
        test = MNIST(root, train=False, download=False)
        self.train_x = self._prepare(train.data).to(device)
        self.train_y = train.targets.long().to(device)
        self.test_x = self._prepare(test.data).to(device)
        self.test_y = test.targets.long().to(device)
        generator = torch.Generator().manual_seed(seed)
        self.permutations = [torch.arange(784, device=device)]
        self.permutations.extend(
            torch.randperm(784, generator=generator).to(device) for _ in range(tasks - 1)
        )
        self.tasks = tasks

    @staticmethod
    def _prepare(images: torch.Tensor) -> torch.Tensor:
        return images.float().reshape(-1, 784).div_(255.0).sub_(0.1307).div_(0.3081)

    def task(self, index: int) -> TaskData:
        permutation = self.permutations[index]
        return TaskData(
            self.train_x[:, permutation],
            self.train_y,
            self.test_x[:, permutation],
            self.test_y,
        )


def synthesize_inducing(
    stored: torch.Tensor,
    task_ids: torch.Tensor,
    *,
    kind: str,
    count: int,
    strength: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Extra inducing inputs derived from the stored ones, costing no storage.

    The constraint on the old predictive only binds where inducing points are,
    so coverage is bought either with memory or with synthesis.  These are the
    cheap, unlearned controls for that trade; the transported version replaces
    them if the trade turns out to be real.
    """
    if count <= 0 or kind == "none":
        return stored[:0]
    device = stored.device
    first = torch.randint(stored.shape[0], (count,), device=device, generator=generator)
    if kind == "noise":
        noise = torch.randn(
            (count, stored.shape[1]), device=device, dtype=stored.dtype, generator=generator
        )
        return stored[first] + strength * noise
    if kind == "mixup":
        # Pair within a task so a synthetic point stays on one task's manifold.
        second = torch.empty_like(first)
        for index in range(count):
            same = (task_ids == task_ids[first[index]]).nonzero(as_tuple=True)[0]
            choice = torch.randint(same.shape[0], (1,), device=device, generator=generator)
            second[index] = same[choice]
        weight = torch.rand((count, 1), device=device, generator=generator)
        weight = 0.5 + 0.5 * weight if strength <= 0 else strength + (1.0 - 2.0 * strength) * weight
        return weight * stored[first] + (1.0 - weight) * stored[second]
    raise ValueError(f"unknown inducing augmentation: {kind}")


@dataclass
class InducingMemory:
    """Fixed-size, task-balanced input memory. Labels are never retained."""

    inputs: torch.Tensor | None = None
    task_ids: torch.Tensor | None = None

    @property
    def size(self) -> int:
        return 0 if self.inputs is None else self.inputs.shape[0]

    def update(
        self,
        candidates: torch.Tensor,
        task: int,
        capacity: int,
        generator: torch.Generator,
    ) -> None:
        if capacity <= 0:
            self.inputs = None
            self.task_ids = None
            return
        groups: list[torch.Tensor] = []
        task_groups: list[torch.Tensor] = []
        quota = capacity // (task + 1)
        remainder = capacity - quota * (task + 1)
        for old_task in range(task):
            assert self.inputs is not None and self.task_ids is not None
            available = self.inputs[self.task_ids == old_task]
            count = min(available.shape[0], quota + int(old_task < remainder))
            order = torch.randperm(available.shape[0], device=available.device, generator=generator)
            groups.append(available[order[:count]])
            task_groups.append(
                torch.full((count,), old_task, device=available.device, dtype=torch.long)
            )
        current_count = min(candidates.shape[0], quota + int(task < remainder))
        order = torch.randperm(candidates.shape[0], device=candidates.device, generator=generator)
        groups.append(candidates[order[:current_count]].detach().clone())
        task_groups.append(
            torch.full((current_count,), task, device=candidates.device, dtype=torch.long)
        )
        self.inputs = torch.cat(groups, dim=0)
        self.task_ids = torch.cat(task_groups, dim=0)

