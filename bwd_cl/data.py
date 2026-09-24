from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST


@dataclass
class TaskData:
    """One permuted task with both splits resident on the compute device."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor

    def stream(self, batch_size: int, generator: torch.Generator) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield shuffled training minibatches forever."""
        count = self.train_x.shape[0]
        while True:
            order = torch.randperm(count, device=self.train_x.device, generator=generator)
            for start in range(0, count - batch_size + 1, batch_size):
                index = order[start : start + batch_size]
                yield self.train_x[index], self.train_y[index]

    def particle_stream(
        self, particles: int, batch_size: int, generator: torch.Generator
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield an independent minibatch per particle, shaped [K,B,...].

        Giving each particle its own data order is what makes the population a
        genuine ensemble rather than K copies following one trajectory.
        """
        count = self.train_x.shape[0]
        while True:
            index = torch.randint(
                count, (particles, batch_size), device=self.train_x.device, generator=generator
            )
            yield self.train_x[index], self.train_y[index]

    def evaluation_batches(self, batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for start in range(0, self.test_x.shape[0], batch_size):
            yield self.test_x[start : start + batch_size], self.test_y[start : start + batch_size]


class PermutedMNIST:
    """Permuted MNIST held entirely in device memory.

    Keeping the tensors on the GPU removes the host-to-device copy from every
    optimisation step, which matters here because a task transition performs
    thousands of small steps over a network that is itself tiny.
    """

    def __init__(
        self,
        root: str,
        *,
        tasks: int,
        seed: int,
        device: torch.device,
        identity_first_task: bool = True,
    ) -> None:
        train = MNIST(root, train=True, download=True)
        test = MNIST(root, train=False, download=True)
        train_x = train.data.float().reshape(-1, 784).div_(255.0).sub_(0.1307).div_(0.3081)
        test_x = test.data.float().reshape(-1, 784).div_(255.0).sub_(0.1307).div_(0.3081)
        self._train_x = train_x.to(device)
        self._train_y = train.targets.long().to(device)
        self._test_x = test_x.to(device)
        self._test_y = test.targets.long().to(device)

        generator = torch.Generator().manual_seed(seed)
        self.permutations: list[torch.Tensor] = []
        for task in range(tasks):
            if task == 0 and identity_first_task:
                permutation = torch.arange(784)
            else:
                permutation = torch.randperm(784, generator=generator)
            self.permutations.append(permutation.to(device))
        self.tasks = tasks

    def task(self, index: int) -> TaskData:
        permutation = self.permutations[index]
        return TaskData(
            train_x=self._train_x[:, permutation],
            train_y=self._train_y,
            test_x=self._test_x[:, permutation],
            test_y=self._test_y,
        )


def _load_mnist(root: str, device: torch.device):
    train = MNIST(root, train=True, download=True)
    test = MNIST(root, train=False, download=True)
    train_x = train.data.float().reshape(-1, 784).div_(255.0).sub_(0.1307).div_(0.3081)
    test_x = test.data.float().reshape(-1, 784).div_(255.0).sub_(0.1307).div_(0.3081)
    return (
        train_x.to(device),
        train.targets.long().to(device),
        test_x.to(device),
        test.targets.long().to(device),
    )


class RotatedMNIST:
    """Each task rotates the input by a fixed angle; the label space is shared.

    Structurally a sibling of Permuted MNIST -- the classes never change, only
    the input geometry -- so consecutive posteriors stay close and the geometric
    path between them remains in functional territory.
    """

    output_dim = 10

    def __init__(
        self,
        root: str,
        *,
        tasks: int,
        seed: int,
        device: torch.device,
        max_angle: float = 180.0,
        identity_first_task: bool = True,
    ) -> None:
        self._train_x, self._train_y, self._test_x, self._test_y = _load_mnist(root, device)
        self.tasks = tasks
        self.device = device
        step = max_angle / max(tasks - 1, 1) if identity_first_task else max_angle / tasks
        self.angles = [step * index for index in range(tasks)]
        self._cache: dict[int, TaskData] = {}

    def _rotate(self, flat: torch.Tensor, angle: float) -> torch.Tensor:
        if abs(angle) < 1e-8:
            return flat
        images = flat.reshape(-1, 1, 28, 28)
        rotated = TF.rotate(images, angle, interpolation=TF.InterpolationMode.BILINEAR)
        return rotated.reshape(-1, 784)

    def task(self, index: int) -> TaskData:
        # Deliberately not cached: holding every rotated copy costs ~2.2 GB,
        # while recomputing is one grid_sample per call (~20 ms).
        angle = self.angles[index]
        return TaskData(
            train_x=self._rotate(self._train_x, angle),
            train_y=self._train_y,
            test_x=self._rotate(self._test_x, angle),
            test_y=self._test_y,
        )


class SplitMNIST:
    """Disjoint label pairs, one shared head.

    ``mode="domain"`` relabels each pair to {0,1}, so the label space is shared
    across tasks and a single two-way head is well defined throughout.
    ``mode="class"`` keeps the original labels against a ten-way head, which is
    the class-incremental setting: only two classes are ever present at a time,
    so nothing re-calibrates the absent classes' logits. Parameter-space methods
    are not expected to work there, and it is included to measure that rather
    than to assume it.
    """

    def __init__(
        self,
        root: str,
        *,
        tasks: int,
        seed: int,
        device: torch.device,
        mode: str = "domain",
        identity_first_task: bool = True,
    ) -> None:
        if mode not in {"domain", "class"}:
            raise ValueError(f"unknown split mode: {mode}")
        self._train_x, self._train_y, self._test_x, self._test_y = _load_mnist(root, device)
        self.pairs = [(2 * i, 2 * i + 1) for i in range(5)][:tasks]
        self.tasks = len(self.pairs)
        self.mode = mode
        self.output_dim = 2 if mode == "domain" else 10
        self._cache: dict[int, TaskData] = {}

    def task(self, index: int) -> TaskData:
        if index not in self._cache:
            low, high = self.pairs[index]
            def carve(x, y):
                keep = (y == low) | (y == high)
                labels = y[keep]
                if self.mode == "domain":
                    labels = (labels == high).long()
                return x[keep], labels
            train_x, train_y = carve(self._train_x, self._train_y)
            test_x, test_y = carve(self._test_x, self._test_y)
            self._cache[index] = TaskData(
                train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y
            )
        return self._cache[index]


def build_benchmark(name: str, root: str, *, tasks: int, seed: int, device, **kwargs):
    """Return (benchmark, output_dim) for one of the supported task streams."""
    if name == "permuted":
        bench = PermutedMNIST(root, tasks=tasks, seed=seed, device=device,
                              identity_first_task=kwargs.get("identity_first_task", True))
        return bench, 10
    if name == "rotated":
        bench = RotatedMNIST(root, tasks=tasks, seed=seed, device=device,
                             max_angle=kwargs.get("max_angle", 180.0))
        return bench, bench.output_dim
    if name == "split":
        bench = SplitMNIST(root, tasks=tasks, seed=seed, device=device,
                           mode=kwargs.get("split_mode", "domain"))
        return bench, bench.output_dim
    raise ValueError(f"unknown benchmark: {name}")
