from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST


@dataclass
class PermutedMNIST:
    root: str
    tasks: int
    seed: int
    identity_first_task: bool = True

    def __post_init__(self) -> None:
        train = MNIST(self.root, train=True, download=True)
        test = MNIST(self.root, train=False, download=True)
        self.train_x = train.data.float().reshape(-1, 784).div_(255.0)
        self.train_x.sub_(0.1307).div_(0.3081)
        self.train_y = train.targets.long()
        self.test_x = test.data.float().reshape(-1, 784).div_(255.0)
        self.test_x.sub_(0.1307).div_(0.3081)
        self.test_y = test.targets.long()

        generator = torch.Generator().manual_seed(self.seed)
        self.permutations: list[torch.Tensor] = []
        for task in range(self.tasks):
            if task == 0 and self.identity_first_task:
                permutation = torch.arange(784)
            else:
                permutation = torch.randperm(784, generator=generator)
            self.permutations.append(permutation)

    def loader(
        self,
        task: int,
        *,
        train: bool,
        batch_size: int,
        workers: int,
        seed: int,
        pin_memory: bool,
    ) -> DataLoader:
        base_x = self.train_x if train else self.test_x
        base_y = self.train_y if train else self.test_y
        dataset = TensorDataset(base_x[:, self.permutations[task]], base_y)
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=workers > 0,
            generator=generator,
        )

