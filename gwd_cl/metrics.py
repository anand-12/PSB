from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from .model import FlatMLP


@torch.no_grad()
def evaluate(
    model: FlatMLP,
    networks: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    correct = 0
    count = 0
    negative_log_likelihood = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        probabilities = model.ensemble_probabilities(networks, inputs)
        correct += int((probabilities.argmax(dim=-1) == targets).sum().item())
        count += targets.numel()
        selected = probabilities.gather(1, targets[:, None]).clamp_min(1e-12)
        negative_log_likelihood += float(-selected.log().sum().item())
    return correct / count, negative_log_likelihood / count


def continual_metrics(accuracy_matrix: torch.Tensor, current_task: int) -> tuple[float, float, float]:
    row = accuracy_matrix[current_task, : current_task + 1]
    average_accuracy = float(row.mean().item())
    if current_task == 0:
        return average_accuracy, 0.0, 0.0
    forgetting_values = []
    backward_values = []
    for task in range(current_task):
        best_previous = accuracy_matrix[task:current_task, task]
        best_previous = best_previous[~torch.isnan(best_previous)].max()
        forgetting_values.append(best_previous - accuracy_matrix[current_task, task])
        backward_values.append(accuracy_matrix[current_task, task] - accuracy_matrix[task, task])
    forgetting = float(torch.stack(forgetting_values).mean().item())
    backward_transfer = float(torch.stack(backward_values).mean().item())
    return average_accuracy, forgetting, backward_transfer

