from __future__ import annotations

import torch

from .data import TaskData
from .model import FlatMLP


@torch.no_grad()
def evaluate(model: FlatMLP, networks: torch.Tensor, task: TaskData, batch_size: int) -> tuple[float, float]:
    correct = 0
    count = 0
    negative_log_likelihood = 0.0
    for inputs, targets in task.evaluation_batches(batch_size):
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
    forgetting = []
    backward = []
    for task in range(current_task):
        history = accuracy_matrix[task:current_task, task]
        best = history[~torch.isnan(history)].max()
        forgetting.append(best - accuracy_matrix[current_task, task])
        backward.append(accuracy_matrix[current_task, task] - accuracy_matrix[task, task])
    return (
        average_accuracy,
        float(torch.stack(forgetting).mean().item()),
        float(torch.stack(backward).mean().item()),
    )
