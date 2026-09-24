"""MNIST task streams as fixed-shape device arrays.

Every task is a ``TaskArrays`` with identical shapes within a benchmark, so one
compiled function serves all tasks: the task is an argument, never a constant.
Images stay uint8 on device and are permuted and normalised per batch.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
DEFAULT_MNIST = Path(__file__).resolve().parents[1] / "data" / "mnist.npz"


class TaskArrays(NamedTuple):
    train_x: np.ndarray   # uint8 [N_pad, 784]
    train_y: np.ndarray   # int32 [N_pad]
    n_train: np.ndarray   # int32 []
    test_x: np.ndarray    # uint8 [M_pad, 784]
    test_y: np.ndarray    # int32 [M_pad]
    n_test: np.ndarray    # int32 []
    perm: np.ndarray      # int32 [784]


def load_mnist(path: str | Path = DEFAULT_MNIST) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in ("train_x", "train_y", "test_x", "test_y")}


def _pad(x: np.ndarray, y: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    extra = length - x.shape[0]
    if extra <= 0:
        return x, y
    return (np.concatenate([x, np.zeros((extra, x.shape[1]), x.dtype)]),
            np.concatenate([y, np.zeros(extra, y.dtype)]))


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _rotate(images: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-8:
        return images
    from scipy.ndimage import rotate
    stack = images.reshape(-1, 28, 28).astype(np.float32)
    turned = rotate(stack, angle, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0)
    return np.clip(np.rint(turned), 0, 255).astype(np.uint8).reshape(-1, 784)


def build_tasks(config: dict, mnist: dict[str, np.ndarray], eval_batch: int) -> tuple[list[TaskArrays], int]:
    """Return the task list (host arrays) and the output dimension."""
    name, tasks = config["benchmark"], config["tasks"]
    identity = np.arange(784, dtype=np.int32)
    test_len = _round_up(mnist["test_x"].shape[0], eval_batch)
    test_x, test_y = _pad(mnist["test_x"], mnist["test_y"], test_len)
    n_train = np.int32(mnist["train_x"].shape[0])
    n_test = np.int32(mnist["test_x"].shape[0])

    if name == "permuted":
        rng = np.random.default_rng(config["permutation_seed"])
        out = []
        for index in range(tasks):
            perm = identity if index == 0 else rng.permutation(784).astype(np.int32)
            out.append(TaskArrays(mnist["train_x"], mnist["train_y"], n_train,
                                  test_x, test_y, n_test, perm))
        return out, 10

    if name == "rotated":
        step = config["max_angle"] / max(tasks - 1, 1)
        out = []
        for index in range(tasks):
            angle = step * index
            rotated_test, _ = _pad(_rotate(mnist["test_x"], angle), mnist["test_y"], test_len)
            out.append(TaskArrays(_rotate(mnist["train_x"], angle), mnist["train_y"], n_train,
                                  rotated_test, test_y, n_test, identity))
        return out, 10

    if name == "split":
        pairs = [(2 * i, 2 * i + 1) for i in range(5)]
        domain = config["split_mode"] == "domain"
        carved = []
        for low, high in pairs:
            parts = []
            for x, y in ((mnist["train_x"], mnist["train_y"]), (mnist["test_x"], mnist["test_y"])):
                keep = (y == low) | (y == high)
                labels = y[keep]
                labels = (labels == high).astype(np.int32) if domain else labels.astype(np.int32)
                parts.append((x[keep], labels))
            carved.append(parts)
        train_len = max(p[0][0].shape[0] for p in carved)
        test_len = _round_up(max(p[1][0].shape[0] for p in carved), eval_batch)
        out = []
        for (train_part, test_part) in carved:
            tx, ty = _pad(*train_part, train_len)
            vx, vy = _pad(*test_part, test_len)
            out.append(TaskArrays(tx, ty, np.int32(train_part[0].shape[0]),
                                  vx, vy, np.int32(test_part[0].shape[0]), identity))
        return out, (2 if domain else 10)

    raise ValueError(f"unknown benchmark {name}")
