"""Differentiable helper operations built from Tensor primitives."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


def softmax(logits: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax.

    The shift term is treated as a constant to preserve numerical stability
    without adding max() gradients to the graph.
    """
    shift = np.max(logits.data, axis=axis, keepdims=True)
    shifted = logits - Tensor(shift)
    exps = shifted.exp()
    return exps / exps.sum(axis=axis, keepdims=True)


def log_softmax(logits: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable log-softmax built from Tensor operations."""
    shift = np.max(logits.data, axis=axis, keepdims=True)
    shifted = logits - Tensor(shift)
    log_norm = shifted.exp().sum(axis=axis, keepdims=True).log()
    return shifted - log_norm


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Cross-entropy loss for integer class labels.

    Args:
        logits: Tensor of shape (batch, classes).
        targets: Integer labels of shape (batch,).
    """
    if logits.data.ndim != 2:
        raise ValueError("logits must have shape (batch, classes)")

    batch_size, num_classes = logits.data.shape
    if batch_size == 0:
        raise ValueError("cross_entropy does not support empty batches")
    if num_classes == 0:
        raise ValueError("logits must include at least one class")

    raw_targets = np.asarray(targets)
    if raw_targets.ndim != 1 or raw_targets.shape[0] != batch_size:
        raise ValueError("targets must have shape (batch,)")
    if not np.issubdtype(raw_targets.dtype, np.integer):
        raise TypeError("targets must be an integer array of class indices")

    targets = raw_targets.astype(np.int64, copy=False)
    if np.any(targets < 0) or np.any(targets >= num_classes):
        raise ValueError(
            f"target class indices must be in [0, {num_classes - 1}]"
        )

    one_hot = np.zeros_like(logits.data)
    one_hot[np.arange(batch_size), targets] = 1.0
    one_hot_tensor = Tensor(one_hot)

    log_probs = log_softmax(logits, axis=1)
    selected = (log_probs * one_hot_tensor).sum(axis=1)
    return -(selected.mean())


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred - target
    return (diff * diff).mean()
