"""Masking helpers used by Phase 1/3 correctness tests and pruning flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

from .tensor import Tensor


def validate_binary_mask(
    mask: np.ndarray, expected_shape: Tuple[int, ...]
) -> np.ndarray:
    """Validate mask semantics used throughout the project.

    Returns a float64 copy so it can be multiplied with Tensor weights directly.
    """
    mask_array = np.array(mask, dtype=np.float64, copy=True)
    if mask_array.shape != expected_shape:
        raise ValueError(
            f"mask shape {mask_array.shape} does not match expected shape {expected_shape}"
        )

    unique_values = np.unique(mask_array)
    if not np.all(np.isin(unique_values, [0.0, 1.0])):
        raise ValueError("mask must be binary with values in {0, 1}")

    return mask_array


def _build_effective_weight_tensor(
    weight: Tensor,
    mask_array: np.ndarray,
    *,
    grad_sink: Optional[Callable[[np.ndarray], None]] = None,
) -> Tensor:
    """Create an effective-weight tensor with custom backward semantics.

    Backward behavior:
    - records `dL/dW_eff` to `grad_sink` when provided
    - propagates `dL/dW = M * dL/dW_eff` to the underlying weight tensor
    """
    if mask_array.shape != weight.shape:
        raise ValueError(
            f"mask shape {mask_array.shape} does not match expected shape {weight.shape}"
        )

    out = Tensor(
        weight.data * mask_array,
        requires_grad=weight.requires_grad,
        _children=(weight,),
        _op="effective_weight",
    )

    def _backward() -> None:
        if out.grad is None or not weight.requires_grad:
            return

        grad_w_eff = np.array(out.grad, dtype=np.float64, copy=True)
        if grad_sink is not None:
            grad_sink(grad_w_eff)
        weight.grad += mask_array * grad_w_eff

    out._backward = _backward
    return out


def effective_weight(weight: Tensor, mask: np.ndarray) -> Tensor:
    """Apply a fixed binary mask to a weight tensor.

    The returned tensor defines the effective parameter semantics:
    W_eff = M * W
    """
    mask_array = validate_binary_mask(mask, weight.shape)
    return _build_effective_weight_tensor(weight, mask_array)


@dataclass
class MaskedParameter:
    """Trainable parameter with an explicit binary mask.

    This class provides the Phase 3 API for prune/revive transitions while
    preserving deterministic masking semantics:

    - effective weights are `mask * tensor.data`
    - inactive entries are clamped to zero in the underlying tensor
    - optional effective-position gradients can be stored for future regrowth
    """

    tensor: Tensor
    mask: np.ndarray
    name: str = ""
    effective_position_grad: Optional[np.ndarray] = None
    _transition_counts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.tensor.requires_grad:
            raise ValueError("masked parameter tensor must require gradients")
        self.mask = validate_binary_mask(self.mask, self.tensor.shape)

        if self.effective_position_grad is not None:
            self.set_effective_position_grad(self.effective_position_grad)

        self._transition_counts = np.zeros(self.tensor.shape, dtype=np.int64)
        self.enforce_zero_inactive()

    @property
    def transition_counts(self) -> np.ndarray:
        return self._transition_counts.copy()

    def clone_mask(self) -> np.ndarray:
        return self.mask.copy()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.tensor.shape

    def effective_tensor(self) -> Tensor:
        # Avoid per-forward validation in the hot path: `self.mask` is validated on
        # construction and every mask mutation.
        return _build_effective_weight_tensor(
            self.tensor,
            self.mask,
            grad_sink=self._capture_effective_position_grad,
        )

    def effective_numpy(self) -> np.ndarray:
        return self.tensor.data * self.mask

    def active_count(self) -> int:
        return int(np.count_nonzero(self.mask))

    def inactive_count(self) -> int:
        return int(self.mask.size - np.count_nonzero(self.mask))

    def set_mask(self, new_mask: np.ndarray) -> None:
        previous_mask = self.mask.copy()
        validated = validate_binary_mask(new_mask, self.tensor.shape)

        pruned = (previous_mask == 1.0) & (validated == 0.0)
        revived = (previous_mask == 0.0) & (validated == 1.0)
        changed = pruned | revived

        self.mask = validated
        if np.any(revived):
            self.tensor.data[revived] = 0.0

        self._mark_transitions(changed)
        self._zero_effective_grad_at(changed)
        self.enforce_zero_inactive()

    def prune(self, prune_selector: np.ndarray) -> np.ndarray:
        """Prune currently active entries where selector is 1/True.

        Returns a binary array indicating which entries transitioned from
        active to inactive in this operation.
        """
        selector = validate_binary_mask(prune_selector, self.tensor.shape).astype(bool)
        currently_active = self.mask == 1.0
        actually_pruned = selector & currently_active
        self.mask = np.where(actually_pruned, 0.0, self.mask)
        self._mark_transitions(actually_pruned)
        self._zero_effective_grad_at(actually_pruned)
        self.enforce_zero_inactive()
        return actually_pruned.astype(np.float64)

    def revive(self, revive_selector: np.ndarray, *, init: str = "zeros") -> np.ndarray:
        """Revive currently inactive entries and initialize their values.

        Returns a binary array indicating which entries transitioned from
        inactive to active in this operation.
        """
        selector = validate_binary_mask(revive_selector, self.tensor.shape).astype(bool)
        currently_inactive = self.mask == 0.0
        actually_revived = selector & currently_inactive
        self.mask = np.where(actually_revived, 1.0, self.mask)
        self._mark_transitions(actually_revived)
        self._zero_effective_grad_at(actually_revived)

        if init == "zeros":
            self.tensor.data[actually_revived] = 0.0
        else:
            raise ValueError("unsupported revive init mode; expected 'zeros'")

        self.enforce_zero_inactive()
        return actually_revived.astype(np.float64)

    def enforce_zero_inactive(self) -> None:
        inactive = self.mask == 0.0
        self.tensor.data[inactive] = 0.0
        if self.tensor.grad is not None:
            self.tensor.grad[inactive] = 0.0

    def set_effective_position_grad(self, grad: np.ndarray) -> None:
        grad_array = np.array(grad, dtype=np.float64, copy=True)
        if grad_array.shape != self.tensor.shape:
            raise ValueError(
                f"effective_position_grad shape {grad_array.shape} does not match parameter shape {self.tensor.shape}"
            )
        self.effective_position_grad = grad_array

    def _capture_effective_position_grad(self, grad: np.ndarray) -> None:
        self.effective_position_grad = np.array(grad, dtype=np.float64, copy=True)

    def _mark_transitions(self, changed: np.ndarray) -> None:
        if np.any(changed):
            self._transition_counts[changed] += 1

    def _zero_effective_grad_at(self, changed: np.ndarray) -> None:
        if self.effective_position_grad is not None and np.any(changed):
            self.effective_position_grad[changed] = 0.0

    def clear_effective_position_grad(self) -> None:
        self.effective_position_grad = None
