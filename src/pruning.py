"""Masking helpers used by Phase 1/3 correctness tests and pruning flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class SparsityStats:
    total: int
    active: int
    inactive: int
    sparsity: float


@dataclass(frozen=True)
class PruneStepReport:
    target_sparsity: float
    enforced_target_sparsity: float
    active_before: int
    active_after: int
    total: int
    requested_pruned: int
    applied_pruned: int
    max_prunable_with_guards: int
    reached_target: bool
    limitation: Optional[str]


@dataclass(frozen=True)
class LayerConnectivityReport:
    layer_index: int
    layer_name: str
    shape: Tuple[int, ...]
    active: int
    total: int
    zero_rows: Optional[int]
    zero_cols: Optional[int]


@dataclass(frozen=True)
class HiddenUnitConnectivityReport:
    left_layer_index: int
    right_layer_index: int
    hidden_width: int
    no_incoming: int
    no_outgoing: int
    disconnected_either_side: int


@dataclass
class TaylorScoreAccumulator:
    """Accumulate minibatch Taylor saliency scores across an epoch."""

    sums: list[np.ndarray]
    batches: int = 0

    @classmethod
    def from_masked_params(
        cls,
        masked_params: Sequence[MaskedParameter],
    ) -> "TaylorScoreAccumulator":
        if not masked_params:
            raise ValueError("masked_params must contain at least one parameter")
        sums = [np.zeros_like(masked_param.mask) for masked_param in masked_params]
        return cls(sums=sums, batches=0)

    def add_batch(self, masked_params: Sequence[MaskedParameter]) -> None:
        batch_scores = compute_taylor_scores(masked_params)
        if len(batch_scores) != len(self.sums):
            raise ValueError("batch score count does not match accumulator layout")
        for idx, batch_score in enumerate(batch_scores):
            if batch_score.shape != self.sums[idx].shape:
                raise ValueError("batch score shape does not match accumulator layout")
            self.sums[idx] += batch_score
        self.batches += 1

    def mean_scores(self) -> list[np.ndarray]:
        if self.batches <= 0:
            raise ValueError("cannot compute mean scores without accumulated batches")
        return [score_sum / float(self.batches) for score_sum in self.sums]


def cubic_sparsity_target(
    *,
    epoch: int,
    prune_start_epoch: int,
    prune_end_epoch: int,
    final_sparsity: float,
) -> float:
    """Cubic sparsity ramp from 0 to final_sparsity across configured epochs."""
    if prune_start_epoch < 1:
        raise ValueError("prune_start_epoch must be >= 1")
    if prune_end_epoch < prune_start_epoch:
        raise ValueError("prune_end_epoch must be >= prune_start_epoch")
    if not (0.0 <= final_sparsity < 1.0):
        raise ValueError("final_sparsity must be in [0, 1)")

    if epoch < prune_start_epoch:
        return 0.0
    if epoch >= prune_end_epoch:
        return float(final_sparsity)

    span = prune_end_epoch - prune_start_epoch
    if span == 0:
        return float(final_sparsity)

    progress = (epoch - prune_start_epoch) / span
    ramp = 1.0 - (1.0 - progress) ** 3
    return float(final_sparsity * ramp)


def compute_mask_sparsity(masked_params: Sequence[MaskedParameter]) -> SparsityStats:
    if not masked_params:
        raise ValueError("masked_params must contain at least one parameter")

    total = 0
    active = 0
    for masked_param in masked_params:
        total += int(masked_param.mask.size)
        active += int(np.count_nonzero(masked_param.mask))

    inactive = total - active
    sparsity = inactive / total
    return SparsityStats(total=total, active=active, inactive=inactive, sparsity=float(sparsity))


def _validate_layer_scores(
    masked_params: Sequence[MaskedParameter],
    layer_scores: Sequence[np.ndarray],
) -> None:
    if len(masked_params) != len(layer_scores):
        raise ValueError("layer_scores length must match masked_params length")
    for masked_param, score in zip(masked_params, layer_scores):
        score_array = np.asarray(score, dtype=np.float64)
        if score_array.shape != masked_param.shape:
            raise ValueError("layer score shape does not match masked parameter shape")
        if not np.isfinite(score_array).all():
            raise ValueError("layer scores must be finite")


def compute_layer_connectivity_report(
    masked_params: Sequence[MaskedParameter],
) -> list[LayerConnectivityReport]:
    reports: list[LayerConnectivityReport] = []
    for layer_index, masked_param in enumerate(masked_params):
        mask = masked_param.mask
        layer_name = masked_param.name or f"layer_{layer_index}"
        active = int(np.count_nonzero(mask))
        total = int(mask.size)

        zero_rows: Optional[int] = None
        zero_cols: Optional[int] = None
        if mask.ndim == 2:
            row_counts = np.count_nonzero(mask, axis=1)
            col_counts = np.count_nonzero(mask, axis=0)
            zero_rows = int(np.count_nonzero(row_counts == 0))
            zero_cols = int(np.count_nonzero(col_counts == 0))

        reports.append(
            LayerConnectivityReport(
                layer_index=layer_index,
                layer_name=layer_name,
                shape=mask.shape,
                active=active,
                total=total,
                zero_rows=zero_rows,
                zero_cols=zero_cols,
            )
        )
    return reports


def compute_hidden_unit_connectivity_report(
    masked_params: Sequence[MaskedParameter],
) -> list[HiddenUnitConnectivityReport]:
    reports: list[HiddenUnitConnectivityReport] = []
    for left_idx in range(len(masked_params) - 1):
        right_idx = left_idx + 1
        left_mask = masked_params[left_idx].mask
        right_mask = masked_params[right_idx].mask

        if left_mask.ndim != 2 or right_mask.ndim != 2:
            continue
        if left_mask.shape[1] != right_mask.shape[0]:
            continue

        incoming_counts = np.count_nonzero(left_mask, axis=0)
        outgoing_counts = np.count_nonzero(right_mask, axis=1)

        no_incoming = incoming_counts == 0
        no_outgoing = outgoing_counts == 0
        disconnected = no_incoming | no_outgoing

        reports.append(
            HiddenUnitConnectivityReport(
                left_layer_index=left_idx,
                right_layer_index=right_idx,
                hidden_width=int(left_mask.shape[1]),
                no_incoming=int(np.count_nonzero(no_incoming)),
                no_outgoing=int(np.count_nonzero(no_outgoing)),
                disconnected_either_side=int(np.count_nonzero(disconnected)),
            )
        )
    return reports


def compute_magnitude_scores(masked_params: Sequence[MaskedParameter]) -> list[np.ndarray]:
    scores: list[np.ndarray] = []
    for masked_param in masked_params:
        scores.append(np.abs(masked_param.effective_numpy()))
    return scores


def compute_taylor_scores(masked_params: Sequence[MaskedParameter]) -> list[np.ndarray]:
    scores: list[np.ndarray] = []
    for masked_param in masked_params:
        if masked_param.effective_position_grad is None:
            raise ValueError(
                "effective_position_grad is missing; run backward through effective_tensor first"
            )
        grad_eff = np.asarray(masked_param.effective_position_grad, dtype=np.float64)
        if grad_eff.shape != masked_param.shape:
            raise ValueError("effective_position_grad shape mismatch")
        if not np.isfinite(grad_eff).all():
            raise ValueError("effective_position_grad contains non-finite values")
        scores.append(np.abs(masked_param.effective_numpy() * grad_eff))
    return scores


def enforced_target_active_count(total: int, target_sparsity: float) -> int:
    if total <= 0:
        raise ValueError("total must be positive")
    if not (0.0 <= target_sparsity < 1.0):
        raise ValueError("target_sparsity must be in [0, 1)")
    # Floor active count so achieved sparsity is always >= requested target.
    return int(np.floor((1.0 - target_sparsity) * total))


def apply_global_pruning_target(
    masked_params: Sequence[MaskedParameter],
    *,
    layer_scores: Sequence[np.ndarray],
    target_sparsity: float,
    min_survivors_per_layer: int,
    min_survivors_per_row: int = 0,
    min_survivors_per_col: int = 0,
    min_survivors_per_row_by_layer: Optional[Sequence[int]] = None,
    min_survivors_per_col_by_layer: Optional[Sequence[int]] = None,
) -> PruneStepReport:
    """Prune globally to a mask-count target while respecting per-layer survivor guards."""
    if min_survivors_per_layer < 0:
        raise ValueError("min_survivors_per_layer must be non-negative")
    if min_survivors_per_row < 0:
        raise ValueError("min_survivors_per_row must be non-negative")
    if min_survivors_per_col < 0:
        raise ValueError("min_survivors_per_col must be non-negative")
    if not masked_params:
        raise ValueError("masked_params must contain at least one parameter")

    n_layers = len(masked_params)

    if min_survivors_per_row_by_layer is None:
        row_minima = [int(min_survivors_per_row)] * n_layers
    else:
        if len(min_survivors_per_row_by_layer) != n_layers:
            raise ValueError("min_survivors_per_row_by_layer length mismatch")
        row_minima = [int(v) for v in min_survivors_per_row_by_layer]
        if any(v < 0 for v in row_minima):
            raise ValueError("min_survivors_per_row_by_layer must be non-negative")

    if min_survivors_per_col_by_layer is None:
        col_minima = [int(min_survivors_per_col)] * n_layers
    else:
        if len(min_survivors_per_col_by_layer) != n_layers:
            raise ValueError("min_survivors_per_col_by_layer length mismatch")
        col_minima = [int(v) for v in min_survivors_per_col_by_layer]
        if any(v < 0 for v in col_minima):
            raise ValueError("min_survivors_per_col_by_layer must be non-negative")

    _validate_layer_scores(masked_params, layer_scores)

    for layer_idx, masked_param in enumerate(masked_params):
        if (
            row_minima[layer_idx] > 0 or col_minima[layer_idx] > 0
        ) and masked_param.mask.ndim != 2:
            raise ValueError(
                "row/col survivor guards require 2D masked parameters"
            )

    stats_before = compute_mask_sparsity(masked_params)
    total = stats_before.total
    active_before = stats_before.active

    target_active = enforced_target_active_count(total, target_sparsity)
    target_active = max(0, min(target_active, total))
    requested_pruned = max(0, active_before - target_active)

    selectors = [np.zeros_like(masked_param.mask) for masked_param in masked_params]

    candidates: list[tuple[float, int, int, tuple[int, ...]]] = []
    max_prunable = 0
    layer_prunable_limits: list[int] = []
    layer_selected_counts = [0 for _ in masked_params]
    layer_active_remaining = [
        int(np.count_nonzero(masked_param.mask)) for masked_param in masked_params
    ]
    row_active_remaining: list[Optional[np.ndarray]] = []
    col_active_remaining: list[Optional[np.ndarray]] = []

    for layer_idx, (masked_param, score) in enumerate(zip(masked_params, layer_scores)):
        active_indices = np.argwhere(masked_param.mask == 1.0)
        active_count = int(active_indices.shape[0])
        if active_count == 0:
            layer_prunable_limits.append(0)
            row_active_remaining.append(None)
            col_active_remaining.append(None)
            continue

        min_keep = min(min_survivors_per_layer, active_count)
        layer_prunable = max(0, active_count - min_keep)
        layer_prunable_limits.append(layer_prunable)
        max_prunable += layer_prunable

        if masked_param.mask.ndim == 2 and (
            row_minima[layer_idx] > 0 or col_minima[layer_idx] > 0
        ):
            row_active_remaining.append(
                np.count_nonzero(masked_param.mask, axis=1).astype(np.int64)
            )
            col_active_remaining.append(
                np.count_nonzero(masked_param.mask, axis=0).astype(np.int64)
            )
        else:
            row_active_remaining.append(None)
            col_active_remaining.append(None)

        if layer_prunable == 0:
            continue

        for coord_array in active_indices:
            coord = tuple(int(v) for v in coord_array.tolist())
            flat_idx = int(np.ravel_multi_index(coord, masked_param.shape))
            candidates.append((float(score[coord]), layer_idx, flat_idx, coord))

    candidates.sort(key=lambda entry: (entry[0], entry[1], entry[2]))

    planned_pruned = min(requested_pruned, max_prunable)
    selected = 0
    for _, layer_idx, _, coord in candidates:
        if selected >= planned_pruned:
            break
        if selectors[layer_idx][coord] == 1.0:
            continue
        if layer_selected_counts[layer_idx] >= layer_prunable_limits[layer_idx]:
            continue
        if layer_active_remaining[layer_idx] - 1 < min_survivors_per_layer:
            continue

        masked_param = masked_params[layer_idx]
        layer_row_min = row_minima[layer_idx]
        layer_col_min = col_minima[layer_idx]
        if masked_param.mask.ndim == 2 and (layer_row_min > 0 or layer_col_min > 0):
            row_idx = int(coord[0])
            col_idx = int(coord[1])
            row_counts = row_active_remaining[layer_idx]
            col_counts = col_active_remaining[layer_idx]
            assert row_counts is not None
            assert col_counts is not None
            if layer_row_min > 0 and row_counts[row_idx] - 1 < layer_row_min:
                continue
            if layer_col_min > 0 and col_counts[col_idx] - 1 < layer_col_min:
                continue

            if layer_row_min > 0:
                row_counts[row_idx] -= 1
            if layer_col_min > 0:
                col_counts[col_idx] -= 1

        selectors[layer_idx][coord] = 1.0
        layer_selected_counts[layer_idx] += 1
        layer_active_remaining[layer_idx] -= 1
        selected += 1

    applied_pruned = 0
    for masked_param, selector in zip(masked_params, selectors):
        if np.count_nonzero(selector) == 0:
            continue
        transitioned = masked_param.prune(selector)
        applied_pruned += int(np.count_nonzero(transitioned))

    stats_after = compute_mask_sparsity(masked_params)
    active_after = stats_after.active
    enforced_target_sparsity = 1.0 - (target_active / total)
    reached_target = active_after <= target_active
    guard_blocked = selected < planned_pruned

    limitation: Optional[str] = None
    if requested_pruned > max_prunable:
        limitation = (
            "target blocked by per-layer survivor guards "
            f"(requested_pruned={requested_pruned}, max_prunable={max_prunable})"
        )
    elif guard_blocked:
        limitation = (
            "target blocked by row/col survivor guards "
            f"(planned_pruned={planned_pruned}, selected={selected})"
        )
    elif not reached_target:
        limitation = (
            "target not reached due to discrete mask counts "
            f"(target_active={target_active}, active_after={active_after})"
        )

    return PruneStepReport(
        target_sparsity=float(target_sparsity),
        enforced_target_sparsity=float(enforced_target_sparsity),
        active_before=active_before,
        active_after=active_after,
        total=total,
        requested_pruned=requested_pruned,
        applied_pruned=applied_pruned,
        max_prunable_with_guards=max_prunable,
        reached_target=reached_target,
        limitation=limitation,
    )
