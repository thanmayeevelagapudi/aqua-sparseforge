"""Masking helpers used by Phase 1/3 correctness tests and later pruning flows."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .tensor import Tensor


def validate_binary_mask(mask: np.ndarray, expected_shape: Tuple[int, ...]) -> np.ndarray:
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


def effective_weight(weight: Tensor, mask: np.ndarray) -> Tensor:
	"""Apply a fixed binary mask to a weight tensor.

	The returned tensor defines the effective parameter semantics:
	W_eff = M * W
	"""
	mask_array = validate_binary_mask(mask, weight.shape)
	return weight * Tensor(mask_array)

