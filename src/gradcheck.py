"""Finite-difference helpers for validating Tensor gradients."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np

from .tensor import Tensor


ArrayFn = Callable[..., Tensor]


def finite_difference_gradient(
    scalar_fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Numerically approximate d(scalar_fn)/dx using centered differences."""
    grad = np.zeros_like(x, dtype=np.float64)
    iterator = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not iterator.finished:
        idx = iterator.multi_index
        x_pos = x.copy()
        x_neg = x.copy()
        x_pos[idx] += eps
        x_neg[idx] -= eps
        grad[idx] = (scalar_fn(x_pos) - scalar_fn(x_neg)) / (2.0 * eps)
        iterator.iternext()

    return grad


def _to_scalar_output(
    output: Tensor,
    output_weights: Optional[np.ndarray],
) -> Tensor:
    if output_weights is None:
        if output.data.size != 1:
            raise ValueError(
                "output_weights must be provided when the function output is non-scalar"
            )
        return output

    if output_weights.shape != output.data.shape:
        raise ValueError(
            f"output_weights shape {output_weights.shape} does not match output shape {output.data.shape}"
        )

    return (output * Tensor(output_weights)).sum()


def assert_gradients_close(
    fn: ArrayFn,
    inputs: Sequence[np.ndarray],
    *,
    output_weights: Optional[np.ndarray] = None,
    eps: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> None:
    """Compare autodiff gradients against centered finite differences.

    For non-scalar function outputs, provide ``output_weights`` so the checked
    scalar objective is ``sum(fn(inputs) * output_weights)``.
    """
    input_copies = [np.array(array, dtype=np.float64, copy=True) for array in inputs]
    tensors = [Tensor(array.copy(), requires_grad=True) for array in input_copies]

    autodiff_output = fn(*tensors)
    _to_scalar_output(autodiff_output, output_weights).backward()

    for input_index, x in enumerate(input_copies):

        def scalar_fn(local_x: np.ndarray) -> float:
            local_inputs: List[np.ndarray] = [arr.copy() for arr in input_copies]
            local_inputs[input_index] = local_x
            local_tensors = [Tensor(arr, requires_grad=False) for arr in local_inputs]
            local_output = fn(*local_tensors)
            scalar_tensor = _to_scalar_output(local_output, output_weights)
            return float(scalar_tensor.data)

        numerical = finite_difference_gradient(scalar_fn, x, eps=eps)
        np.testing.assert_allclose(tensors[input_index].grad, numerical, atol=atol, rtol=rtol)

