"""Finite-difference gradient checks for supported differentiable ops."""

from __future__ import annotations

import numpy as np
import pytest

from src.gradcheck import assert_gradients_close
from src.ops import cross_entropy


def _weights(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(shape) == 0:
        return np.array(1.7, dtype=np.float64)

    weights = rng.normal(size=shape).astype(np.float64)
    offsets = np.linspace(-0.35, 0.35, weights.size, dtype=np.float64).reshape(shape)
    return weights + offsets


def test_grad_unary_exp_weighted() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3)).astype(np.float64)
    w = _weights((2, 3), seed=101)

    assert_gradients_close(lambda a: a.exp(), [x], output_weights=w)


def test_grad_unary_log_weighted() -> None:
    rng = np.random.default_rng(2)
    x = (np.abs(rng.normal(size=(3, 2))) + 0.5).astype(np.float64)
    w = _weights((3, 2), seed=102)

    assert_gradients_close(lambda a: a.log(), [x], output_weights=w)


def test_grad_unary_pow_weighted() -> None:
    rng = np.random.default_rng(3)
    x = (np.abs(rng.normal(size=(2, 2))) + 0.5).astype(np.float64)
    w = _weights((2, 2), seed=103)

    assert_gradients_close(lambda a: a ** 2.5, [x], output_weights=w)


def test_grad_activation_tanh_weighted() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 4)).astype(np.float64)
    w = _weights((2, 4), seed=104)

    assert_gradients_close(lambda a: a.tanh(), [x], output_weights=w)


def test_grad_activation_sigmoid_weighted() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(3, 2)).astype(np.float64)
    w = _weights((3, 2), seed=105)

    assert_gradients_close(lambda a: a.sigmoid(), [x], output_weights=w)


def test_grad_activation_relu_weighted_away_from_kinks() -> None:
    x = np.array([[-1.5, -0.25, 0.75], [1.25, 2.0, -3.0]], dtype=np.float64)
    w = _weights((2, 3), seed=106)

    assert_gradients_close(lambda a: a.relu(), [x], output_weights=w, atol=1e-6, rtol=1e-6)


def test_grad_add_broadcast_weighted() -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(size=(3, 4)).astype(np.float64)
    y = rng.normal(size=(4,)).astype(np.float64)
    w = _weights((3, 4), seed=107)

    assert_gradients_close(lambda a, b: a + b, [x, y], output_weights=w)


def test_grad_sub_broadcast_weighted() -> None:
    rng = np.random.default_rng(16)
    x = rng.normal(size=(4, 3)).astype(np.float64)
    y = rng.normal(size=(4, 1)).astype(np.float64)
    w = _weights((4, 3), seed=116)

    assert_gradients_close(lambda a, b: a - b, [x, y], output_weights=w)


def test_grad_binary_multiply_weighted() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 3)).astype(np.float64)
    y = rng.normal(size=(3, 3)).astype(np.float64)
    w = _weights((3, 3), seed=108)

    assert_gradients_close(lambda a, b: a * b, [x, y], output_weights=w)


def test_grad_binary_divide_weighted() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(2, 3)).astype(np.float64)
    y = (rng.normal(size=(2, 3)) + 2.5).astype(np.float64)
    w = _weights((2, 3), seed=109)

    assert_gradients_close(lambda a, b: a / b, [x, y], output_weights=w)


def test_grad_sum_axis_weighted() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(3, 4, 2)).astype(np.float64)
    w = _weights((3, 2), seed=110)

    assert_gradients_close(lambda a: a.sum(axis=1), [x], output_weights=w)


def test_grad_mean_axis_weighted() -> None:
    rng = np.random.default_rng(10)
    x = rng.normal(size=(2, 3, 4)).astype(np.float64)
    w = _weights((3,), seed=111)

    assert_gradients_close(lambda a: a.mean(axis=(0, 2)), [x], output_weights=w)


def test_grad_reshape_weighted() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(2, 3, 4)).astype(np.float64)
    w = _weights((3, 8), seed=117)

    assert_gradients_close(lambda a: a.reshape(3, 8), [x], output_weights=w)


def test_grad_transpose_weighted() -> None:
    rng = np.random.default_rng(18)
    x = rng.normal(size=(2, 3, 4)).astype(np.float64)
    w = _weights((4, 2, 3), seed=118)

    assert_gradients_close(lambda a: a.transpose((2, 0, 1)), [x], output_weights=w)


@pytest.mark.parametrize(
    ("left", "right", "weights_seed"),
    [
        (np.array([0.5, -1.0, 2.0], dtype=np.float64), np.array([1.0, 3.0, -2.0], dtype=np.float64), None),
        (
            np.array([1.5, -2.0, 0.3], dtype=np.float64),
            np.array(
                [[0.2, -0.5, 1.0], [1.1, 0.7, -1.4], [0.3, 0.9, -0.8]],
                dtype=np.float64,
            ),
            201,
        ),
        (
            np.array([[0.5, -1.2, 2.3], [1.1, 0.4, -0.7]], dtype=np.float64),
            np.array([0.25, -1.5, 2.0], dtype=np.float64),
            202,
        ),
        (
            np.array([[0.3, -0.4, 1.2], [1.1, 0.9, -0.8]], dtype=np.float64),
            np.array(
                [[0.7, -1.0, 0.5, 1.2], [1.3, 0.2, -0.9, 0.4], [0.8, -1.1, 0.6, -0.3]],
                dtype=np.float64,
            ),
            203,
        ),
        (
            np.arange(24.0, dtype=np.float64).reshape(2, 3, 4) / 10.0,
            np.arange(40.0, dtype=np.float64).reshape(2, 4, 5) / 10.0,
            204,
        ),
        (
            np.arange(24.0, dtype=np.float64).reshape(2, 3, 4) / 7.0,
            np.arange(20.0, dtype=np.float64).reshape(4, 5) / 9.0,
            205,
        ),
        (
            np.arange(12.0, dtype=np.float64).reshape(3, 4) / 5.0,
            np.arange(40.0, dtype=np.float64).reshape(2, 4, 5) / 11.0,
            206,
        ),
    ],
)
def test_grad_matmul_weighted(
    left: np.ndarray,
    right: np.ndarray,
    weights_seed: int | None,
) -> None:
    output_shape = np.matmul(left, right).shape
    output_weights = None if weights_seed is None else _weights(output_shape, seed=weights_seed)

    assert_gradients_close(
        lambda a, b: a @ b,
        [left, right],
        output_weights=output_weights,
        atol=2e-6,
        rtol=2e-4,
    )


def test_grad_softmax_cross_entropy() -> None:
    logits = np.array(
        [[0.25, -0.1, 0.9], [0.5, 1.3, -0.7]],
        dtype=np.float64,
    )
    targets = np.array([2, 1], dtype=np.int64)

    assert_gradients_close(
        lambda l: cross_entropy(l, targets),
        [logits],
        atol=5e-5,
        rtol=5e-4,
    )
