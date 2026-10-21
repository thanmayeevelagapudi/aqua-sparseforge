"""Unit tests for the Adam optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from src.optim import Adam
from src.tensor import Tensor


def test_adam_first_step_matches_manual_formula() -> None:
    param = Tensor(np.array([1.0, -2.0], dtype=np.float64), requires_grad=True)
    param.grad = np.array([0.5, -0.25], dtype=np.float64)

    optimizer = Adam([param], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    original = param.data.copy()
    grad = param.grad.copy()

    optimizer.step()

    m = (1.0 - 0.9) * grad
    v = (1.0 - 0.999) * (grad * grad)
    m_hat = m / (1.0 - 0.9**1)
    v_hat = v / (1.0 - 0.999**1)
    expected = original - 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)

    np.testing.assert_allclose(param.data, expected, rtol=1e-12, atol=1e-12)


def test_adam_zero_grad_clears_gradients() -> None:
    param1 = Tensor(np.array([1.0, 2.0], dtype=np.float64), requires_grad=True)
    param2 = Tensor(np.array([[1.0], [2.0]], dtype=np.float64), requires_grad=True)
    param1.grad = np.array([3.0, -4.0], dtype=np.float64)
    param2.grad = np.array([[5.0], [6.0]], dtype=np.float64)

    optimizer = Adam([param1, param2], lr=1e-3)
    optimizer.zero_grad()

    np.testing.assert_allclose(param1.grad, np.zeros_like(param1.data))
    np.testing.assert_allclose(param2.grad, np.zeros_like(param2.data))


def test_adam_rejects_non_finite_gradients() -> None:
    param = Tensor(np.array([1.0], dtype=np.float64), requires_grad=True)
    param.grad = np.array([np.nan], dtype=np.float64)
    optimizer = Adam([param], lr=1e-3)

    with pytest.raises(ValueError, match="non-finite gradient"):
        optimizer.step()
