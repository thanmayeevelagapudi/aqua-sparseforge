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


def test_adam_matches_manual_formula_for_two_steps() -> None:
    param = Tensor(np.array([1.0], dtype=np.float64), requires_grad=True)
    optimizer = Adam([param], lr=0.01, betas=(0.9, 0.999), eps=1e-8)

    # Step 1
    g1 = np.array([0.2], dtype=np.float64)
    param.grad = g1.copy()
    optimizer.step()

    m1 = 0.1 * g1
    v1 = 0.001 * (g1 * g1)
    m1_hat = m1 / (1.0 - 0.9)
    v1_hat = v1 / (1.0 - 0.999)
    expected_after_step1 = (1.0 - 0.01 * m1_hat / (np.sqrt(v1_hat) + 1e-8)).item()
    np.testing.assert_allclose(
        param.data, np.array([expected_after_step1]), atol=1e-12, rtol=1e-12
    )

    # Step 2
    g2 = np.array([-0.4], dtype=np.float64)
    param.grad = g2.copy()
    optimizer.step()

    m2 = 0.9 * m1 + 0.1 * g2
    v2 = 0.999 * v1 + 0.001 * (g2 * g2)
    m2_hat = m2 / (1.0 - 0.9**2)
    v2_hat = v2 / (1.0 - 0.999**2)
    expected_after_step2 = (
        expected_after_step1 - 0.01 * m2_hat / (np.sqrt(v2_hat) + 1e-8)
    ).item()
    np.testing.assert_allclose(
        param.data, np.array([expected_after_step2]), atol=1e-12, rtol=1e-12
    )


def test_adam_converges_on_simple_quadratic() -> None:
    x = Tensor(np.array([5.0], dtype=np.float64), requires_grad=True)
    optimizer = Adam([x], lr=0.1)

    for _ in range(400):
        optimizer.zero_grad()
        # f(x) = (x - 3)^2
        loss = ((x - 3.0) * (x - 3.0)).mean()
        loss.backward()
        optimizer.step()

    assert abs(float(x.data[0]) - 3.0) < 1e-2
