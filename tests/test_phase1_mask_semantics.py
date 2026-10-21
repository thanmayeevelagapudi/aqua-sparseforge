"""Phase 1 masked-weight correctness tests required by challenge criteria."""

from __future__ import annotations

import numpy as np
import pytest

from src.pruning import effective_weight
from src.tensor import Tensor


def test_masked_entries_do_not_affect_forward_output() -> None:
    rng = np.random.default_rng(123)

    x_data = rng.normal(size=(5, 4)).astype(np.float64)
    w_base = rng.normal(size=(4, 3)).astype(np.float64)
    mask = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    w_mutated = w_base.copy()
    w_mutated[mask == 0.0] = rng.normal(
        loc=0.0, scale=1000.0, size=int((mask == 0.0).sum())
    )

    x1 = Tensor(x_data)
    y1 = x1 @ effective_weight(Tensor(w_base, requires_grad=False), mask)

    x2 = Tensor(x_data)
    y2 = x2 @ effective_weight(Tensor(w_mutated, requires_grad=False), mask)

    np.testing.assert_allclose(y1.data, y2.data, atol=1e-12, rtol=0.0)


def test_masked_gradient_matches_chain_rule_for_underlying_weights() -> None:
    rng = np.random.default_rng(456)

    x_data = rng.normal(size=(6, 4)).astype(np.float64)
    w_data = rng.normal(size=(4, 3)).astype(np.float64)
    mask = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    x_masked = Tensor(x_data)
    w_underlying = Tensor(w_data, requires_grad=True)
    w_eff = effective_weight(w_underlying, mask)
    loss_masked = ((x_masked @ w_eff) ** 2.0).mean()
    loss_masked.backward()

    x_effective = Tensor(x_data)
    w_direct_eff = Tensor(w_data * mask, requires_grad=True)
    loss_effective = ((x_effective @ w_direct_eff) ** 2.0).mean()
    loss_effective.backward()

    expected_grad = mask * w_direct_eff.grad
    np.testing.assert_allclose(w_underlying.grad, expected_grad, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        w_underlying.grad[mask == 0.0], 0.0, atol=1e-12, rtol=0.0
    )


def test_effective_weight_rejects_non_binary_or_shape_mismatch_masks() -> None:
    weight = Tensor(np.ones((2, 3), dtype=np.float64), requires_grad=True)

    with pytest.raises(ValueError, match="does not match expected shape"):
        effective_weight(weight, np.ones((3, 2), dtype=np.float64))

    with pytest.raises(ValueError, match="mask must be binary"):
        effective_weight(
            weight, np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 1.0]], dtype=np.float64)
        )
