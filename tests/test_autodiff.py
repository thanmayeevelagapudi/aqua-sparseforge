"""Core regression tests for the Tensor autodiff engine."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.ops import cross_entropy
from src.tensor import Tensor


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def _upstream_for_shape(shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) == 0:
        return np.array(1.25, dtype=np.float64)
    values = np.linspace(-1.3, 1.7, int(np.prod(shape)), dtype=np.float64)
    return values.reshape(shape)


def test_shared_parent_accumulation() -> None:
    x = Tensor(3.0, requires_grad=True)
    y = x * x + x
    y.backward()
    assert np.isclose(float(x.grad), 7.0)


def test_repeated_parent_reference_accumulates_twice() -> None:
    x = Tensor(np.array([1.0, -2.0], dtype=np.float64), requires_grad=True)
    y = x + x
    y.backward(np.array([2.0, -3.0], dtype=np.float64))

    np.testing.assert_allclose(x.grad, np.array([4.0, -6.0], dtype=np.float64))


def test_parent_container_is_ordered_and_deduplicated() -> None:
    a = Tensor(1.0, requires_grad=True)
    b = Tensor(2.0, requires_grad=True)

    c = a + b
    d = a + a

    assert isinstance(c._prev, tuple)
    assert c._prev == (a, b)
    assert d._prev == (a,)


def test_broadcast_add_backward_for_bias() -> None:
    n, d = 4, 3
    a = Tensor(np.arange(n * d, dtype=np.float64).reshape(n, d), requires_grad=True)
    b = Tensor(np.array([0.1, -0.2, 0.3], dtype=np.float64), requires_grad=True)

    z = a + b
    loss = z.sum()
    loss.backward()

    np.testing.assert_allclose(a.grad, np.ones((n, d), dtype=np.float64))
    np.testing.assert_allclose(b.grad, np.full((d,), n, dtype=np.float64))


def test_broadcast_subtraction_backward_column_vector() -> None:
    a = Tensor(np.arange(12, dtype=np.float64).reshape(4, 3), requires_grad=True)
    b = Tensor(np.arange(4, dtype=np.float64).reshape(4, 1), requires_grad=True)

    loss = (a - b).sum()
    loss.backward()

    np.testing.assert_allclose(a.grad, np.ones((4, 3), dtype=np.float64))
    np.testing.assert_allclose(b.grad, -np.full((4, 1), 3.0, dtype=np.float64))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (
            np.array([1.0, -2.0, 3.0], dtype=np.float64),
            np.array([0.5, 2.0, -1.5], dtype=np.float64),
        ),
        (
            np.array([1.0, -2.0, 3.0], dtype=np.float64),
            np.array(
                [[0.1, -0.2, 0.3], [1.2, 0.8, -0.4], [-0.6, 0.7, 0.9]],
                dtype=np.float64,
            ),
        ),
        (
            np.array([[0.5, -1.2, 2.3], [1.1, 0.4, -0.7]], dtype=np.float64),
            np.array([0.25, -1.5, 2.0], dtype=np.float64),
        ),
        (
            np.array([[0.3, -0.4, 1.2], [1.1, 0.9, -0.8]], dtype=np.float64),
            np.array(
                [[0.7, -1.0, 0.5, 1.2], [1.3, 0.2, -0.9, 0.4], [0.8, -1.1, 0.6, -0.3]],
                dtype=np.float64,
            ),
        ),
        (
            np.arange(24.0, dtype=np.float64).reshape(2, 3, 4) / 10.0,
            np.arange(40.0, dtype=np.float64).reshape(2, 4, 5) / 10.0,
        ),
        (
            np.arange(24.0, dtype=np.float64).reshape(2, 3, 4) / 7.0,
            np.arange(20.0, dtype=np.float64).reshape(4, 5) / 9.0,
        ),
        (
            np.arange(12.0, dtype=np.float64).reshape(3, 4) / 5.0,
            np.arange(40.0, dtype=np.float64).reshape(2, 4, 5) / 11.0,
        ),
    ],
)
def test_matmul_backward_supported_shapes_have_finite_original_shape_grads(
    left: np.ndarray,
    right: np.ndarray,
) -> None:
    a = Tensor(left, requires_grad=True)
    b = Tensor(right, requires_grad=True)
    out = a @ b

    upstream = _upstream_for_shape(out.shape)
    out.backward(upstream)

    assert a.grad.shape == left.shape
    assert b.grad.shape == right.shape
    assert np.isfinite(a.grad).all()
    assert np.isfinite(b.grad).all()


def test_matmul_rejects_scalar_operands_with_clear_error() -> None:
    a = Tensor(np.array(2.0, dtype=np.float64), requires_grad=True)
    b = Tensor(np.array([1.0, 2.0], dtype=np.float64), requires_grad=True)

    with pytest.raises(ValueError, match="at least 1 dimension"):
        _ = a @ b


def test_cross_entropy_confident_wrong_extreme_logits() -> None:
    logits = Tensor(np.array([[1000.0, 0.0]], dtype=np.float64), requires_grad=True)
    targets = np.array([1], dtype=np.int64)

    loss = cross_entropy(logits, targets)
    loss.backward()

    assert np.isclose(loss.item(), 1000.0, atol=1e-8)
    np.testing.assert_allclose(logits.grad, np.array([[1.0, -1.0]], dtype=np.float64), atol=1e-12)


def test_cross_entropy_confident_wrong_large_logits() -> None:
    logits = Tensor(np.array([[50.0, 0.0]], dtype=np.float64), requires_grad=True)
    targets = np.array([1], dtype=np.int64)

    loss = cross_entropy(logits, targets)
    loss.backward()

    assert np.isclose(loss.item(), 50.0, atol=1e-8)
    np.testing.assert_allclose(logits.grad, np.array([[1.0, -1.0]], dtype=np.float64), atol=1e-8)


def test_cross_entropy_correct_extreme_logits_is_nonnegative_and_near_zero() -> None:
    logits = Tensor(np.array([[1000.0, 0.0]], dtype=np.float64), requires_grad=True)
    targets = np.array([0], dtype=np.int64)

    loss = cross_entropy(logits, targets)
    loss.backward()

    assert loss.item() >= -1e-12
    assert loss.item() <= 1e-12
    np.testing.assert_allclose(logits.grad, np.array([[0.0, 0.0]], dtype=np.float64), atol=1e-12)


def test_cross_entropy_batch_average_scaling_and_gradient() -> None:
    logits_data = np.array(
        [[2.0, 0.0, -1.0], [1.5, -0.5, 0.2]],
        dtype=np.float64,
    )
    targets = np.array([0, 2], dtype=np.int64)

    logits = Tensor(logits_data, requires_grad=True)
    loss = cross_entropy(logits, targets)
    loss.backward()

    probs = _softmax_numpy(logits_data)
    one_hot = np.zeros_like(logits_data)
    one_hot[np.arange(logits_data.shape[0]), targets] = 1.0
    expected_grad = (probs - one_hot) / logits_data.shape[0]

    np.testing.assert_allclose(logits.grad, expected_grad, atol=1e-10, rtol=1e-10)


def test_cross_entropy_rejects_invalid_inputs() -> None:
    logits = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), requires_grad=True)

    with pytest.raises(TypeError, match="integer array"):
        cross_entropy(logits, np.array([0.0, 1.0], dtype=np.float64))

    with pytest.raises(ValueError, match=r"shape \(batch,\)"):
        cross_entropy(logits, np.array([[0], [1]], dtype=np.int64))

    with pytest.raises(ValueError, match="class indices"):
        cross_entropy(logits, np.array([0, 2], dtype=np.int64))

    empty_logits = Tensor(np.empty((0, 3), dtype=np.float64), requires_grad=True)
    with pytest.raises(ValueError, match="empty batches"):
        cross_entropy(empty_logits, np.empty((0,), dtype=np.int64))


def test_backward_accumulates_leaf_gradients_until_zero_grad() -> None:
    w = Tensor(2.0, requires_grad=True)

    (w * 3.0).backward()
    assert np.isclose(float(w.grad), 3.0)

    (w * 5.0).backward()
    assert np.isclose(float(w.grad), 8.0)

    w.zero_grad()
    assert np.isclose(float(w.grad), 0.0)


def test_backward_twice_on_same_graph_does_not_reuse_stale_intermediate_gradients() -> None:
    x = Tensor(2.0, requires_grad=True)
    y = x * x

    y.backward()
    y.backward()

    assert np.isclose(float(x.grad), 8.0)


def test_backward_respects_explicit_upstream_gradient() -> None:
    x = Tensor(np.array([1.0, -2.0], dtype=np.float64), requires_grad=True)
    y = x * 3.0

    upstream = np.array([2.0, -4.0], dtype=np.float64)
    y.backward(upstream)

    np.testing.assert_allclose(x.grad, np.array([6.0, -12.0], dtype=np.float64))


def test_backward_requires_grad_for_non_scalar_output() -> None:
    x = Tensor(np.array([1.0, 2.0], dtype=np.float64), requires_grad=True)
    y = x * 2.0

    with pytest.raises(ValueError, match="grad must be provided"):
        y.backward()


def test_backward_rejects_wrong_upstream_shape() -> None:
    x = Tensor(np.array([1.0, 2.0], dtype=np.float64), requires_grad=True)
    y = x * 2.0

    with pytest.raises(ValueError, match="does not match output shape"):
        y.backward(np.array([[1.0, 1.0]], dtype=np.float64))


def test_sigmoid_extremes_no_overflow_warning_and_finite_gradient() -> None:
    x = Tensor(np.array([-1000.0, 0.0, 1000.0], dtype=np.float64), requires_grad=True)

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        y = x.sigmoid()

    np.testing.assert_allclose(y.data, np.array([0.0, 0.5, 1.0], dtype=np.float64), atol=0.0, rtol=0.0)

    y.backward(np.array([1.0, 1.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(
        x.grad,
        np.array([0.0, 0.25, 0.0], dtype=np.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_tensor_construction_does_not_alias_input_numpy_array() -> None:
    arr = np.array([1.0, 2.0], dtype=np.float64)
    tensor = Tensor(arr, requires_grad=True)

    arr[0] = 99.0
    assert np.isclose(tensor.data[0], 1.0)

    tensor.data[1] = -42.0
    assert np.isclose(arr[1], 2.0)


def test_deep_graph_backward_avoids_recursion_limit() -> None:
    x = Tensor(1.0, requires_grad=True)
    y = x

    for _ in range(3500):
        y = y * 1.0001 + 0.0001

    y.backward()

    assert np.isfinite(y.item())
    assert np.isfinite(float(x.grad))


def test_reproducibility_across_processes_for_identical_graph() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = """
import json
import numpy as np
from src.tensor import Tensor

x = Tensor(np.array(1.0, dtype=np.float64), requires_grad=True)
y = x * 0.0
for i in range(1, 500):
    coeff = ((-1.0) ** i) * (10.0 ** (i % 6))
    y = y + (x * coeff)

y.backward()
print(json.dumps(float(x.grad)))
"""

    run1 = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    run2 = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    grad1 = json.loads(run1.stdout.strip())
    grad2 = json.loads(run2.stdout.strip())
    assert grad1 == grad2
