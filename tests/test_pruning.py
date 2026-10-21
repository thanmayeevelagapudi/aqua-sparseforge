"""Phase 3 pruning and masked-optimizer behavior tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.nn import MLP, MaskedLinear
from src.ops import cross_entropy
from src.optim import MaskedAdam
from src.pruning import MaskedParameter
from src.tensor import Tensor


def test_masked_linear_forward_uses_effective_weights_only() -> None:
    layer = MaskedLinear(
        in_features=2,
        out_features=1,
        use_bias=False,
        rng=np.random.default_rng(0),
        init_mode="xavier",
    )
    layer.weight.data[...] = np.array([[2.0], [3.0]], dtype=np.float64)
    layer.masked_weight.set_mask(np.array([[1.0], [0.0]], dtype=np.float64))

    x = Tensor(np.array([[4.0, 5.0]], dtype=np.float64))
    y = layer(x)

    # Second input dimension is masked, so only 4 * 2 contributes.
    np.testing.assert_allclose(y.data, np.array([[8.0]], dtype=np.float64))


def test_masked_mlp_predict_logits_numpy_respects_masks() -> None:
    model = MLP(
        in_features=2,
        hidden_sizes=[],
        out_features=1,
        activation="relu",
        masked=True,
        rng=np.random.default_rng(4),
    )

    layer = model.layers[0]
    assert isinstance(layer, MaskedLinear)
    layer.weight.data[...] = np.array([[2.0], [3.0]], dtype=np.float64)
    layer.masked_weight.set_mask(np.array([[1.0], [0.0]], dtype=np.float64))

    logits = model.predict_logits_numpy(np.array([[4.0, 5.0]], dtype=np.float64))
    np.testing.assert_allclose(logits, np.array([[8.0]], dtype=np.float64))


def test_masked_parameter_gradient_matches_effective_chain_rule() -> None:
    rng = np.random.default_rng(7)

    x_data = rng.normal(size=(5, 3)).astype(np.float64)
    w_data = rng.normal(size=(3, 2)).astype(np.float64)
    mask = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

    masked = MaskedParameter(Tensor(w_data, requires_grad=True), mask)
    loss_masked = ((Tensor(x_data) @ masked.effective_tensor()) ** 2.0).mean()
    loss_masked.backward()

    direct_effective = Tensor(w_data * mask, requires_grad=True)
    loss_effective = ((Tensor(x_data) @ direct_effective) ** 2.0).mean()
    loss_effective.backward()

    expected_underlying_grad = mask * direct_effective.grad
    np.testing.assert_allclose(
        masked.tensor.grad,
        expected_underlying_grad,
        atol=1e-10,
        rtol=1e-10,
    )


def test_maskedadam_keeps_inactive_weights_and_grads_zero() -> None:
    masked = MaskedParameter(
        tensor=Tensor(np.array([[0.3], [-0.2]], dtype=np.float64), requires_grad=True),
        mask=np.array([[1.0], [0.0]], dtype=np.float64),
    )
    optimizer = MaskedAdam([masked], lr=0.05)

    # Simulate stale inactive value before update; optimizer should clamp it back to zero.
    masked.tensor.data[1, 0] = 11.0

    for _ in range(3):
        optimizer.zero_grad()
        x = Tensor(np.array([[1.0, 2.0]], dtype=np.float64))
        loss = ((x @ masked.effective_tensor()) - 1.0) ** 2.0
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        np.testing.assert_allclose(masked.tensor.data[1, 0], 0.0, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(masked.tensor.grad[1, 0], 0.0, atol=1e-12, rtol=0.0)


def test_prune_then_revive_resets_state_and_restarts_age_from_one() -> None:
    masked = MaskedParameter(
        tensor=Tensor(np.array([[0.5]], dtype=np.float64), requires_grad=True),
        mask=np.array([[1.0]], dtype=np.float64),
    )
    optimizer = MaskedAdam([masked], lr=0.01, betas=(0.9, 0.999), eps=1e-8)

    for _ in range(5):
        optimizer.zero_grad()
        w_eff = masked.effective_tensor()
        loss = ((w_eff - 1.0) * (w_eff - 1.0)).mean()
        loss.backward()
        optimizer.step()

    state_before_prune = optimizer.state_for(masked)
    assert state_before_prune["age"][0, 0] == 5

    masked.prune(np.array([[1.0]], dtype=np.float64))
    masked.tensor.data[0, 0] = 7.0

    optimizer.zero_grad()
    w_eff_pruned = masked.effective_tensor()
    loss_pruned = ((w_eff_pruned - 1.0) * (w_eff_pruned - 1.0)).mean()
    loss_pruned.backward()
    optimizer.step()

    state_after_prune = optimizer.state_for(masked)
    np.testing.assert_allclose(state_after_prune["m"], np.array([[0.0]]))
    np.testing.assert_allclose(state_after_prune["v"], np.array([[0.0]]))
    np.testing.assert_array_equal(state_after_prune["age"], np.array([[0]], dtype=np.int64))
    np.testing.assert_allclose(masked.tensor.data, np.array([[0.0]]), atol=1e-12, rtol=0.0)

    masked.revive(np.array([[1.0]], dtype=np.float64), init="zeros")
    state_after_revive = optimizer.state_for(masked)
    np.testing.assert_allclose(state_after_revive["m"], np.array([[0.0]]))
    np.testing.assert_allclose(state_after_revive["v"], np.array([[0.0]]))
    np.testing.assert_array_equal(state_after_revive["age"], np.array([[0]], dtype=np.int64))
    np.testing.assert_allclose(masked.tensor.data, np.array([[0.0]]), atol=1e-12, rtol=0.0)

    optimizer.zero_grad()
    w_eff_revived = masked.effective_tensor()
    loss_revived = ((w_eff_revived - 1.0) * (w_eff_revived - 1.0)).mean()
    loss_revived.backward()
    grad = masked.tensor.grad.copy()
    optimizer.step()

    state_after_revival_step = optimizer.state_for(masked)
    np.testing.assert_array_equal(
        state_after_revival_step["age"], np.array([[1]], dtype=np.int64)
    )

    g = grad[0, 0]
    m = (1.0 - 0.9) * g
    v = (1.0 - 0.999) * (g * g)
    m_hat = m / (1.0 - 0.9**1)
    v_hat = v / (1.0 - 0.999**1)
    expected = 0.0 - 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)
    np.testing.assert_allclose(
        masked.tensor.data,
        np.array([[expected]], dtype=np.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def test_effective_position_grad_can_be_recorded_for_inactive_edges() -> None:
    masked = MaskedParameter(
        tensor=Tensor(np.array([[0.2, 0.0]], dtype=np.float64), requires_grad=True),
        mask=np.array([[1.0, 0.0]], dtype=np.float64),
    )
    optimizer = MaskedAdam([masked], lr=0.01)

    optimizer.zero_grad()
    loss = (masked.effective_tensor() ** 2.0).sum()
    loss.backward()

    injected_effective_grad = np.array([[masked.tensor.grad[0, 0], 5.0]], dtype=np.float64)
    optimizer.set_effective_position_grad(masked, injected_effective_grad)
    optimizer.step()

    np.testing.assert_allclose(masked.effective_position_grad, injected_effective_grad)
    np.testing.assert_allclose(masked.tensor.data[0, 1], 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(masked.tensor.grad[0, 1], 0.0, atol=1e-12, rtol=0.0)


def test_revived_active_update_uses_fresh_grad_not_stale_effective_buffer() -> None:
    masked = MaskedParameter(
        tensor=Tensor(np.array([[0.0]], dtype=np.float64), requires_grad=True),
        mask=np.array([[1.0]], dtype=np.float64),
    )
    optimizer = MaskedAdam([masked], lr=0.01, betas=(0.9, 0.999), eps=1e-8)

    masked.prune(np.array([[1.0]], dtype=np.float64))
    masked.revive(np.array([[1.0]], dtype=np.float64), init="zeros")

    # Inject a stale positive value; fresh backward gradient below is negative.
    masked.set_effective_position_grad(np.array([[5.0]], dtype=np.float64))

    optimizer.zero_grad()
    w_eff = masked.effective_tensor()
    loss = ((w_eff - 1.0) * (w_eff - 1.0)).mean()
    loss.backward()
    optimizer.step()

    assert masked.tensor.data[0, 0] > 0.0
    assert masked.effective_position_grad[0, 0] < 0.0


def test_prune_then_revive_without_intervening_step_resets_age() -> None:
    masked = MaskedParameter(
        tensor=Tensor(np.array([[0.5]], dtype=np.float64), requires_grad=True),
        mask=np.array([[1.0]], dtype=np.float64),
    )
    optimizer = MaskedAdam([masked], lr=0.01, betas=(0.9, 0.999), eps=1e-8)

    for _ in range(5):
        optimizer.zero_grad()
        loss = ((masked.effective_tensor() - 1.0) ** 2.0).mean()
        loss.backward()
        optimizer.step()

    assert optimizer.state_for(masked)["age"][0, 0] == 5

    masked.prune(np.array([[1.0]], dtype=np.float64))
    masked.revive(np.array([[1.0]], dtype=np.float64), init="zeros")

    optimizer.zero_grad()
    loss = ((masked.effective_tensor() - 1.0) ** 2.0).mean()
    loss.backward()
    optimizer.step()

    np.testing.assert_array_equal(
        optimizer.state_for(masked)["age"], np.array([[1]], dtype=np.int64)
    )


def test_effective_position_grad_is_computed_for_inactive_connections() -> None:
    x_data = np.array(
        [[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]],
        dtype=np.float64,
    )
    w_data = np.array(
        [[0.8, -0.4], [0.1, 0.2], [-0.7, 0.3]],
        dtype=np.float64,
    )
    mask = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=np.float64,
    )

    masked = MaskedParameter(Tensor(w_data, requires_grad=True), mask)
    masked_loss = ((Tensor(x_data) @ masked.effective_tensor()) ** 2.0).mean()
    masked_loss.backward()

    effective_ref = Tensor(w_data * mask, requires_grad=True)
    ref_loss = ((Tensor(x_data) @ effective_ref) ** 2.0).mean()
    ref_loss.backward()

    assert masked.effective_position_grad is not None
    np.testing.assert_allclose(
        masked.effective_position_grad,
        effective_ref.grad,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        masked.tensor.grad,
        mask * effective_ref.grad,
        atol=1e-10,
        rtol=1e-10,
    )

    inactive_values = np.abs(masked.effective_position_grad[mask == 0.0])
    assert np.any(inactive_values > 1e-12)


def test_effective_position_grad_refreshes_after_pruning() -> None:
    masked = MaskedParameter(
        tensor=Tensor(np.array([[0.5], [-0.25]], dtype=np.float64), requires_grad=True),
        mask=np.array([[1.0], [1.0]], dtype=np.float64),
    )
    optimizer = MaskedAdam([masked], lr=0.01)

    optimizer.zero_grad()
    loss_a = ((Tensor(np.array([[1.0, 2.0]], dtype=np.float64)) @ masked.effective_tensor()) - 1.0) ** 2.0
    loss_a = loss_a.mean()
    loss_a.backward()
    first_effective_grad = masked.effective_position_grad.copy()
    optimizer.step()

    masked.prune(np.array([[0.0], [1.0]], dtype=np.float64))

    optimizer.zero_grad()
    loss_b = ((Tensor(np.array([[3.0, -1.0]], dtype=np.float64)) @ masked.effective_tensor()) + 1.0) ** 2.0
    loss_b = loss_b.mean()
    loss_b.backward()
    second_effective_grad = masked.effective_position_grad.copy()

    effective_ref = Tensor(masked.effective_numpy(), requires_grad=True)
    ref_loss_b = ((Tensor(np.array([[3.0, -1.0]], dtype=np.float64)) @ effective_ref) + 1.0) ** 2.0
    ref_loss_b = ref_loss_b.mean()
    ref_loss_b.backward()

    optimizer.step()

    assert not np.isclose(first_effective_grad[1, 0], second_effective_grad[1, 0])
    np.testing.assert_allclose(
        second_effective_grad,
        effective_ref.grad,
        atol=1e-10,
        rtol=1e-10,
    )


def test_maskedadam_rejects_double_managed_dense_tensor() -> None:
    masked = MaskedParameter(
        tensor=Tensor(np.array([[1.0]], dtype=np.float64), requires_grad=True),
        mask=np.array([[1.0]], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="already managed by masked_params"):
        MaskedAdam([masked], dense_params=[masked.tensor], lr=1e-3)


def test_maskedadam_step_does_not_overwrite_raw_tensor_grad() -> None:
    masked = MaskedParameter(
        tensor=Tensor(np.array([[0.3], [0.1]], dtype=np.float64), requires_grad=True),
        mask=np.array([[1.0], [0.0]], dtype=np.float64),
    )
    optimizer = MaskedAdam([masked], lr=0.05)

    optimizer.zero_grad()
    loss = ((Tensor(np.array([[2.0, -1.0]], dtype=np.float64)) @ masked.effective_tensor()) - 0.25) ** 2.0
    loss = loss.mean()
    loss.backward()

    raw_grad = masked.tensor.grad.copy()
    optimizer.step()

    np.testing.assert_allclose(masked.tensor.grad, raw_grad, atol=0.0, rtol=0.0)


def test_masked_mlp_training_keeps_pruned_weights_exactly_zero() -> None:
    rng = np.random.default_rng(123)
    model = MLP(
        in_features=4,
        hidden_sizes=[6],
        out_features=3,
        activation="relu",
        masked=True,
        rng=rng,
    )

    masked_params = model.masked_parameters()
    assert masked_params

    first_masked = masked_params[0]
    prune_selector = np.zeros_like(first_masked.mask)
    prune_selector[::2, ::2] = 1.0
    first_masked.prune(prune_selector)
    pruned_entries = prune_selector.astype(bool)

    dense_params = [layer.bias for layer in model.layers if layer.bias is not None]
    optimizer = MaskedAdam(masked_params=masked_params, dense_params=dense_params, lr=1e-2)

    x = rng.normal(size=(24, 4)).astype(np.float64)
    y = rng.integers(0, 3, size=(24,), endpoint=False, dtype=np.int64)

    for _ in range(6):
        optimizer.zero_grad()
        logits = model(Tensor(x, requires_grad=False))
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        np.testing.assert_allclose(
            first_masked.tensor.data[pruned_entries],
            0.0,
            atol=1e-12,
            rtol=0.0,
        )
