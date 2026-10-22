"""Sparse-aware inference correctness and accounting tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.nn import MLP, MaskedLinear


def test_sparse_predict_matches_masked_dense_forward() -> None:
    model = MLP(
        in_features=4,
        hidden_sizes=[5],
        out_features=3,
        activation="relu",
        masked=True,
        rng=np.random.default_rng(7),
    )

    masked_layers = [layer for layer in model.layers if isinstance(layer, MaskedLinear)]
    assert len(masked_layers) == 2

    masked_layers[0].masked_weight.set_mask(
        np.array(
            [
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
            ],
            dtype=np.float64,
        )
    )
    masked_layers[1].masked_weight.set_mask(
        np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [0, 1, 1],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
    )

    x = np.random.default_rng(11).normal(size=(9, 4)).astype(np.float64)
    dense_logits = model.predict_logits_numpy(x)
    sparse_logits = model.predict_logits_numpy_sparse(x)

    np.testing.assert_allclose(sparse_logits, dense_logits, atol=1e-10, rtol=0.0)


def test_sparse_predict_matches_dense_for_unmasked_model() -> None:
    model = MLP(
        in_features=4,
        hidden_sizes=[6],
        out_features=3,
        activation="tanh",
        masked=False,
        rng=np.random.default_rng(13),
    )

    x = np.random.default_rng(17).normal(size=(7, 4)).astype(np.float64)
    dense_logits = model.predict_logits_numpy(x)
    sparse_logits = model.predict_logits_numpy_sparse(x)

    np.testing.assert_allclose(sparse_logits, dense_logits, atol=1e-10, rtol=0.0)


def test_sparse_inference_cost_reflects_active_connections() -> None:
    model = MLP(
        in_features=3,
        hidden_sizes=[2],
        out_features=2,
        activation="relu",
        masked=True,
        rng=np.random.default_rng(23),
    )

    masked_layers = [layer for layer in model.layers if isinstance(layer, MaskedLinear)]
    assert len(masked_layers) == 2

    masked_layers[0].masked_weight.set_mask(
        np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
            ],
            dtype=np.float64,
        )
    )
    masked_layers[1].masked_weight.set_mask(
        np.array(
            [
                [1, 0],
                [0, 1],
            ],
            dtype=np.float64,
        )
    )

    dense_cost = model.estimate_inference_cost(batch_size=5, sparse=False)
    sparse_cost = model.estimate_inference_cost(batch_size=5, sparse=True)

    assert dense_cost["total_weights"] == 10
    assert sparse_cost["active_weights"] == 6
    assert dense_cost["linear_mac"] == 50
    assert sparse_cost["linear_mac"] == 30
    assert dense_cost["linear_flops"] == 120
    assert sparse_cost["linear_flops"] == 80


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
def test_sparse_predict_matches_masked_dense_for_batch_size_one(activation: str) -> None:
    model = MLP(
        in_features=6,
        hidden_sizes=[7],
        out_features=4,
        activation=activation,
        masked=True,
        rng=np.random.default_rng(31),
    )

    masked_layers = [layer for layer in model.layers if isinstance(layer, MaskedLinear)]
    assert len(masked_layers) == 2

    rng = np.random.default_rng(32)
    for layer in masked_layers:
        mask = (rng.random(layer.weight.data.shape) < 0.55).astype(np.float64)
        if np.count_nonzero(mask) == 0:
            mask.flat[0] = 1.0
        layer.masked_weight.set_mask(mask)

    x = np.random.default_rng(33).normal(size=(1, 6)).astype(np.float64)
    dense_logits = model.predict_logits_numpy(x)
    sparse_logits = model.predict_logits_numpy_sparse(x)

    np.testing.assert_allclose(sparse_logits, dense_logits, atol=1e-10, rtol=0.0)


def test_sparse_predict_handles_all_pruned_hidden_layer() -> None:
    model = MLP(
        in_features=5,
        hidden_sizes=[6],
        out_features=3,
        activation="relu",
        masked=True,
        rng=np.random.default_rng(41),
    )

    masked_layers = [layer for layer in model.layers if isinstance(layer, MaskedLinear)]
    assert len(masked_layers) == 2

    masked_layers[0].masked_weight.set_mask(np.zeros_like(masked_layers[0].weight.data, dtype=np.float64))

    output_mask = np.zeros_like(masked_layers[1].weight.data, dtype=np.float64)
    output_mask[0, 0] = 1.0
    output_mask[1, 1] = 1.0
    output_mask[2, 2] = 1.0
    masked_layers[1].masked_weight.set_mask(output_mask)

    x = np.random.default_rng(42).normal(size=(4, 5)).astype(np.float64)
    dense_logits = model.predict_logits_numpy(x)
    sparse_logits = model.predict_logits_numpy_sparse(x)

    np.testing.assert_allclose(sparse_logits, dense_logits, atol=1e-10, rtol=0.0)


def test_sparse_predict_matches_dense_for_trained_size_high_sparsity() -> None:
    model = MLP(
        in_features=64,
        hidden_sizes=[256],
        out_features=10,
        activation="relu",
        masked=True,
        rng=np.random.default_rng(51),
    )

    masked_layers = [layer for layer in model.layers if isinstance(layer, MaskedLinear)]
    assert len(masked_layers) == 2

    rng = np.random.default_rng(52)
    keep_prob = 0.05
    for idx, layer in enumerate(masked_layers):
        mask = (rng.random(layer.weight.data.shape) < keep_prob).astype(np.float64)
        if idx == len(masked_layers) - 1:
            for col_idx in range(mask.shape[1]):
                if np.count_nonzero(mask[:, col_idx]) == 0:
                    row_idx = int(rng.integers(0, mask.shape[0]))
                    mask[row_idx, col_idx] = 1.0
        if np.count_nonzero(mask) == 0:
            mask.flat[0] = 1.0
        layer.masked_weight.set_mask(mask)

    x = np.random.default_rng(53).normal(size=(180, 64)).astype(np.float64)
    dense_logits = model.predict_logits_numpy(x)
    sparse_logits = model.predict_logits_numpy_sparse(x)

    np.testing.assert_allclose(sparse_logits, dense_logits, atol=1e-10, rtol=0.0)

    dense_logits_b1 = model.predict_logits_numpy(x[:1])
    sparse_logits_b1 = model.predict_logits_numpy_sparse(x[:1])
    np.testing.assert_allclose(sparse_logits_b1, dense_logits_b1, atol=1e-10, rtol=0.0)
