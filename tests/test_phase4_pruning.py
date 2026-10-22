"""Phase 4 saliency/schedule pruning tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.pruning import (
    MaskedParameter,
    TaylorScoreAccumulator,
    apply_global_pruning_target,
    compute_hidden_unit_connectivity_report,
    compute_layer_connectivity_report,
    compute_magnitude_scores,
    compute_mask_sparsity,
    compute_taylor_scores,
    cubic_sparsity_target,
    enforced_target_active_count,
)
from src.tensor import Tensor
from src.train import (
    SelfPruningConfig,
    TrainConfig,
    run_paired_seed_comparison,
    train_self_pruning,
)


def test_cubic_schedule_is_monotonic_and_hits_target() -> None:
    values = [
        cubic_sparsity_target(
            epoch=epoch,
            prune_start_epoch=2,
            prune_end_epoch=6,
            final_sparsity=0.8,
        )
        for epoch in range(1, 8)
    ]

    assert values[0] == 0.0
    assert values[1] == 0.0
    assert values[-1] == 0.8
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def test_hard_target_pruning_reaches_requested_mask_count_when_feasible() -> None:
    masked_params = [
        MaskedParameter(
            tensor=Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True),
            mask=np.ones((2, 3), dtype=np.float64),
        ),
        MaskedParameter(
            tensor=Tensor(np.arange(4.0).reshape(2, 2), requires_grad=True),
            mask=np.ones((2, 2), dtype=np.float64),
        ),
    ]

    scores = [
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
        np.array([[0.7, 0.8], [0.9, 1.0]], dtype=np.float64),
    ]

    report = apply_global_pruning_target(
        masked_params,
        layer_scores=scores,
        target_sparsity=0.5,
        min_survivors_per_layer=1,
    )

    stats = compute_mask_sparsity(masked_params)
    expected_active = int(np.floor((1.0 - 0.5) * stats.total))

    assert report.reached_target
    assert report.limitation is None
    assert stats.active == expected_active


def test_enforced_target_active_count_achieves_at_least_requested_sparsity() -> None:
    total = 18_944
    for target in (0.6, 0.8, 0.85, 0.98):
        active = enforced_target_active_count(total, target)
        achieved = 1.0 - (active / total)
        assert achieved >= target
        assert active == int(np.floor((1.0 - target) * total))


def test_compute_magnitude_scores_matches_absolute_effective_weights() -> None:
    masked = MaskedParameter(
        tensor=Tensor(np.array([[-2.0, 0.5], [3.0, -4.0]], dtype=np.float64), requires_grad=True),
        mask=np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
    )
    scores = compute_magnitude_scores([masked])
    np.testing.assert_allclose(
        scores[0],
        np.array([[2.0, 0.0], [3.0, 4.0]], dtype=np.float64),
    )


def test_compute_taylor_scores_matches_w_times_grad_and_requires_grad_signal() -> None:
    masked = MaskedParameter(
        tensor=Tensor(np.array([[2.0, -3.0, 4.0]], dtype=np.float64), requires_grad=True),
        mask=np.array([[1.0, 0.0, 1.0]], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="effective_position_grad is missing"):
        compute_taylor_scores([masked])

    masked.set_effective_position_grad(np.array([[0.5, 7.0, -0.25]], dtype=np.float64))
    scores = compute_taylor_scores([masked])

    # Middle entry is masked, so effective weight is zero and score is zero.
    np.testing.assert_allclose(
        scores[0],
        np.array([[1.0, 0.0, 1.0]], dtype=np.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_apply_global_pruning_prunes_lowest_scores_first() -> None:
    masked = MaskedParameter(
        tensor=Tensor(np.array([[10.0, 20.0, 30.0, 40.0, 50.0]], dtype=np.float64), requires_grad=True),
        mask=np.ones((1, 5), dtype=np.float64),
    )
    scores = [np.array([[5.0, 1.0, 4.0, 2.0, 3.0]], dtype=np.float64)]

    report = apply_global_pruning_target(
        [masked],
        layer_scores=scores,
        target_sparsity=0.4,
        min_survivors_per_layer=0,
    )

    assert report.applied_pruned == 2
    expected_mask = np.array([[1.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float64)
    np.testing.assert_allclose(masked.mask, expected_mask, atol=0.0, rtol=0.0)


def test_row_col_survivor_guards_prevent_dimension_collapse() -> None:
    masked = MaskedParameter(
        tensor=Tensor(np.arange(9.0).reshape(3, 3), requires_grad=True),
        mask=np.ones((3, 3), dtype=np.float64),
    )
    scores = [
        np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float64,
        )
    ]

    apply_global_pruning_target(
        [masked],
        layer_scores=scores,
        target_sparsity=0.8,
        min_survivors_per_layer=0,
        min_survivors_per_row=1,
        min_survivors_per_col=1,
    )

    row_counts = np.count_nonzero(masked.mask, axis=1)
    col_counts = np.count_nonzero(masked.mask, axis=0)
    assert np.all(row_counts >= 1)
    assert np.all(col_counts >= 1)


def test_output_layer_column_guard_only_preserves_class_connectivity() -> None:
    hidden = MaskedParameter(
        tensor=Tensor(np.arange(24.0).reshape(4, 6), requires_grad=True),
        mask=np.ones((4, 6), dtype=np.float64),
    )
    output = MaskedParameter(
        tensor=Tensor(np.arange(18.0).reshape(6, 3), requires_grad=True),
        mask=np.ones((6, 3), dtype=np.float64),
    )

    report = apply_global_pruning_target(
        [hidden, output],
        layer_scores=[
            np.full((4, 6), 100.0, dtype=np.float64),
            np.arange(18.0, dtype=np.float64).reshape(6, 3),
        ],
        target_sparsity=0.9,
        min_survivors_per_layer=0,
        min_survivors_per_row=0,
        min_survivors_per_col=0,
        min_survivors_per_col_by_layer=[0, 1],
    )

    output_col_counts = np.count_nonzero(output.mask, axis=0)
    assert report.reached_target
    assert np.all(output_col_counts >= 1)


def test_taylor_and_magnitude_can_rank_connections_differently() -> None:
    masked = MaskedParameter(
        tensor=Tensor(np.array([[100.0, 0.1]], dtype=np.float64), requires_grad=True),
        mask=np.array([[1.0, 1.0]], dtype=np.float64),
    )
    masked.set_effective_position_grad(np.array([[1e-5, 10.0]], dtype=np.float64))

    magnitude_scores = compute_magnitude_scores([masked])[0]
    taylor_scores = compute_taylor_scores([masked])[0]

    assert int(np.argmin(magnitude_scores)) == 1
    assert int(np.argmin(taylor_scores)) == 0


def test_taylor_score_accumulator_averages_batches() -> None:
    masked = MaskedParameter(
        tensor=Tensor(np.array([[2.0, 1.0]], dtype=np.float64), requires_grad=True),
        mask=np.array([[1.0, 1.0]], dtype=np.float64),
    )
    accumulator = TaylorScoreAccumulator.from_masked_params([masked])

    masked.set_effective_position_grad(np.array([[1.0, 3.0]], dtype=np.float64))
    accumulator.add_batch([masked])

    masked.set_effective_position_grad(np.array([[3.0, 1.0]], dtype=np.float64))
    accumulator.add_batch([masked])

    mean_scores = accumulator.mean_scores()[0]
    np.testing.assert_allclose(mean_scores, np.array([[4.0, 2.0]], dtype=np.float64))


def test_layer_guard_prevents_collapse_and_reports_limitation() -> None:
    masked_params = [
        MaskedParameter(
            tensor=Tensor(np.arange(4.0).reshape(2, 2), requires_grad=True),
            mask=np.ones((2, 2), dtype=np.float64),
        ),
        MaskedParameter(
            tensor=Tensor(np.arange(4.0).reshape(2, 2), requires_grad=True),
            mask=np.ones((2, 2), dtype=np.float64),
        ),
    ]

    scores = [
        np.full((2, 2), 0.1, dtype=np.float64),
        np.full((2, 2), 0.2, dtype=np.float64),
    ]

    report = apply_global_pruning_target(
        masked_params,
        layer_scores=scores,
        target_sparsity=0.95,
        min_survivors_per_layer=3,
    )

    stats = compute_mask_sparsity(masked_params)
    assert stats.active == 6
    assert not report.reached_target
    assert report.limitation is not None


def test_sparsity_computed_from_masks_not_weight_magnitudes() -> None:
    masked = MaskedParameter(
        tensor=Tensor(np.array([[1e-30, 1e6]], dtype=np.float64), requires_grad=True),
        mask=np.array([[1.0, 0.0]], dtype=np.float64),
    )

    stats = compute_mask_sparsity([masked])
    assert stats.active == 1
    assert stats.inactive == 1
    assert np.isclose(stats.sparsity, 0.5)


def test_self_pruning_training_progressively_prunes_and_reports_target() -> None:
    config = TrainConfig(
        dataset="synthetic",
        seed=31,
        n_features=8,
        n_classes=3,
        samples_per_class=24,
        hidden_size=16,
        batch_size=8,
        epochs=6,
        lr=1e-2,
    )
    pruning_config = SelfPruningConfig(
        criterion="taylor",
        target_sparsity=0.5,
        prune_start_epoch=2,
        prune_end_epoch=6,
        prune_frequency=1,
        min_survivors_per_layer=1,
    )

    artifacts = train_self_pruning(config, pruning_config, verbose=False)
    history = artifacts["history"]

    sparsities = history["mask_sparsity"]
    assert all(sparsities[i] <= sparsities[i + 1] + 1e-12 for i in range(len(sparsities) - 1))
    assert any(pruned > 0 for pruned in history["pruned_this_epoch"])

    model = artifacts["model"]
    for masked_param in model.masked_parameters():
        np.testing.assert_allclose(
            masked_param.tensor.data[masked_param.mask == 0.0],
            0.0,
            atol=1e-12,
            rtol=0.0,
        )

    final_stats = compute_mask_sparsity(model.masked_parameters())
    assert np.isclose(final_stats.sparsity, artifacts["final_sparsity"], atol=0.0, rtol=0.0)
    assert artifacts["target_reached"]
    assert artifacts["target_limitation"] is None


def test_connectivity_reports_expose_dead_units() -> None:
    layer_a = MaskedParameter(
        tensor=Tensor(np.ones((3, 4), dtype=np.float64), requires_grad=True),
        mask=np.array(
            [[1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]],
            dtype=np.float64,
        ),
    )
    layer_b = MaskedParameter(
        tensor=Tensor(np.ones((4, 2), dtype=np.float64), requires_grad=True),
        mask=np.array(
            [[1, 0], [0, 0], [0, 0], [1, 1]],
            dtype=np.float64,
        ),
    )

    layer_reports = compute_layer_connectivity_report([layer_a, layer_b])
    hidden_reports = compute_hidden_unit_connectivity_report([layer_a, layer_b])

    assert layer_reports[0].zero_rows == 1
    assert layer_reports[0].zero_cols == 1
    assert hidden_reports[0].hidden_width == 4
    assert hidden_reports[0].disconnected_either_side >= 1


def test_self_pruning_schedule_adheres_to_configured_target_curve() -> None:
    config = TrainConfig(
        dataset="synthetic",
        seed=21,
        n_features=8,
        n_classes=3,
        samples_per_class=18,
        hidden_size=12,
        batch_size=6,
        epochs=6,
        lr=1e-2,
    )
    pruning_config = SelfPruningConfig(
        criterion="magnitude",
        target_sparsity=0.6,
        prune_start_epoch=2,
        prune_end_epoch=6,
        prune_frequency=2,
        min_survivors_per_layer=1,
    )

    artifacts = train_self_pruning(config, pruning_config, verbose=False)
    reports = artifacts["pruning_reports"]

    for report in reports:
        epoch = report["epoch"]
        expected_target = cubic_sparsity_target(
            epoch=epoch,
            prune_start_epoch=pruning_config.prune_start_epoch,
            prune_end_epoch=pruning_config.prune_end_epoch,
            final_sparsity=pruning_config.target_sparsity,
        )
        assert np.isclose(report["target_sparsity"], expected_target)
        if "active_after" in report and "total" in report:
            expected_active_cap = int(
                np.floor((1.0 - report["target_sparsity"]) * report["total"])
            )
            assert int(report["active_after"]) <= expected_active_cap
            assert bool(report["reached_target"])

    non_prune_epochs = [1, 3, 5]
    for epoch in non_prune_epochs:
        assert reports[epoch - 1]["applied_pruned"] == 0


def test_self_pruning_is_deterministic_for_fixed_seed() -> None:
    config = TrainConfig(
        dataset="synthetic",
        seed=77,
        n_features=8,
        n_classes=3,
        samples_per_class=18,
        hidden_size=12,
        batch_size=6,
        epochs=5,
        lr=1e-2,
    )
    pruning_config = SelfPruningConfig(
        criterion="taylor",
        target_sparsity=0.5,
        prune_start_epoch=2,
        prune_end_epoch=5,
        prune_frequency=1,
        min_survivors_per_layer=1,
    )

    run_a = train_self_pruning(config, pruning_config, verbose=False)
    run_b = train_self_pruning(config, pruning_config, verbose=False)

    for key in (
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "mask_sparsity",
        "target_sparsity",
    ):
        np.testing.assert_allclose(
            np.asarray(run_a["history"][key], dtype=np.float64),
            np.asarray(run_b["history"][key], dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )

    assert run_a["pruning_reports"] == run_b["pruning_reports"]
    assert run_a["target_reached"] == run_b["target_reached"]


def test_paired_seed_comparison_reports_expected_shape() -> None:
    config = TrainConfig(
        dataset="synthetic",
        seed=5,
        n_features=8,
        n_classes=3,
        samples_per_class=12,
        hidden_size=8,
        batch_size=6,
        epochs=3,
        lr=1e-2,
        test_ratio=0.2,
    )
    pruning_config = SelfPruningConfig(
        criterion="magnitude",
        target_sparsity=0.3,
        prune_start_epoch=2,
        prune_end_epoch=3,
        prune_frequency=1,
        min_survivors_per_layer=1,
        min_survivors_per_row=1,
        min_survivors_per_col=1,
    )

    report = run_paired_seed_comparison(
        base_config=config,
        pruning_config=pruning_config,
        seed_count=2,
        start_seed=101,
    )

    assert report["seed_count"] == 2
    assert report["split"] == "test"
    assert len(report["per_seed"]) == 2


def test_self_pruning_magnitude_path_runs_and_prunes() -> None:
    config = TrainConfig(
        dataset="synthetic",
        seed=9,
        n_features=8,
        n_classes=3,
        samples_per_class=18,
        hidden_size=12,
        batch_size=6,
        epochs=5,
        lr=1e-2,
    )
    pruning_config = SelfPruningConfig(
        criterion="magnitude",
        target_sparsity=0.4,
        prune_start_epoch=2,
        prune_end_epoch=5,
        prune_frequency=1,
        min_survivors_per_layer=1,
    )

    artifacts = train_self_pruning(config, pruning_config, verbose=False)
    assert artifacts["history"]["mask_sparsity"][-1] > 0.0
