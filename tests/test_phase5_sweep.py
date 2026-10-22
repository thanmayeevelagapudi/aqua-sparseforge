"""Unit tests for Phase 5 sweep/statistics helpers."""

from __future__ import annotations

import numpy as np

from experiments import run_sweep


def test_bootstrap_ci_contains_mean_for_symmetric_diffs() -> None:
    diffs = np.array([-0.02, -0.01, 0.0, 0.01, 0.02], dtype=np.float64)
    low, high = run_sweep._bootstrap_paired_mean_ci(
        diffs,
        resamples=4000,
        seed=123,
    )
    mean = float(np.mean(diffs))

    assert low <= mean <= high
    assert high >= low


def test_paired_t_interval_contains_sample_mean() -> None:
    diffs = np.array([-0.03, -0.01, 0.00, 0.01, 0.02], dtype=np.float64)
    low, high = run_sweep._paired_t_interval(diffs)
    mean = float(np.mean(diffs))

    assert low <= mean <= high
    assert high >= low


def test_build_paired_rows_marks_matched_and_unmatched_pairs() -> None:
    raw_rows = [
        {
            "seed": 1,
            "criterion": "taylor",
            "target_requested": 0.9,
            "final_acc": 0.95,
            "achieved_sparsity": 0.9,
            "active_params": 100,
        },
        {
            "seed": 1,
            "criterion": "magnitude",
            "target_requested": 0.9,
            "final_acc": 0.94,
            "achieved_sparsity": 0.9,
            "active_params": 100,
        },
        {
            "seed": 2,
            "criterion": "taylor",
            "target_requested": 0.9,
            "final_acc": 0.93,
            "achieved_sparsity": 0.905,
            "active_params": 101,
        },
        {
            "seed": 2,
            "criterion": "magnitude",
            "target_requested": 0.9,
            "final_acc": 0.92,
            "achieved_sparsity": 0.9,
            "active_params": 100,
        },
    ]

    paired = run_sweep.build_paired_rows(raw_rows=raw_rows, seeds=[1, 2], targets=[0.9])
    assert len(paired) == 2

    matched = [row for row in paired if row["seed"] == 1][0]
    unmatched = [row for row in paired if row["seed"] == 2][0]

    assert matched["matched"] is True
    assert np.isclose(matched["paired_diff_taylor_minus_magnitude"], 0.01)

    assert unmatched["matched"] is False
    assert unmatched["active_gap"] == 1


def test_summarize_paired_statistics_reports_expected_keys() -> None:
    paired_rows = [
        {
            "seed": 10,
            "target_requested": 0.95,
            "matched_achieved_sparsity": 0.95,
            "matched": True,
            "active_gap": 0,
            "achieved_gap": 0.0,
            "taylor_acc": 0.97,
            "magnitude_acc": 0.96,
            "paired_diff_taylor_minus_magnitude": 0.01,
        },
        {
            "seed": 11,
            "target_requested": 0.95,
            "matched_achieved_sparsity": 0.95,
            "matched": True,
            "active_gap": 0,
            "achieved_gap": 0.0,
            "taylor_acc": 0.96,
            "magnitude_acc": 0.965,
            "paired_diff_taylor_minus_magnitude": -0.005,
        },
    ]

    stats = run_sweep.summarize_paired_statistics(
        paired_rows=paired_rows,
        targets=[0.95],
        bootstrap_resamples=2000,
        bootstrap_seed=9,
        permutation_resamples=2000,
    )

    assert len(stats) == 1
    stat = stats[0]

    for key in (
        "mean_diff_taylor_minus_magnitude",
        "std_diff_taylor_minus_magnitude",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "t_ci95_low",
        "t_ci95_high",
        "permutation_pvalue_two_sided",
    ):
        assert key in stat



def test_phase5_csv_schema_contains_latency_and_cost_columns() -> None:
    expected_columns = {
        "sparse_csr_build_ms",
        "dense_latency_batch_eval_median_ms",
        "sparse_latency_batch_eval_total_median_ms",
        "sparse_latency_batch_eval_run_only_median_ms",
        "dense_latency_batch1_median_ms",
        "sparse_latency_batch1_total_median_ms",
        "sparse_latency_batch1_run_only_median_ms",
        "latency_speedup_dense_over_sparse_total",
        "latency_speedup_dense_over_sparse_run_only",
        "theory_sparse_over_dense_flops",
        "theory_sparse_over_dense_batch1_flops",
    }

    assert expected_columns.issubset(set(run_sweep.RAW_CSV_FIELDNAMES))
    assert "paired_diff_taylor_minus_magnitude" in run_sweep.PAIRED_CSV_FIELDNAMES
