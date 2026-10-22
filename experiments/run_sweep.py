"""Phase 5 Pareto sweep runner.

This script produces reproducible evidence for sparsity/accuracy trade-offs:
- raw per-run CSV
- paired Taylor-vs-magnitude CSV at matched achieved sparsity
- Pareto plot
- JSON summary with paired uncertainty estimates and claim payload
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.train import (
    SelfPruningConfig,
    TrainConfig,
    load_phase2_dataset,
    standardize_train_eval,
    train_baseline,
    train_self_pruning,
    train_val_test_split,
)


RAW_CSV_FIELDNAMES = [
    "seed",
    "criterion",
    "target_requested",
    "metric_split",
    "eval_batch_size",
    "final_acc",
    "final_loss",
    "achieved_sparsity",
    "active_params",
    "total_params",
    "target_reached",
    "target_limitation",
    "train_seconds",
    "sparse_dense_max_abs_err",
    "sparse_csr_build_ms",
    "dense_latency_batch_eval_median_ms",
    "sparse_latency_batch_eval_total_median_ms",
    "sparse_latency_batch_eval_run_only_median_ms",
    "dense_latency_batch1_median_ms",
    "sparse_latency_batch1_total_median_ms",
    "sparse_latency_batch1_run_only_median_ms",
    "latency_speedup_dense_over_sparse_total",
    "latency_speedup_dense_over_sparse_run_only",
    "theory_dense_linear_flops",
    "theory_sparse_linear_flops",
    "theory_sparse_over_dense_flops",
    "theory_dense_batch1_linear_flops",
    "theory_sparse_batch1_linear_flops",
    "theory_sparse_over_dense_batch1_flops",
    "prune_start_epoch",
    "prune_end_epoch",
    "prune_frequency",
    "min_survivors_per_layer",
    "min_survivors_per_row",
    "min_survivors_per_col",
    "min_survivors_per_output_col",
]

PAIRED_CSV_FIELDNAMES = [
    "seed",
    "target_requested",
    "matched_achieved_sparsity",
    "matched",
    "active_gap",
    "achieved_gap",
    "taylor_acc",
    "magnitude_acc",
    "paired_diff_taylor_minus_magnitude",
]


_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _parse_float_list(text: str) -> List[float]:
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("expected at least one float value")
    return values


def _parse_int_list(text: str) -> List[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def _resolve_eval_metric(artifacts: Dict[str, object]) -> Tuple[str, float, float]:
    final_test_acc = artifacts.get("final_test_acc")
    final_test_loss = artifacts.get("final_test_loss")

    if final_test_acc is not None and final_test_loss is not None:
        return "test", float(final_test_acc), float(final_test_loss)

    history = artifacts["history"]
    return (
        "val",
        float(history["val_acc"][-1]),
        float(history["val_loss"][-1]),
    )


def _build_eval_split(config: TrainConfig) -> Tuple[np.ndarray, np.ndarray, str]:
    x, y, _ = load_phase2_dataset(config)
    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(
        x,
        y,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed + 1,
    )
    x_train, x_val, x_test = standardize_train_eval(x_train, x_val, x_test)

    if x_test.shape[0] > 0:
        return x_test, y_test, "test"
    return x_val, y_val, "val"


def _measure_latency_ms(
    fn,
    *,
    warmup_runs: int,
    timed_runs: int,
) -> Dict[str, float]:
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if timed_runs <= 0:
        raise ValueError("timed_runs must be positive")

    for _ in range(warmup_runs):
        fn()

    times_ms: List[float] = []
    for _ in range(timed_runs):
        start = time.perf_counter_ns()
        fn()
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        times_ms.append(float(elapsed_ms))

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr, ddof=0)),
        "p10_ms": float(np.percentile(arr, 10.0)),
        "p90_ms": float(np.percentile(arr, 90.0)),
    }


def _bootstrap_paired_mean_ci(
    diffs: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> Tuple[float, float]:
    if diffs.ndim != 1:
        raise ValueError("diffs must be one-dimensional")
    if diffs.size == 0:
        raise ValueError("diffs must be non-empty")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    rng = np.random.default_rng(seed)
    means = np.empty((resamples,), dtype=np.float64)
    for idx in range(resamples):
        sampled = rng.choice(diffs, size=diffs.size, replace=True)
        means[idx] = float(np.mean(sampled))

    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _paired_t_interval(diffs: np.ndarray) -> Tuple[float, float]:
    if diffs.ndim != 1:
        raise ValueError("diffs must be one-dimensional")
    if diffs.size == 0:
        raise ValueError("diffs must be non-empty")

    mean = float(np.mean(diffs))
    n = int(diffs.size)
    if n == 1:
        return mean, mean

    std = float(np.std(diffs, ddof=1))
    if std == 0.0:
        return mean, mean

    se = std / math.sqrt(float(n))
    df = n - 1
    t_crit = _T_CRITICAL_95.get(df, 1.96)
    margin = t_crit * se
    return mean - margin, mean + margin


def _paired_sign_flip_pvalue(
    diffs: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> float:
    if diffs.ndim != 1:
        raise ValueError("diffs must be one-dimensional")
    if diffs.size == 0:
        raise ValueError("diffs must be non-empty")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    observed = abs(float(np.mean(diffs)))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(resamples, diffs.size))
    sampled_means = np.abs(np.mean(signs * diffs[np.newaxis, :], axis=1))
    extreme = int(np.count_nonzero(sampled_means >= observed))
    return float((extreme + 1) / (resamples + 1))


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_paired_rows(
    *,
    raw_rows: List[Dict[str, object]],
    seeds: List[int],
    targets: List[float],
) -> List[Dict[str, object]]:
    keyed_rows: Dict[Tuple[int, float, str], Dict[str, object]] = {}
    for row in raw_rows:
        criterion = str(row["criterion"])
        if criterion not in {"taylor", "magnitude"}:
            continue
        key = (int(row["seed"]), float(row["target_requested"]), criterion)
        keyed_rows[key] = row

    paired_rows: List[Dict[str, object]] = []
    for seed in seeds:
        for target in targets:
            taylor_row = keyed_rows.get((int(seed), float(target), "taylor"))
            magnitude_row = keyed_rows.get((int(seed), float(target), "magnitude"))
            if taylor_row is None or magnitude_row is None:
                continue

            active_gap = int(taylor_row["active_params"]) - int(magnitude_row["active_params"])
            achieved_gap = float(taylor_row["achieved_sparsity"]) - float(magnitude_row["achieved_sparsity"])
            matched = active_gap == 0 and np.isclose(achieved_gap, 0.0, atol=1e-12)

            paired_rows.append(
                {
                    "seed": int(seed),
                    "target_requested": float(target),
                    "matched_achieved_sparsity": float(
                        min(
                            float(taylor_row["achieved_sparsity"]),
                            float(magnitude_row["achieved_sparsity"]),
                        )
                    ),
                    "matched": bool(matched),
                    "active_gap": int(active_gap),
                    "achieved_gap": float(achieved_gap),
                    "taylor_acc": float(taylor_row["final_acc"]),
                    "magnitude_acc": float(magnitude_row["final_acc"]),
                    "paired_diff_taylor_minus_magnitude": float(
                        float(taylor_row["final_acc"]) - float(magnitude_row["final_acc"])
                    ),
                }
            )
    return paired_rows


def summarize_paired_statistics(
    *,
    paired_rows: List[Dict[str, object]],
    targets: List[float],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    permutation_resamples: int,
) -> List[Dict[str, object]]:
    paired_stats = []
    for target in targets:
        target_rows = [
            row
            for row in paired_rows
            if np.isclose(float(row["target_requested"]), float(target))
            and bool(row["matched"])
        ]
        if not target_rows:
            continue

        diffs = np.asarray(
            [float(row["paired_diff_taylor_minus_magnitude"]) for row in target_rows],
            dtype=np.float64,
        )
        ci_low, ci_high = _bootstrap_paired_mean_ci(
            diffs,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + int(round(target * 1000.0)),
        )
        t_low, t_high = _paired_t_interval(diffs)
        perm_p = _paired_sign_flip_pvalue(
            diffs,
            resamples=permutation_resamples,
            seed=bootstrap_seed + 10_000 + int(round(target * 1000.0)),
        )

        paired_stats.append(
            {
                "target_requested": float(target),
                "matched_seed_count": int(diffs.size),
                "mean_diff_taylor_minus_magnitude": float(np.mean(diffs)),
                "std_diff_taylor_minus_magnitude": float(np.std(diffs, ddof=1) if diffs.size > 1 else 0.0),
                "bootstrap_ci95_low": float(ci_low),
                "bootstrap_ci95_high": float(ci_high),
                "t_ci95_low": float(t_low),
                "t_ci95_high": float(t_high),
                "permutation_pvalue_two_sided": float(perm_p),
                "ci_bootstrap_includes_zero": bool(ci_low <= 0.0 <= ci_high),
                "ci_t_includes_zero": bool(t_low <= 0.0 <= t_high),
                "zero_target_identical_by_construction": bool(np.isclose(float(target), 0.0, atol=1e-12)),
            }
        )
    return paired_stats


def build_quantitative_claim(
    *,
    raw_rows: List[Dict[str, object]],
    paired_rows: List[Dict[str, object]],
    paired_stats: List[Dict[str, object]],
    target: float,
) -> Dict[str, object]:
    matched_rows = [
        row
        for row in paired_rows
        if np.isclose(float(row["target_requested"]), target)
        and bool(row["matched"])
    ]
    if not matched_rows:
        raise ValueError(f"no matched paired rows for target={target}")

    taylor_acc = np.asarray([float(row["taylor_acc"]) for row in matched_rows], dtype=np.float64)
    magnitude_acc = np.asarray([float(row["magnitude_acc"]) for row in matched_rows], dtype=np.float64)
    matched_sparsity = np.asarray(
        [float(row["matched_achieved_sparsity"]) for row in matched_rows],
        dtype=np.float64,
    )

    stat_row = None
    for item in paired_stats:
        if np.isclose(float(item["target_requested"]), target):
            stat_row = item
            break
    if stat_row is None:
        raise ValueError(f"missing paired statistics for target={target}")

    target_raw_rows = [
        row
        for row in raw_rows
        if str(row["criterion"]) in {"taylor", "magnitude"}
        and np.isclose(float(row["target_requested"]), target)
    ]
    active_values = np.asarray([int(row["active_params"]) for row in target_raw_rows], dtype=np.int64)
    total_values = np.asarray([int(row["total_params"]) for row in target_raw_rows], dtype=np.int64)

    active_mean = int(round(float(np.mean(active_values))))
    total_mean = int(round(float(np.mean(total_values))))

    claim_text = (
        f"At achieved sparsity {float(np.mean(matched_sparsity)):.10f} "
        f"({active_mean} active of {total_mean}), across {len(matched_rows)} paired seeds, "
        f"Taylor achieved mean accuracy {float(np.mean(taylor_acc)):.6f} versus "
        f"{float(np.mean(magnitude_acc)):.6f} for magnitude; mean paired difference "
        f"(Taylor - magnitude) was {float(stat_row['mean_diff_taylor_minus_magnitude']):.6f} "
        f"with bootstrap 95% CI [{float(stat_row['bootstrap_ci95_low']):.6f}, {float(stat_row['bootstrap_ci95_high']):.6f}] "
        f"and paired t-interval [{float(stat_row['t_ci95_low']):.6f}, {float(stat_row['t_ci95_high']):.6f}]."
    )

    return {
        "target_requested": float(target),
        "paired_seed_ids": [int(row["seed"]) for row in matched_rows],
        "paired_seed_count": int(len(matched_rows)),
        "achieved_sparsity_mean": float(np.mean(matched_sparsity)),
        "active_params_mean": int(active_mean),
        "total_params_mean": int(total_mean),
        "taylor_acc_mean": float(np.mean(taylor_acc)),
        "magnitude_acc_mean": float(np.mean(magnitude_acc)),
        "mean_diff_taylor_minus_magnitude": float(stat_row["mean_diff_taylor_minus_magnitude"]),
        "bootstrap_ci95_low": float(stat_row["bootstrap_ci95_low"]),
        "bootstrap_ci95_high": float(stat_row["bootstrap_ci95_high"]),
        "t_ci95_low": float(stat_row["t_ci95_low"]),
        "t_ci95_high": float(stat_row["t_ci95_high"]),
        "permutation_pvalue_two_sided": float(stat_row["permutation_pvalue_two_sided"]),
        "claim_text": claim_text,
    }


def _plot_pareto(
    *,
    rows: List[Dict[str, object]],
    targets: List[float],
    output_path: Path,
    zoom_y_min: float,
    zoom_y_max: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    method_styles = {
        "taylor": {"color": "#1f77b4", "marker": "o", "label": "Taylor pruning"},
        "magnitude": {"color": "#d62728", "marker": "s", "label": "Magnitude pruning"},
    }

    method_curves: Dict[str, Dict[str, List[float]]] = {}
    for method in method_styles:
        method_curves[method] = {"x": [], "y": [], "yerr": []}

    for method, style in method_styles.items():
        x_values = []
        y_values = []
        y_err = []

        for target in targets:
            method_rows = [
                row
                for row in rows
                if row["criterion"] == method and np.isclose(float(row["target_requested"]), target)
            ]
            if not method_rows:
                continue

            achieved = np.asarray([float(row["achieved_sparsity"]) for row in method_rows], dtype=np.float64)
            acc = np.asarray([float(row["final_acc"]) for row in method_rows], dtype=np.float64)

            x_values.append(float(np.mean(achieved)))
            y_values.append(float(np.mean(acc)))
            y_err.append(float(np.std(acc, ddof=0)))

        method_curves[method]["x"] = x_values
        method_curves[method]["y"] = y_values
        method_curves[method]["yerr"] = y_err

    for ax in axes:
        for method, style in method_styles.items():
            x_values = method_curves[method]["x"]
            y_values = method_curves[method]["y"]
            y_err = method_curves[method]["yerr"]
            if not x_values:
                continue
            ax.errorbar(
                x_values,
                y_values,
                yerr=y_err,
                marker=style["marker"],
                color=style["color"],
                linewidth=1.5,
                capsize=3,
                label=style["label"],
            )

    baseline_rows = [row for row in rows if row["criterion"] == "dense_baseline"]
    if baseline_rows:
        baseline_acc = np.asarray([float(row["final_acc"]) for row in baseline_rows], dtype=np.float64)
        baseline_mean = float(np.mean(baseline_acc))
        for ax in axes:
            ax.scatter(
                [0.0],
                [baseline_mean],
                marker="*",
                s=120,
                color="#2ca02c",
                label="Dense baseline",
            )

    axes[0].set_xlabel("Achieved sparsity")
    axes[0].set_ylabel("Final accuracy")
    axes[0].set_title("Pareto sweep (full accuracy range)")
    axes[0].set_xlim(-0.02, 1.0)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Achieved sparsity")
    axes[1].set_ylabel("Final accuracy (zoom)")
    axes[1].set_title("Pareto sweep (zoomed y-axis)")
    axes[1].set_xlim(-0.02, 1.0)
    axes[1].set_ylim(zoom_y_min, zoom_y_max)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 5 Pareto evaluation sweep")
    parser.add_argument("--targets", default="0.0,0.5,0.75,0.9,0.95,0.97,0.98,0.99")
    parser.add_argument(
        "--seeds",
        default="100,101,102,103,104,105,106,107,108,109",
        help="Comma-separated paired seeds",
    )

    parser.add_argument("--dataset", choices=["digits", "synthetic"], default="digits")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], default="relu")

    parser.add_argument("--prune-start-epoch", type=int, default=2)
    parser.add_argument("--prune-end-epoch", type=int, default=-1)
    parser.add_argument("--prune-frequency", type=int, default=1)
    parser.add_argument("--min-survivors-per-layer", type=int, default=1)
    parser.add_argument("--min-survivors-per-row", type=int, default=0)
    parser.add_argument("--min-survivors-per-col", type=int, default=0)
    parser.add_argument("--min-survivors-per-output-col", type=int, default=1)

    parser.add_argument("--latency-warmup-runs", type=int, default=8)
    parser.add_argument("--latency-timed-runs", type=int, default=40)
    parser.add_argument("--equivalence-atol", type=float, default=1e-10)

    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261022)
    parser.add_argument("--permutation-resamples", type=int, default=10000)
    parser.add_argument(
        "--claim-target",
        type=float,
        default=-1.0,
        help="Target sparsity used for summary claim; -1 means highest target with matched rows",
    )
    parser.add_argument("--zoom-y-min", type=float, default=0.90)
    parser.add_argument("--zoom-y-max", type=float, default=1.00)

    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--raw-csv-name", default="phase5_pareto_raw.csv")
    parser.add_argument("--paired-csv-name", default="phase5_paired_differences.csv")
    parser.add_argument("--summary-json-name", default="phase5_summary.json")
    parser.add_argument("--pareto-plot-name", default="phase5_pareto_plot.png")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    targets = _parse_float_list(args.targets)
    seeds = _parse_int_list(args.seeds)
    if any(target < 0.0 or target >= 1.0 for target in targets):
        raise ValueError("all targets must be in [0, 1)")

    prune_end_epoch = args.epochs if args.prune_end_epoch == -1 else args.prune_end_epoch
    results_dir = Path(args.results_dir)
    raw_csv_path = results_dir / args.raw_csv_name
    paired_csv_path = results_dir / args.paired_csv_name
    summary_json_path = results_dir / args.summary_json_name
    pareto_plot_path = results_dir / args.pareto_plot_name

    methods = ["taylor", "magnitude"]

    raw_rows: List[Dict[str, object]] = []

    for seed in seeds:
        config = TrainConfig(
            seed=seed,
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            activation=args.activation,
            dataset=args.dataset,
            results_dir=str(results_dir),
        )

        x_eval, _y_eval, eval_split = _build_eval_split(config)
        if x_eval.shape[0] <= 0:
            raise ValueError("evaluation split must be non-empty")
        x_eval_batch1 = x_eval[:1]

        dense_start = time.perf_counter()
        dense_artifacts = train_baseline(config, verbose=False)
        dense_train_seconds = float(time.perf_counter() - dense_start)
        dense_metric_split, dense_acc, dense_loss = _resolve_eval_metric(dense_artifacts)
        if dense_metric_split != eval_split:
            raise ValueError("dense baseline metric split mismatch with eval split")

        dense_model = dense_artifacts["model"]
        dense_latency = _measure_latency_ms(
            lambda: dense_model.predict_logits_numpy(x_eval),
            warmup_runs=args.latency_warmup_runs,
            timed_runs=args.latency_timed_runs,
        )
        dense_latency_batch1 = _measure_latency_ms(
            lambda: dense_model.predict_logits_numpy(x_eval_batch1),
            warmup_runs=args.latency_warmup_runs,
            timed_runs=args.latency_timed_runs,
        )
        dense_cost = dense_model.estimate_inference_cost(batch_size=x_eval.shape[0], sparse=False)
        dense_cost_batch1 = dense_model.estimate_inference_cost(batch_size=1, sparse=False)

        raw_rows.append(
            {
                "seed": int(seed),
                "criterion": "dense_baseline",
                "target_requested": 0.0,
                "metric_split": dense_metric_split,
                "eval_batch_size": int(x_eval.shape[0]),
                "final_acc": dense_acc,
                "final_loss": dense_loss,
                "achieved_sparsity": 0.0,
                "active_params": int(dense_cost["active_weights"]),
                "total_params": int(dense_cost["total_weights"]),
                "target_reached": True,
                "target_limitation": "",
                "train_seconds": dense_train_seconds,
                "sparse_dense_max_abs_err": 0.0,
                "sparse_csr_build_ms": 0.0,
                "dense_latency_batch_eval_median_ms": float(dense_latency["median_ms"]),
                "sparse_latency_batch_eval_total_median_ms": float(dense_latency["median_ms"]),
                "sparse_latency_batch_eval_run_only_median_ms": float(dense_latency["median_ms"]),
                "dense_latency_batch1_median_ms": float(dense_latency_batch1["median_ms"]),
                "sparse_latency_batch1_total_median_ms": float(dense_latency_batch1["median_ms"]),
                "sparse_latency_batch1_run_only_median_ms": float(dense_latency_batch1["median_ms"]),
                "latency_speedup_dense_over_sparse_total": 1.0,
                "latency_speedup_dense_over_sparse_run_only": 1.0,
                "theory_dense_linear_flops": int(dense_cost["linear_flops"]),
                "theory_sparse_linear_flops": int(dense_cost["linear_flops"]),
                "theory_sparse_over_dense_flops": 1.0,
                "theory_dense_batch1_linear_flops": int(dense_cost_batch1["linear_flops"]),
                "theory_sparse_batch1_linear_flops": int(dense_cost_batch1["linear_flops"]),
                "theory_sparse_over_dense_batch1_flops": 1.0,
                "prune_start_epoch": int(args.prune_start_epoch),
                "prune_end_epoch": int(prune_end_epoch),
                "prune_frequency": int(args.prune_frequency),
                "min_survivors_per_layer": int(args.min_survivors_per_layer),
                "min_survivors_per_row": int(args.min_survivors_per_row),
                "min_survivors_per_col": int(args.min_survivors_per_col),
                "min_survivors_per_output_col": int(args.min_survivors_per_output_col),
            }
        )

        for target in targets:
            for criterion in methods:
                pruning_config = SelfPruningConfig(
                    criterion=criterion,
                    target_sparsity=float(target),
                    prune_start_epoch=args.prune_start_epoch,
                    prune_end_epoch=prune_end_epoch,
                    prune_frequency=args.prune_frequency,
                    min_survivors_per_layer=args.min_survivors_per_layer,
                    min_survivors_per_row=args.min_survivors_per_row,
                    min_survivors_per_col=args.min_survivors_per_col,
                    min_survivors_per_output_col=args.min_survivors_per_output_col,
                )

                train_start = time.perf_counter()
                artifacts = train_self_pruning(config, pruning_config, verbose=False)
                train_seconds = float(time.perf_counter() - train_start)

                metric_split, final_acc, final_loss = _resolve_eval_metric(artifacts)
                if metric_split != eval_split:
                    raise ValueError("sparse run metric split mismatch with eval split")

                model = artifacts["model"]
                plan_start = time.perf_counter_ns()
                sparse_plan = model.build_sparse_inference_plan()
                sparse_csr_build_ms = (time.perf_counter_ns() - plan_start) / 1_000_000.0

                dense_logits = model.predict_logits_numpy(x_eval)
                sparse_logits_total = model.predict_logits_numpy_sparse(x_eval)
                sparse_logits_run_only = model.predict_logits_numpy_sparse_with_plan(
                    x_eval,
                    layer_specs=sparse_plan,
                )
                max_abs_err = float(
                    max(
                        np.max(np.abs(dense_logits - sparse_logits_total)),
                        np.max(np.abs(dense_logits - sparse_logits_run_only)),
                    )
                )
                if max_abs_err > args.equivalence_atol:
                    raise ValueError(
                        "sparse-aware path deviates from dense masked path beyond tolerance: "
                        f"max_abs_err={max_abs_err}, atol={args.equivalence_atol}"
                    )

                dense_latency = _measure_latency_ms(
                    lambda: model.predict_logits_numpy(x_eval),
                    warmup_runs=args.latency_warmup_runs,
                    timed_runs=args.latency_timed_runs,
                )
                sparse_latency = _measure_latency_ms(
                    lambda: model.predict_logits_numpy_sparse(x_eval),
                    warmup_runs=args.latency_warmup_runs,
                    timed_runs=args.latency_timed_runs,
                )
                sparse_latency_run_only = _measure_latency_ms(
                    lambda: model.predict_logits_numpy_sparse_with_plan(
                        x_eval,
                        layer_specs=sparse_plan,
                    ),
                    warmup_runs=args.latency_warmup_runs,
                    timed_runs=args.latency_timed_runs,
                )
                dense_latency_batch1 = _measure_latency_ms(
                    lambda: model.predict_logits_numpy(x_eval_batch1),
                    warmup_runs=args.latency_warmup_runs,
                    timed_runs=args.latency_timed_runs,
                )
                sparse_latency_batch1 = _measure_latency_ms(
                    lambda: model.predict_logits_numpy_sparse(x_eval_batch1),
                    warmup_runs=args.latency_warmup_runs,
                    timed_runs=args.latency_timed_runs,
                )
                sparse_latency_batch1_run_only = _measure_latency_ms(
                    lambda: model.predict_logits_numpy_sparse_with_plan(
                        x_eval_batch1,
                        layer_specs=sparse_plan,
                    ),
                    warmup_runs=args.latency_warmup_runs,
                    timed_runs=args.latency_timed_runs,
                )

                dense_cost = model.estimate_inference_cost(batch_size=x_eval.shape[0], sparse=False)
                sparse_cost = model.estimate_inference_cost(batch_size=x_eval.shape[0], sparse=True)
                dense_cost_batch1 = model.estimate_inference_cost(batch_size=1, sparse=False)
                sparse_cost_batch1 = model.estimate_inference_cost(batch_size=1, sparse=True)

                sparse_median_ms = float(sparse_latency["median_ms"])
                dense_median_ms = float(dense_latency["median_ms"])
                sparse_run_only_median_ms = float(sparse_latency_run_only["median_ms"])
                latency_speedup_total = (
                    dense_median_ms / sparse_median_ms
                    if sparse_median_ms > 0.0
                    else float("nan")
                )
                latency_speedup_run_only = (
                    dense_median_ms / sparse_run_only_median_ms
                    if sparse_run_only_median_ms > 0.0
                    else float("nan")
                )

                sparse_over_dense_flops = (
                    float(sparse_cost["linear_flops"]) / float(dense_cost["linear_flops"])
                    if dense_cost["linear_flops"] > 0
                    else float("nan")
                )
                sparse_over_dense_batch1_flops = (
                    float(sparse_cost_batch1["linear_flops"]) / float(dense_cost_batch1["linear_flops"])
                    if dense_cost_batch1["linear_flops"] > 0
                    else float("nan")
                )

                raw_rows.append(
                    {
                        "seed": int(seed),
                        "criterion": criterion,
                        "target_requested": float(target),
                        "metric_split": metric_split,
                        "eval_batch_size": int(x_eval.shape[0]),
                        "final_acc": float(final_acc),
                        "final_loss": float(final_loss),
                        "achieved_sparsity": float(artifacts["final_sparsity"]),
                        "active_params": int(artifacts["final_active"]),
                        "total_params": int(artifacts["final_total"]),
                        "target_reached": bool(artifacts["target_reached"]),
                        "target_limitation": "" if artifacts["target_limitation"] is None else str(artifacts["target_limitation"]),
                        "train_seconds": float(train_seconds),
                        "sparse_dense_max_abs_err": float(max_abs_err),
                        "sparse_csr_build_ms": float(sparse_csr_build_ms),
                        "dense_latency_batch_eval_median_ms": dense_median_ms,
                        "sparse_latency_batch_eval_total_median_ms": sparse_median_ms,
                        "sparse_latency_batch_eval_run_only_median_ms": sparse_run_only_median_ms,
                        "dense_latency_batch1_median_ms": float(dense_latency_batch1["median_ms"]),
                        "sparse_latency_batch1_total_median_ms": float(sparse_latency_batch1["median_ms"]),
                        "sparse_latency_batch1_run_only_median_ms": float(sparse_latency_batch1_run_only["median_ms"]),
                        "latency_speedup_dense_over_sparse_total": float(latency_speedup_total),
                        "latency_speedup_dense_over_sparse_run_only": float(latency_speedup_run_only),
                        "theory_dense_linear_flops": int(dense_cost["linear_flops"]),
                        "theory_sparse_linear_flops": int(sparse_cost["linear_flops"]),
                        "theory_sparse_over_dense_flops": float(sparse_over_dense_flops),
                        "theory_dense_batch1_linear_flops": int(dense_cost_batch1["linear_flops"]),
                        "theory_sparse_batch1_linear_flops": int(sparse_cost_batch1["linear_flops"]),
                        "theory_sparse_over_dense_batch1_flops": float(sparse_over_dense_batch1_flops),
                        "prune_start_epoch": int(args.prune_start_epoch),
                        "prune_end_epoch": int(prune_end_epoch),
                        "prune_frequency": int(args.prune_frequency),
                        "min_survivors_per_layer": int(args.min_survivors_per_layer),
                        "min_survivors_per_row": int(args.min_survivors_per_row),
                        "min_survivors_per_col": int(args.min_survivors_per_col),
                        "min_survivors_per_output_col": int(args.min_survivors_per_output_col),
                    }
                )

                if not args.quiet:
                    print(
                        f"seed={seed} criterion={criterion} target={target:.4f} "
                        f"acc={final_acc:.4f} sparsity={artifacts['final_sparsity']:.4f}"
                    )

    _write_csv(raw_csv_path, raw_rows, RAW_CSV_FIELDNAMES)

    paired_rows = build_paired_rows(raw_rows=raw_rows, seeds=seeds, targets=targets)
    _write_csv(paired_csv_path, paired_rows, PAIRED_CSV_FIELDNAMES)

    paired_stats = summarize_paired_statistics(
        paired_rows=paired_rows,
        targets=targets,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        permutation_resamples=args.permutation_resamples,
    )

    _plot_pareto(
        rows=raw_rows,
        targets=targets,
        output_path=pareto_plot_path,
        zoom_y_min=float(args.zoom_y_min),
        zoom_y_max=float(args.zoom_y_max),
    )

    if args.claim_target < 0.0:
        claim_target = max(targets)
    else:
        claim_target = float(args.claim_target)
        if not any(np.isclose(claim_target, target) for target in targets):
            raise ValueError("claim_target must be included in --targets")

    claim_payload = build_quantitative_claim(
        raw_rows=raw_rows,
        paired_rows=paired_rows,
        paired_stats=paired_stats,
        target=float(claim_target),
    )

    summary = {
        "phase": "phase5",
        "sweep_config": {
            "targets": [float(v) for v in targets],
            "seeds": [int(v) for v in seeds],
            "methods": methods,
            "train_config": asdict(
                TrainConfig(
                    seed=int(seeds[0]),
                    hidden_size=args.hidden_size,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    activation=args.activation,
                    dataset=args.dataset,
                    results_dir=str(results_dir),
                )
            ),
            "pruning_defaults": asdict(
                SelfPruningConfig(
                    criterion="taylor",
                    target_sparsity=float(targets[0]),
                    prune_start_epoch=args.prune_start_epoch,
                    prune_end_epoch=prune_end_epoch,
                    prune_frequency=args.prune_frequency,
                    min_survivors_per_layer=args.min_survivors_per_layer,
                    min_survivors_per_row=args.min_survivors_per_row,
                    min_survivors_per_col=args.min_survivors_per_col,
                    min_survivors_per_output_col=args.min_survivors_per_output_col,
                )
            ),
            "latency_warmup_runs": int(args.latency_warmup_runs),
            "latency_timed_runs": int(args.latency_timed_runs),
            "equivalence_atol": float(args.equivalence_atol),
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "bootstrap_seed": int(args.bootstrap_seed),
            "permutation_resamples": int(args.permutation_resamples),
            "claim_target": float(claim_target),
            "zoom_y_min": float(args.zoom_y_min),
            "zoom_y_max": float(args.zoom_y_max),
        },
        "artifacts": {
            "raw_csv": str(raw_csv_path),
            "paired_csv": str(paired_csv_path),
            "pareto_plot": str(pareto_plot_path),
        },
        "paired_statistics": paired_stats,
        "quantitative_claim": claim_payload,
        "notes": [
            "theory_sparse_over_dense_flops is a linear-layer arithmetic estimate, not wall-clock time",
            "latency_speedup_dense_over_sparse_total > 1 means sparse path (including CSR build) is faster",
            "latency_speedup_dense_over_sparse_run_only > 1 means sparse arithmetic path is faster",
            "0% target paired rows are expected to be identical by construction",
            "no superiority claim should be made when confidence intervals include zero",
        ],
    }

    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
