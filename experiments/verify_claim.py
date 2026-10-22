"""Verify the Phase 5 quantitative claim against committed CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from .run_sweep import build_quantitative_claim, summarize_paired_statistics


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse boolean: {value!r}")


def _load_raw_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "criterion": str(row["criterion"]),
                    "target_requested": float(row["target_requested"]),
                    "final_acc": float(row["final_acc"]),
                    "achieved_sparsity": float(row["achieved_sparsity"]),
                    "active_params": int(row["active_params"]),
                    "total_params": int(row["total_params"]),
                }
            )
    return rows


def _load_paired_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "target_requested": float(row["target_requested"]),
                    "matched_achieved_sparsity": float(row["matched_achieved_sparsity"]),
                    "matched": _parse_bool(row["matched"]),
                    "active_gap": int(row["active_gap"]),
                    "achieved_gap": float(row["achieved_gap"]),
                    "taylor_acc": float(row["taylor_acc"]),
                    "magnitude_acc": float(row["magnitude_acc"]),
                    "paired_diff_taylor_minus_magnitude": float(
                        row["paired_diff_taylor_minus_magnitude"]
                    ),
                }
            )
    return rows


def _assert_close(name: str, observed: float, expected: float, atol: float) -> None:
    if abs(observed - expected) > atol:
        raise AssertionError(
            f"{name} mismatch: observed={observed:.15g}, expected={expected:.15g}, atol={atol}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Phase 5 claim from committed artifacts")
    parser.add_argument("--raw-csv", default="experiments/results/phase5_pareto_raw.csv")
    parser.add_argument("--paired-csv", default="experiments/results/phase5_paired_differences.csv")
    parser.add_argument("--summary-json", default="experiments/results/phase5_summary.json")
    parser.add_argument("--atol", type=float, default=1e-12)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise FileNotFoundError(f"summary file not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    claim_expected = summary.get("quantitative_claim")
    if not isinstance(claim_expected, dict):
        raise ValueError("summary does not include quantitative_claim")

    sweep_config = summary.get("sweep_config")
    if not isinstance(sweep_config, dict):
        raise ValueError("summary missing sweep_config")

    claim_target = float(claim_expected["target_requested"])

    raw_rows = _load_raw_rows(Path(args.raw_csv))
    paired_rows = _load_paired_rows(Path(args.paired_csv))

    paired_stats = summarize_paired_statistics(
        paired_rows=paired_rows,
        targets=[claim_target],
        bootstrap_resamples=int(sweep_config["bootstrap_resamples"]),
        bootstrap_seed=int(sweep_config["bootstrap_seed"]),
        permutation_resamples=int(sweep_config["permutation_resamples"]),
    )
    claim_observed = build_quantitative_claim(
        raw_rows=raw_rows,
        paired_rows=paired_rows,
        paired_stats=paired_stats,
        target=claim_target,
    )

    if claim_observed["paired_seed_ids"] != claim_expected["paired_seed_ids"]:
        raise AssertionError(
            f"paired_seed_ids mismatch: observed={claim_observed['paired_seed_ids']}, expected={claim_expected['paired_seed_ids']}"
        )

    numeric_keys = [
        "achieved_sparsity_mean",
        "taylor_acc_mean",
        "magnitude_acc_mean",
        "mean_diff_taylor_minus_magnitude",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "t_ci95_low",
        "t_ci95_high",
        "permutation_pvalue_two_sided",
    ]
    for key in numeric_keys:
        _assert_close(
            key,
            float(claim_observed[key]),
            float(claim_expected[key]),
            atol=float(args.atol),
        )

    if int(claim_observed["active_params_mean"]) != int(claim_expected["active_params_mean"]):
        raise AssertionError(
            "active_params_mean mismatch: "
            f"observed={claim_observed['active_params_mean']}, expected={claim_expected['active_params_mean']}"
        )
    if int(claim_observed["total_params_mean"]) != int(claim_expected["total_params_mean"]):
        raise AssertionError(
            "total_params_mean mismatch: "
            f"observed={claim_observed['total_params_mean']}, expected={claim_expected['total_params_mean']}"
        )

    print("Phase 5 claim verification: PASS")
    print(claim_observed["claim_text"])


if __name__ == "__main__":
    main()
