"""Integration tests for Phase 2 baseline training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import src.train as train_module

from src.train import (
    TrainConfig,
    load_phase2_dataset,
    profile_training_step,
    save_history_artifacts,
    train_baseline,
)


def test_train_baseline_learns_and_remains_finite() -> None:
    config = TrainConfig(
        dataset="digits",
        seed=7,
        hidden_size=128,
        batch_size=32,
        epochs=20,
        lr=1e-2,
    )

    artifacts = train_baseline(config, verbose=False)
    history = artifacts["history"]

    for key in ("train_loss", "val_loss", "train_acc", "val_acc"):
        values = np.asarray(history[key], dtype=np.float64)
        assert np.isfinite(values).all()

    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["val_acc"][-1] >= 0.90
    assert history["val_loss"][-1] < history["val_loss"][0]


def test_train_baseline_is_reproducible_for_fixed_seed() -> None:
    config = TrainConfig(
        dataset="digits",
        seed=11,
        hidden_size=64,
        batch_size=24,
        epochs=8,
        lr=1e-2,
    )

    run_one = train_baseline(config, verbose=False)["history"]
    run_two = train_baseline(config, verbose=False)["history"]

    for key in ("train_loss", "val_loss", "train_acc", "val_acc"):
        np.testing.assert_allclose(
            np.asarray(run_one[key], dtype=np.float64),
            np.asarray(run_two[key], dtype=np.float64),
            atol=0.0,
            rtol=0.0,
        )


def test_profile_training_step_generates_report(tmp_path) -> None:
    config = TrainConfig(
        dataset="digits",
        seed=3,
        hidden_size=64,
        batch_size=16,
        epochs=4,
        lr=1e-2,
    )

    output_path = tmp_path / "phase2_profile.txt"
    report = profile_training_step(config, output_path=str(output_path), top_n=10)

    assert "function calls" in report
    assert output_path.exists()
    assert "function calls" in output_path.read_text(encoding="utf-8")


def test_save_history_artifacts_writes_csv_and_png(tmp_path) -> None:
    history = {
        "epoch": [1, 2, 3],
        "train_loss": [1.2, 0.8, 0.5],
        "train_acc": [0.4, 0.7, 0.9],
        "val_loss": [1.3, 0.9, 0.6],
        "val_acc": [0.35, 0.65, 0.88],
    }
    paths = save_history_artifacts(
        history=history, output_dir=str(tmp_path), prefix="phase2_test"
    )

    assert "history_csv" in paths and "history_png" in paths
    assert (tmp_path / "phase2_test_history.csv").exists()
    assert (tmp_path / "phase2_test_learning_curve.png").exists()


def test_train_baseline_raises_on_non_finite_loss() -> None:
    config = TrainConfig(dataset="synthetic", seed=9, epochs=2, batch_size=16)

    x = np.ones((80, 64), dtype=np.float64)
    y = np.zeros((80,), dtype=np.int64)
    x[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite loss detected"):
        train_baseline(config, verbose=False, data_override=(x, y))


def test_synthetic_loader_honors_samples_per_class() -> None:
    config = TrainConfig(
        dataset="synthetic",
        seed=17,
        n_features=8,
        n_classes=5,
        samples_per_class=13,
    )

    x, y, _ = load_phase2_dataset(config)
    counts = np.bincount(y, minlength=config.n_classes)

    assert x.shape == (config.n_classes * config.samples_per_class, config.n_features)
    assert y.shape == (config.n_classes * config.samples_per_class,)
    np.testing.assert_array_equal(
        counts,
        np.full((config.n_classes,), config.samples_per_class, dtype=np.int64),
    )


def test_profile_report_sanitizes_python_install_paths(tmp_path) -> None:
    config = TrainConfig(dataset="digits", seed=5, hidden_size=32, epochs=2, batch_size=32)
    report_path = tmp_path / "profile.txt"
    report = profile_training_step(config, output_path=str(report_path), steps=5, top_n=200)

    normalized_report = report.casefold().replace("/", "\\")
    normalized_prefix = str(Path(sys.prefix)).casefold().replace("/", "\\")
    normalized_base_prefix = str(Path(sys.base_prefix)).casefold().replace("/", "\\")

    assert normalized_prefix not in normalized_report
    assert normalized_base_prefix not in normalized_report
    assert "<python>" in report or "<site-packages>" in report


def test_main_profile_step_exits_without_full_training(monkeypatch, tmp_path) -> None:
    called = {"train": False}

    def fake_profile(config, output_path=None, steps=20, top_n=20):
        return "profile report"

    def fake_train(config, verbose=False, data_override=None):
        called["train"] = True
        return {}

    monkeypatch.setattr(train_module, "profile_training_step", fake_profile)
    monkeypatch.setattr(train_module, "train_baseline", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--profile-step",
            "--quiet",
            "--profile-output",
            str(tmp_path / "profile.txt"),
        ],
    )

    train_module.main()
    assert called["train"] is False
