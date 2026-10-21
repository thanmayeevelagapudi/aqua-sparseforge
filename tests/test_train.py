"""Integration tests for Phase 2 baseline training."""

from __future__ import annotations

import numpy as np

from src.train import TrainConfig, profile_training_step, train_baseline


def test_train_baseline_learns_and_remains_finite() -> None:
    config = TrainConfig(
        seed=7,
        samples_per_class=80,
        hidden_size=24,
        batch_size=32,
        epochs=25,
        lr=1e-2,
    )

    artifacts = train_baseline(config, verbose=False)
    history = artifacts["history"]

    for key in ("train_loss", "val_loss", "train_acc", "val_acc"):
        values = np.asarray(history[key], dtype=np.float64)
        assert np.isfinite(values).all()

    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["val_acc"][-1] >= 0.75


def test_train_baseline_is_reproducible_for_fixed_seed() -> None:
    config = TrainConfig(
        seed=11,
        samples_per_class=60,
        hidden_size=16,
        batch_size=24,
        epochs=12,
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
        seed=3,
        samples_per_class=40,
        hidden_size=12,
        batch_size=16,
        epochs=3,
        lr=1e-2,
    )

    output_path = tmp_path / "phase2_profile.txt"
    report = profile_training_step(config, output_path=str(output_path), top_n=10)

    assert "function calls" in report
    assert output_path.exists()
    assert "function calls" in output_path.read_text(encoding="utf-8")
