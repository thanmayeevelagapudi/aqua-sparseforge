"""Training and profiling entrypoint for dense baseline and self-pruning runs."""

from __future__ import annotations

import argparse
import cProfile
import csv
import io
import json
import pstats
import site
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from .nn import MLP
from .ops import cross_entropy
from .optim import Adam, MaskedAdam
from .pruning import (
    TaylorScoreAccumulator,
    apply_global_pruning_target,
    compute_hidden_unit_connectivity_report,
    compute_layer_connectivity_report,
    compute_magnitude_scores,
    compute_mask_sparsity,
    cubic_sparsity_target,
    enforced_target_active_count,
)
from .tensor import Tensor


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    n_features: int = 64
    n_classes: int = 10
    samples_per_class: int = 120
    hidden_size: int = 256
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-2
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    activation: str = "relu"
    dataset: str = "digits"
    results_dir: str = "experiments/results"


@dataclass(frozen=True)
class SelfPruningConfig:
    criterion: str = "taylor"
    target_sparsity: float = 0.8
    prune_start_epoch: int = 2
    prune_end_epoch: int = 30
    prune_frequency: int = 1
    min_survivors_per_layer: int = 1
    min_survivors_per_row: int = 0
    min_survivors_per_col: int = 0
    min_survivors_per_output_col: int = 1


def _validate_config(config: TrainConfig) -> None:
    if config.n_features < 2:
        raise ValueError("n_features must be at least 2 for synthetic geometry")
    if config.n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if config.samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    if config.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if not (0.0 < config.val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")
    if not (0.0 <= config.test_ratio < 1.0):
        raise ValueError("test_ratio must be in [0, 1)")
    if config.val_ratio + config.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")
    if config.dataset not in {"digits", "synthetic"}:
        raise ValueError("dataset must be 'digits' or 'synthetic'")


def _validate_self_pruning_config(
    config: TrainConfig,
    pruning_config: SelfPruningConfig,
) -> None:
    if pruning_config.criterion not in {"magnitude", "taylor"}:
        raise ValueError("criterion must be 'magnitude' or 'taylor'")
    if not (0.0 <= pruning_config.target_sparsity < 1.0):
        raise ValueError("target_sparsity must be in [0, 1)")
    if pruning_config.prune_start_epoch < 1:
        raise ValueError("prune_start_epoch must be >= 1")
    if pruning_config.prune_end_epoch < pruning_config.prune_start_epoch:
        raise ValueError("prune_end_epoch must be >= prune_start_epoch")
    if pruning_config.prune_frequency <= 0:
        raise ValueError("prune_frequency must be positive")
    if pruning_config.min_survivors_per_layer < 0:
        raise ValueError("min_survivors_per_layer must be non-negative")
    if pruning_config.min_survivors_per_row < 0:
        raise ValueError("min_survivors_per_row must be non-negative")
    if pruning_config.min_survivors_per_col < 0:
        raise ValueError("min_survivors_per_col must be non-negative")
    if pruning_config.min_survivors_per_output_col < 0:
        raise ValueError("min_survivors_per_output_col must be non-negative")
    if pruning_config.prune_end_epoch > config.epochs:
        raise ValueError("prune_end_epoch must be <= total epochs")


def make_synthetic_classification(
    *,
    seed: int,
    n_features: int,
    n_classes: int,
    samples_per_class: int,
    cluster_std: float = 0.65,
    radius: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic, separable multi-class synthetic dataset."""
    rng = np.random.default_rng(seed)

    all_x = []
    all_y = []
    angles = np.linspace(0.0, 2.0 * np.pi, n_classes, endpoint=False)

    for class_id, angle in enumerate(angles):
        center = np.zeros((n_features,), dtype=np.float64)
        center[0] = radius * np.cos(angle)
        center[1] = radius * np.sin(angle)
        if n_features > 2:
            center[2:] = np.linspace(-0.5, 0.5, n_features - 2, dtype=np.float64)

        class_x = rng.normal(
            loc=center,
            scale=cluster_std,
            size=(samples_per_class, n_features),
        ).astype(np.float64)
        class_y = np.full((samples_per_class,), class_id, dtype=np.int64)

        all_x.append(class_x)
        all_y.append(class_y)

    x = np.vstack(all_x)
    y = np.concatenate(all_y)

    perm = rng.permutation(x.shape[0])
    return x[perm], y[perm]


def load_phase2_dataset(config: TrainConfig) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load dataset used for Phase 2 baseline."""
    if config.dataset == "digits":
        digits = load_digits()
        x = digits.data.astype(np.float64)
        y = digits.target.astype(np.int64)
        desc = "sklearn.datasets.load_digits (1797 samples, 64 features, 10 classes)"
        return x, y, desc

    x, y = make_synthetic_classification(
        seed=config.seed,
        n_features=config.n_features,
        n_classes=config.n_classes,
        samples_per_class=config.samples_per_class,
    )
    desc = "NumPy synthetic Gaussian clusters"
    return x, y, desc


def train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(x.shape[0])

    val_count = int(round(x.shape[0] * val_ratio))
    val_count = max(1, min(val_count, x.shape[0] - 1))

    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def train_val_test_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have matching first dimension")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(x.shape[0])

    n_total = x.shape[0]
    test_count = int(round(n_total * test_ratio))
    val_count = int(round(n_total * val_ratio))

    if test_count + val_count >= n_total:
        raise ValueError("invalid val/test split leaves no training samples")

    test_idx = perm[:test_count]
    val_idx = perm[test_count : test_count + val_count]
    train_idx = perm[test_count + val_count :]

    return (
        x[train_idx],
        y[train_idx],
        x[val_idx],
        y[val_idx],
        x[test_idx],
        y[test_idx],
    )


def standardize_train_val(
    x_train: np.ndarray,
    x_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x_train - mean) / std, (x_val - mean) / std


def standardize_train_eval(
    x_train: np.ndarray,
    *eval_sets: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    outputs = [(x_train - mean) / std]
    for eval_set in eval_sets:
        outputs.append((eval_set - mean) / std)
    return tuple(outputs)


def iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have matching first dimension")

    indices = np.arange(x.shape[0])
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]


def accuracy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    predictions = np.argmax(logits, axis=1)
    return float(np.mean(predictions == targets))


def _check_finite_params_and_grads(model: MLP) -> None:
    for param in model.parameters():
        if not np.isfinite(param.data).all():
            raise ValueError("encountered non-finite parameter value")
        if param.grad is not None and not np.isfinite(param.grad).all():
            raise ValueError("encountered non-finite gradient")


def evaluate_model(model: MLP, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    logits_data = model.predict_logits_numpy(x)
    logits = Tensor(logits_data, requires_grad=False)
    loss = cross_entropy(logits, y).item()
    if not np.isfinite(loss):
        raise ValueError("evaluation produced non-finite loss")
    acc = accuracy_from_logits(logits_data, y)
    return float(loss), float(acc)


def estimate_compute_budget(
    *,
    n_train: int,
    batch_size: int,
    epochs: int,
    seconds_per_step: float,
    planned_runs: int,
) -> Dict[str, float]:
    steps_per_epoch = int(np.ceil(n_train / batch_size))
    steps_per_run = steps_per_epoch * epochs
    total_steps = steps_per_run * planned_runs
    estimated_total_seconds = total_steps * seconds_per_step
    return {
        "steps_per_epoch": float(steps_per_epoch),
        "steps_per_run": float(steps_per_run),
        "planned_runs": float(planned_runs),
        "total_steps": float(total_steps),
        "seconds_per_step": float(seconds_per_step),
        "estimated_total_seconds": float(estimated_total_seconds),
    }


def _sanitize_profile_report(report: str) -> str:
    def _variants(path_str: str) -> set[str]:
        return {
            path_str,
            path_str.replace("\\", "/"),
            path_str.replace("/", "\\"),
        }

    replacement_pairs = []
    replacement_pairs.append((str(Path.cwd().resolve()), "<repo>"))

    site_paths = []
    try:
        site_paths.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            site_paths.append(user_site)
    except Exception:
        pass

    for site_path in site_paths:
        replacement_pairs.append((str(Path(site_path).resolve()), "<site-packages>"))

    python_roots = {
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(sys.exec_prefix),
        Path(sys.base_exec_prefix),
    }
    try:
        python_exe = Path(sys.executable).resolve()
        python_roots.add(python_exe.parent)
        if python_exe.parent.parent != python_exe.parent:
            python_roots.add(python_exe.parent.parent)
    except Exception:
        pass

    for root in python_roots:
        replacement_pairs.append((str(root.resolve()), "<python>"))

    unique_pairs = {}
    for old_path, marker in replacement_pairs:
        if old_path:
            unique_pairs[old_path] = marker

    for old_path, marker in sorted(
        unique_pairs.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        for variant in _variants(old_path):
            report = report.replace(variant, marker)

    return report


def save_history_artifacts(
    *,
    history: Dict[str, list],
    output_dir: str,
    prefix: str = "phase2",
) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"{prefix}_history.csv"
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    for optional_field in ("mask_sparsity", "target_sparsity", "pruned_this_epoch"):
        if optional_field in history:
            fieldnames.append(optional_field)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(len(history["epoch"])):
            row = {
                "epoch": history["epoch"][idx],
                "train_loss": history["train_loss"][idx],
                "train_acc": history["train_acc"][idx],
                "val_loss": history["val_loss"][idx],
                "val_acc": history["val_acc"][idx],
            }
            if "mask_sparsity" in history:
                row["mask_sparsity"] = history["mask_sparsity"][idx]
            if "target_sparsity" in history:
                row["target_sparsity"] = history["target_sparsity"][idx]
            if "pruned_this_epoch" in history:
                row["pruned_this_epoch"] = history["pruned_this_epoch"][idx]
            writer.writerow(row)

    png_path = output_path / f"{prefix}_learning_curve.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_acc"], label="train acc")
    axes[1].plot(history["epoch"], history["val_acc"], label="val acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    return {
        "history_csv": str(csv_path),
        "history_png": str(png_path),
    }


def _emit_json_summary(summary: Dict[str, object], json_out: str) -> None:
    text = json.dumps(summary, indent=2)
    if json_out:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)


def train_baseline(
    config: TrainConfig,
    *,
    verbose: bool = False,
    data_override: Tuple[np.ndarray, np.ndarray] | None = None,
) -> Dict[str, object]:
    """Train a deterministic dense baseline MLP classifier."""
    _validate_config(config)

    if data_override is None:
        x, y, dataset_desc = load_phase2_dataset(config)
    else:
        x, y = data_override
        x = np.array(x, dtype=np.float64, copy=True)
        y = np.array(y, dtype=np.int64, copy=True)
        dataset_desc = "override dataset"

    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(
        x,
        y,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed + 1,
    )
    x_train, x_val, x_test = standardize_train_eval(x_train, x_val, x_test)

    model_rng = np.random.default_rng(config.seed + 2)
    inferred_n_features = x_train.shape[1]
    inferred_n_classes = int(np.max(y_train)) + 1
    model = MLP(
        in_features=inferred_n_features,
        hidden_sizes=[config.hidden_size],
        out_features=inferred_n_classes,
        activation=config.activation,
        rng=model_rng,
    )
    optimizer = Adam(model.parameters(), lr=config.lr)
    batch_rng = np.random.default_rng(config.seed + 3)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, config.epochs + 1):
        for x_batch, y_batch in iterate_minibatches(
            x_train,
            y_train,
            batch_size=config.batch_size,
            rng=batch_rng,
            shuffle=True,
        ):
            optimizer.zero_grad()

            logits = model(Tensor(x_batch, requires_grad=False))
            loss = cross_entropy(logits, y_batch)
            if not np.isfinite(loss.item()):
                raise ValueError("non-finite loss detected during training")

            loss.backward()
            _check_finite_params_and_grads(model)
            optimizer.step()

        train_loss, train_acc = evaluate_model(model, x_train, y_train)
        val_loss, val_acc = evaluate_model(model, x_val, y_val)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                ,
                file=sys.stderr,
            )

    final_test_loss = None
    final_test_acc = None
    if x_test.shape[0] > 0:
        final_test_loss, final_test_acc = evaluate_model(model, x_test, y_test)

    return {
        "config": config,
        "model": model,
        "history": history,
        "dataset": dataset_desc,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "splits": {
            "train_samples": int(x_train.shape[0]),
            "val_samples": int(x_val.shape[0]),
            "test_samples": int(x_test.shape[0]),
            "n_features": int(inferred_n_features),
            "n_classes": int(inferred_n_classes),
        },
    }


def train_self_pruning(
    config: TrainConfig,
    pruning_config: SelfPruningConfig,
    *,
    verbose: bool = False,
    data_override: Tuple[np.ndarray, np.ndarray] | None = None,
) -> Dict[str, object]:
    """Train a masked MLP with scheduled pruning and hard sparsity targets."""
    _validate_config(config)
    _validate_self_pruning_config(config, pruning_config)

    if data_override is None:
        x, y, dataset_desc = load_phase2_dataset(config)
    else:
        x, y = data_override
        x = np.array(x, dtype=np.float64, copy=True)
        y = np.array(y, dtype=np.int64, copy=True)
        dataset_desc = "override dataset"

    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(
        x,
        y,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed + 1,
    )
    x_train, x_val, x_test = standardize_train_eval(x_train, x_val, x_test)

    model_rng = np.random.default_rng(config.seed + 2)
    inferred_n_features = x_train.shape[1]
    inferred_n_classes = int(np.max(y_train)) + 1
    model = MLP(
        in_features=inferred_n_features,
        hidden_sizes=[config.hidden_size],
        out_features=inferred_n_classes,
        activation=config.activation,
        masked=True,
        rng=model_rng,
    )

    masked_params = model.masked_parameters()
    dense_params = [layer.bias for layer in model.layers if layer.bias is not None]
    optimizer = MaskedAdam(masked_params=masked_params, dense_params=dense_params, lr=config.lr)
    batch_rng = np.random.default_rng(config.seed + 3)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "mask_sparsity": [],
        "target_sparsity": [],
        "pruned_this_epoch": [],
    }
    pruning_reports = []

    for epoch in range(1, config.epochs + 1):
        taylor_accumulator: Optional[TaylorScoreAccumulator] = None
        if pruning_config.criterion == "taylor":
            taylor_accumulator = TaylorScoreAccumulator.from_masked_params(masked_params)

        for x_batch, y_batch in iterate_minibatches(
            x_train,
            y_train,
            batch_size=config.batch_size,
            rng=batch_rng,
            shuffle=True,
        ):
            optimizer.zero_grad()

            logits = model(Tensor(x_batch, requires_grad=False))
            loss = cross_entropy(logits, y_batch)
            if not np.isfinite(loss.item()):
                raise ValueError("non-finite loss detected during training")

            loss.backward()
            _check_finite_params_and_grads(model)

            if taylor_accumulator is not None:
                taylor_accumulator.add_batch(masked_params)

            optimizer.step()

        target_sparsity = cubic_sparsity_target(
            epoch=epoch,
            prune_start_epoch=pruning_config.prune_start_epoch,
            prune_end_epoch=pruning_config.prune_end_epoch,
            final_sparsity=pruning_config.target_sparsity,
        )

        run_prune = (
            epoch >= pruning_config.prune_start_epoch
            and (epoch - pruning_config.prune_start_epoch) % pruning_config.prune_frequency == 0
        )

        report_dict = {
            "epoch": epoch,
            "target_sparsity": float(target_sparsity),
            "applied_pruned": 0,
            "requested_pruned": 0,
            "reached_target": True,
            "limitation": None,
        }

        if run_prune:
            if pruning_config.criterion == "magnitude":
                layer_scores = compute_magnitude_scores(masked_params)
            else:
                if taylor_accumulator is None:
                    raise ValueError("Taylor accumulator is missing")
                layer_scores = taylor_accumulator.mean_scores()

            row_minima_by_layer = [pruning_config.min_survivors_per_row] * len(masked_params)
            col_minima_by_layer = [pruning_config.min_survivors_per_col] * len(masked_params)
            if col_minima_by_layer:
                col_minima_by_layer[-1] = max(
                    col_minima_by_layer[-1],
                    pruning_config.min_survivors_per_output_col,
                )

            prune_report = apply_global_pruning_target(
                masked_params,
                layer_scores=layer_scores,
                target_sparsity=target_sparsity,
                min_survivors_per_layer=pruning_config.min_survivors_per_layer,
                min_survivors_per_row=pruning_config.min_survivors_per_row,
                min_survivors_per_col=pruning_config.min_survivors_per_col,
                min_survivors_per_row_by_layer=row_minima_by_layer,
                min_survivors_per_col_by_layer=col_minima_by_layer,
            )
            report_dict = {
                "epoch": epoch,
                "target_sparsity": float(prune_report.target_sparsity),
                "enforced_target_sparsity": float(prune_report.enforced_target_sparsity),
                "active_before": int(prune_report.active_before),
                "active_after": int(prune_report.active_after),
                "applied_pruned": int(prune_report.applied_pruned),
                "requested_pruned": int(prune_report.requested_pruned),
                "max_prunable_with_guards": int(prune_report.max_prunable_with_guards),
                "reached_target": bool(prune_report.reached_target),
                "limitation": prune_report.limitation,
            }

        pruning_reports.append(report_dict)

        mask_stats = compute_mask_sparsity(masked_params)
        train_loss, train_acc = evaluate_model(model, x_train, y_train)
        val_loss, val_acc = evaluate_model(model, x_val, y_val)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["mask_sparsity"].append(mask_stats.sparsity)
        history["target_sparsity"].append(target_sparsity)
        history["pruned_this_epoch"].append(int(report_dict["applied_pruned"]))

        if verbose:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"sparsity={mask_stats.sparsity:.4f} target={target_sparsity:.4f} "
                f"pruned={int(report_dict['applied_pruned'])}"
                ,
                file=sys.stderr,
            )

    final_stats = compute_mask_sparsity(masked_params)
    final_target_requested = cubic_sparsity_target(
        epoch=config.epochs,
        prune_start_epoch=pruning_config.prune_start_epoch,
        prune_end_epoch=pruning_config.prune_end_epoch,
        final_sparsity=pruning_config.target_sparsity,
    )
    final_target_active = enforced_target_active_count(
        final_stats.total,
        final_target_requested,
    )
    final_target_enforced = 1.0 - (final_target_active / final_stats.total)
    final_layer_connectivity = compute_layer_connectivity_report(masked_params)
    final_hidden_connectivity = compute_hidden_unit_connectivity_report(masked_params)

    limitation = None
    target_reached = final_stats.active <= final_target_active
    if pruning_reports:
        last_limitation = pruning_reports[-1].get("limitation")
        if last_limitation is not None:
            limitation = str(last_limitation)
    if not target_reached and limitation is None:
        limitation = (
            "target sparsity not reached by final epoch after discrete enforcement; "
            f"active={final_stats.active}, target_active={final_target_active}, "
            f"final_sparsity={final_stats.sparsity:.6f}, requested_target={final_target_requested:.6f}"
        )

    final_test_loss = None
    final_test_acc = None
    if x_test.shape[0] > 0:
        final_test_loss, final_test_acc = evaluate_model(model, x_test, y_test)

    return {
        "config": config,
        "pruning_config": pruning_config,
        "model": model,
        "history": history,
        "pruning_reports": pruning_reports,
        "dataset": dataset_desc,
        "final_total": int(final_stats.total),
        "final_active": int(final_stats.active),
        "final_sparsity": float(final_stats.sparsity),
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "final_target_sparsity": float(final_target_requested),
        "final_enforced_target_sparsity": float(final_target_enforced),
        "final_target_active": int(final_target_active),
        "target_reached": bool(target_reached),
        "target_limitation": limitation,
        "final_layer_connectivity": [asdict(item) for item in final_layer_connectivity],
        "final_hidden_connectivity": [asdict(item) for item in final_hidden_connectivity],
        "splits": {
            "train_samples": int(x_train.shape[0]),
            "val_samples": int(x_val.shape[0]),
            "test_samples": int(x_test.shape[0]),
            "n_features": int(inferred_n_features),
            "n_classes": int(inferred_n_classes),
        },
    }


def run_paired_seed_comparison(
    *,
    base_config: TrainConfig,
    pruning_config: SelfPruningConfig,
    seed_count: int,
    start_seed: int,
) -> Dict[str, object]:
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")

    per_seed = []
    for idx in range(seed_count):
        seed = start_seed + idx
        config = replace(base_config, seed=seed)

        dense = train_baseline(config, verbose=False)
        sparse = train_self_pruning(config, pruning_config, verbose=False)

        if dense["final_test_acc"] is not None and sparse["final_test_acc"] is not None:
            split = "test"
            dense_acc = float(dense["final_test_acc"])
            sparse_acc = float(sparse["final_test_acc"])
        else:
            split = "val"
            dense_acc = float(dense["history"]["val_acc"][-1])
            sparse_acc = float(sparse["history"]["val_acc"][-1])

        per_seed.append(
            {
                "seed": seed,
                "split": split,
                "dense_acc": dense_acc,
                "sparse_acc": sparse_acc,
                "accuracy_cost_vs_dense": dense_acc - sparse_acc,
                "sparse_final_sparsity": float(sparse["final_sparsity"]),
                "sparse_target_reached": bool(sparse["target_reached"]),
            }
        )

    dense_values = np.asarray([item["dense_acc"] for item in per_seed], dtype=np.float64)
    sparse_values = np.asarray([item["sparse_acc"] for item in per_seed], dtype=np.float64)
    cost_values = np.asarray(
        [item["accuracy_cost_vs_dense"] for item in per_seed],
        dtype=np.float64,
    )

    return {
        "seed_count": int(seed_count),
        "start_seed": int(start_seed),
        "split": per_seed[0]["split"] if per_seed else "",
        "dense_acc_mean": float(np.mean(dense_values)),
        "dense_acc_std": float(np.std(dense_values)),
        "sparse_acc_mean": float(np.mean(sparse_values)),
        "sparse_acc_std": float(np.std(sparse_values)),
        "accuracy_cost_mean": float(np.mean(cost_values)),
        "accuracy_cost_std": float(np.std(cost_values)),
        "per_seed": per_seed,
    }


def profile_training_step(
    config: TrainConfig,
    *,
    output_path: str | None = None,
    steps: int = 20,
    top_n: int = 20,
) -> str:
    """Profile multiple representative minibatch optimization steps."""
    _validate_config(config)

    x, y, _ = load_phase2_dataset(config)
    x_train, y_train, _, _ = train_val_split(
        x, y, val_ratio=config.val_ratio, seed=config.seed + 1
    )
    x_train, _ = standardize_train_val(x_train, x_train)

    inferred_n_features = x_train.shape[1]
    inferred_n_classes = int(np.max(y_train)) + 1
    model = MLP(
        in_features=inferred_n_features,
        hidden_sizes=[config.hidden_size],
        out_features=inferred_n_classes,
        activation=config.activation,
        rng=np.random.default_rng(config.seed + 2),
    )
    optimizer = Adam(model.parameters(), lr=config.lr)
    batch_rng = np.random.default_rng(config.seed + 3)
    x_batch, y_batch = next(
        iterate_minibatches(
            x_train,
            y_train,
            batch_size=config.batch_size,
            rng=batch_rng,
            shuffle=False,
        )
    )

    def one_step() -> None:
        optimizer.zero_grad()
        logits = model(Tensor(x_batch, requires_grad=False))
        loss = cross_entropy(logits, y_batch)
        loss.backward()
        optimizer.step()

    if steps <= 0:
        raise ValueError("steps must be positive")

    def repeated_steps() -> None:
        for _ in range(steps):
            one_step()

    profiler = cProfile.Profile()
    profiler.enable()
    repeated_steps()
    profiler.disable()

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).sort_stats("tottime")
    stats.print_stats(top_n)
    report = _sanitize_profile_report(buffer.getvalue())

    if output_path is not None:
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train dense baseline or Phase 4 self-pruning model"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--samples-per-class", type=int, default=120)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--n-features", type=int, default=64)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--dataset", choices=["digits", "synthetic"], default="digits")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument(
        "--activation", choices=["relu", "tanh", "sigmoid"], default="relu"
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write the final JSON summary",
    )
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--profile-step", action="store_true")
    parser.add_argument(
        "--profile-and-train",
        action="store_true",
        help="When used with --profile-step, continue with full training after profiling",
    )
    parser.add_argument(
        "--profile-output",
        default="experiments/results/phase2_profile.txt",
        help="Path for profile output when --profile-step is set",
    )
    parser.add_argument("--self-prune", action="store_true")
    parser.add_argument(
        "--prune-criterion",
        choices=["magnitude", "taylor"],
        default="taylor",
    )
    parser.add_argument("--target-sparsity", type=float, default=0.8)
    parser.add_argument("--prune-start-epoch", type=int, default=2)
    parser.add_argument(
        "--prune-end-epoch",
        type=int,
        default=-1,
        help="Set to -1 to use --epochs",
    )
    parser.add_argument("--prune-frequency", type=int, default=1)
    parser.add_argument("--min-survivors-per-layer", type=int, default=1)
    parser.add_argument("--min-survivors-per-row", type=int, default=0)
    parser.add_argument("--min-survivors-per-col", type=int, default=0)
    parser.add_argument("--min-survivors-per-output-col", type=int, default=1)
    parser.add_argument(
        "--skip-dense-compare",
        action="store_true",
        help="Skip dense-baseline comparison when running --self-prune",
    )
    parser.add_argument("--paired-seed-eval", action="store_true")
    parser.add_argument("--paired-seed-count", type=int, default=10)
    parser.add_argument("--paired-seed-start", type=int, default=100)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = TrainConfig(
        seed=args.seed,
        n_features=args.n_features,
        n_classes=args.n_classes,
        samples_per_class=args.samples_per_class,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        activation=args.activation,
        dataset=args.dataset,
        results_dir=args.results_dir,
    )

    if args.profile_step:
        profile_text = profile_training_step(config, output_path=args.profile_output)
        print(f"profile report written to: {args.profile_output}")
        if not args.quiet:
            print(profile_text)
        if not args.profile_and_train:
            return

    if args.self_prune:
        prune_end_epoch = config.epochs if args.prune_end_epoch == -1 else args.prune_end_epoch
        pruning_config = SelfPruningConfig(
            criterion=args.prune_criterion,
            target_sparsity=args.target_sparsity,
            prune_start_epoch=args.prune_start_epoch,
            prune_end_epoch=prune_end_epoch,
            prune_frequency=args.prune_frequency,
            min_survivors_per_layer=args.min_survivors_per_layer,
            min_survivors_per_row=args.min_survivors_per_row,
            min_survivors_per_col=args.min_survivors_per_col,
            min_survivors_per_output_col=args.min_survivors_per_output_col,
        )

        train_start = time.perf_counter()
        artifacts = train_self_pruning(
            config,
            pruning_config,
            verbose=not args.quiet,
        )
        train_seconds = time.perf_counter() - train_start
        history = artifacts["history"]

        dense_compare = None
        if not args.skip_dense_compare:
            dense_start = time.perf_counter()
            dense_artifacts = train_baseline(config, verbose=False)
            dense_seconds = time.perf_counter() - dense_start
            dense_history = dense_artifacts["history"]
            if (
                artifacts["final_test_acc"] is not None
                and dense_artifacts["final_test_acc"] is not None
            ):
                comparison_split = "test"
                dense_final_metric = float(dense_artifacts["final_test_acc"])
                sparse_final_metric = float(artifacts["final_test_acc"])
            else:
                comparison_split = "val"
                dense_final_metric = float(dense_history["val_acc"][-1])
                sparse_final_metric = float(history["val_acc"][-1])

            dense_compare = {
                "comparison_split": comparison_split,
                "dense_final_acc": dense_final_metric,
                "sparse_final_acc": sparse_final_metric,
                "accuracy_cost_vs_dense": dense_final_metric - sparse_final_metric,
                "dense_train_seconds": dense_seconds,
            }

        artifact_paths = {}
        if args.save_artifacts:
            artifact_paths = save_history_artifacts(
                history=history,
                output_dir=config.results_dir,
                prefix="phase4_self_prune",
            )

        mask_stats = {
            "final_sparsity": artifacts["final_sparsity"],
            "final_target_sparsity": artifacts["final_target_sparsity"],
            "final_enforced_target_sparsity": artifacts["final_enforced_target_sparsity"],
            "target_reached": artifacts["target_reached"],
            "target_limitation": artifacts["target_limitation"],
            "final_active": artifacts["final_active"],
            "final_target_active": artifacts["final_target_active"],
            "final_total": artifacts["final_total"],
            "layer_connectivity": artifacts["final_layer_connectivity"],
            "hidden_connectivity": artifacts["final_hidden_connectivity"],
        }

        summary = {
            "mode": "phase4_self_prune",
            "config": asdict(config),
            "pruning_config": asdict(pruning_config),
            "dataset": artifacts["dataset"],
            "splits": artifacts["splits"],
            "final_train_loss": history["train_loss"][-1],
            "final_train_acc": history["train_acc"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_val_acc": history["val_acc"][-1],
            "final_test_loss": artifacts["final_test_loss"],
            "final_test_acc": artifacts["final_test_acc"],
            "train_seconds": train_seconds,
            "artifact_paths": artifact_paths,
            "mask_summary": mask_stats,
            "pruning_reports": artifacts["pruning_reports"],
            "dense_comparison": dense_compare,
        }

        if args.paired_seed_eval:
            summary["paired_seed_comparison"] = run_paired_seed_comparison(
                base_config=config,
                pruning_config=pruning_config,
                seed_count=args.paired_seed_count,
                start_seed=args.paired_seed_start,
            )

        _emit_json_summary(summary, args.json_out)
        return

    train_start = time.perf_counter()
    artifacts = train_baseline(config, verbose=not args.quiet)
    train_seconds = time.perf_counter() - train_start
    history = artifacts["history"]

    artifact_paths = {}
    if args.save_artifacts:
        artifact_paths = save_history_artifacts(
            history=history,
            output_dir=config.results_dir,
            prefix="phase2",
        )

    profile_report = profile_training_step(config, output_path=None, steps=20, top_n=20)
    profile_lines = [line for line in profile_report.splitlines() if line.strip()]
    steps_per_epoch = int(
        np.ceil(artifacts["splits"]["train_samples"] / config.batch_size)
    )
    train_seconds_per_step = max(
        train_seconds / (len(history["epoch"]) * steps_per_epoch), 1e-6
    )
    budget = estimate_compute_budget(
        n_train=artifacts["splits"]["train_samples"],
        batch_size=config.batch_size,
        epochs=config.epochs,
        seconds_per_step=float(train_seconds_per_step),
        planned_runs=50,
    )

    summary = {
        "config": asdict(config),
        "dataset": artifacts["dataset"],
        "splits": artifacts["splits"],
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "final_test_loss": artifacts["final_test_loss"],
        "final_test_acc": artifacts["final_test_acc"],
        "train_seconds": train_seconds,
        "artifact_paths": artifact_paths,
        "compute_budget_estimate": budget,
        "profile_top_line": profile_lines[0] if profile_lines else "",
        "profile_hot_rows": profile_lines[1:6],
    }
    _emit_json_summary(summary, args.json_out)


if __name__ == "__main__":
    main()
