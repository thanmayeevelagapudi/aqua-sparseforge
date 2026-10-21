"""Phase 2 baseline training and profiling entrypoint."""

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from .nn import MLP
from .ops import cross_entropy
from .optim import Adam
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
    activation: str = "relu"
    dataset: str = "digits"
    results_dir: str = "experiments/results"


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
    if config.dataset not in {"digits", "synthetic"}:
        raise ValueError("dataset must be 'digits' or 'synthetic'")


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


def standardize_train_val(
    x_train: np.ndarray,
    x_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x_train - mean) / std, (x_val - mean) / std


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
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        for idx in range(len(history["epoch"])):
            writer.writerow(
                {
                    "epoch": history["epoch"][idx],
                    "train_loss": history["train_loss"][idx],
                    "train_acc": history["train_acc"][idx],
                    "val_loss": history["val_loss"][idx],
                    "val_acc": history["val_acc"][idx],
                }
            )

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

    x_train, y_train, x_val, y_val = train_val_split(
        x,
        y,
        val_ratio=config.val_ratio,
        seed=config.seed + 1,
    )
    x_train, x_val = standardize_train_val(x_train, x_val)

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
            )

    return {
        "config": config,
        "model": model,
        "history": history,
        "dataset": dataset_desc,
        "splits": {
            "train_samples": int(x_train.shape[0]),
            "val_samples": int(x_val.shape[0]),
            "n_features": int(inferred_n_features),
            "n_classes": int(inferred_n_classes),
        },
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
    parser = argparse.ArgumentParser(description="Train Phase 2 dense baseline")
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
    parser.add_argument(
        "--activation", choices=["relu", "tanh", "sigmoid"], default="relu"
    )
    parser.add_argument("--quiet", action="store_true")
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
        "train_seconds": train_seconds,
        "artifact_paths": artifact_paths,
        "compute_budget_estimate": budget,
        "profile_top_line": profile_lines[0] if profile_lines else "",
        "profile_hot_rows": profile_lines[1:6],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
