# AQUA Self-Pruning Network Challenge

NumPy-only implementation of an autodiff + training + pruning pipeline.

This repository is being built in the six-phase order agreed in planning.
Current status: **Phases 1-4 complete (autodiff + dense baseline + controlled self-pruning)**.

## Environment

- Python 3.10+
- NumPy core
- Pytest for validation
- Matplotlib for Phase 2 learning-curve artifacts
- Scikit-learn used only for loading `load_digits`

## Install

```bash
python -m pip install -r requirements.txt
```

## Run tests

```bash
python -m pytest -q
```

## Train Phase 2 baseline

```bash
python -m src.train --dataset digits --seed 42 --epochs 30 --batch-size 32 --hidden-size 256 --lr 0.01 --save-artifacts
```

This command uses `sklearn.datasets.load_digits` (offline dataset packaged with
scikit-learn: 1,797 samples, 64 features, 10 classes). Scikit-learn is used
only for loading the dataset.

For the synthetic dataset path, `--samples-per-class` is respected directly.

The default compute budget is intentionally bounded for reproducibility:

- `epochs=30`
- `batch_size=32`
- one hidden layer (`hidden_size=256`)

These settings are large enough to produce a meaningful dense baseline while
remaining practical for repeated local verification.

The run writes:

- `experiments/results/phase2_history.csv`
- `experiments/results/phase2_learning_curve.png`

## Profile a representative training step

```bash
python -m src.train --profile-step --profile-output experiments/results/phase2_profile.txt --quiet
```

By default, `--profile-step` runs profiling only and exits.
Add `--profile-and-train` to run full training after profiling.

The profiling output is generated from repeated real training steps using the
same implementation and sorted by self time (`tottime`) to surface actual
bottlenecks.

## Current scope

- `src/tensor.py`: reverse-mode autodiff tensor engine
- `src/ops.py`: softmax and cross-entropy helpers
- `src/pruning.py`: mask semantics, saliency scoring, schedules, and hard-target pruning
- `src/nn.py`: MLP components (Linear/MaskedLinear + activation wiring)
- `src/optim.py`: from-scratch Adam + masked-state Adam
- `src/train.py`: dense baseline training, profiling, and Phase 4 self-pruning loop
- `tests/test_autodiff.py`: required regression tests
- `tests/test_gradient_check.py`: finite-difference gradient checks
- `tests/test_phase1_mask_semantics.py`: masked-weight correctness tests
- `tests/test_optimizer.py`: optimizer correctness tests
- `tests/test_train.py`: training, reproducibility, and profiling tests
- `tests/test_pruning.py`: prune/revive state-reset and effective-gradient contracts
- `tests/test_phase4_pruning.py`: schedule, target-enforcement, guard, and Phase 4 integration tests

## Phase 4 self-pruning command

```bash
python -m src.train --self-prune --dataset digits --seed 42 --epochs 30 --batch-size 32 --hidden-size 256 --lr 0.01 --prune-criterion taylor --target-sparsity 0.80 --prune-start-epoch 2 --prune-end-epoch 30 --prune-frequency 1 --min-survivors-per-layer 1 --min-survivors-per-output-col 1 --test-ratio 0.1 --save-artifacts --json-out experiments/results/phase4_self_prune_summary.json
```

This command is deterministic for a fixed seed and prints a JSON summary with:

- final sparse accuracy,
- final held-out test accuracy (when `--test-ratio > 0`),
- achieved mask-based sparsity,
- whether target sparsity was reached (or explicit limitation text),
- dense-baseline comparison and `accuracy_cost_vs_dense`.

By default, `--self-prune` also runs a dense baseline comparison pass.
Use `--skip-dense-compare` to skip that pass.

Training logs are written to stderr; final summary JSON is written to stdout.
Use `--json-out <path>` to additionally persist the summary JSON to disk.

Row/column guards are opt-in and disabled by default (`--min-survivors-per-row 0`
and `--min-survivors-per-col 0`). By default, only output-layer column
connectivity is guarded (`--min-survivors-per-output-col 1`) so all class logits
remain connected while still allowing very high global sparsity targets.

If you enable strict row+column guards globally (for example both set to 1),
they can impose a hard sparsity ceiling. For the 64x256 + 256x10 architecture,
that floor is roughly 512 active weights, i.e. about 97.3% maximum sparsity.

## Paired-seed comparison (recommended before Phase 5)

```bash
python -m src.train --self-prune --dataset digits --epochs 30 --batch-size 32 --hidden-size 256 --lr 0.01 --prune-criterion taylor --target-sparsity 0.80 --prune-start-epoch 2 --prune-end-epoch 30 --prune-frequency 1 --min-survivors-per-layer 1 --min-survivors-per-output-col 1 --test-ratio 0.1 --paired-seed-eval --paired-seed-count 10 --paired-seed-start 100 --quiet --json-out experiments/results/phase4_paired_seed_summary.json
```

This emits per-seed dense/sparse metrics and aggregate mean/std on the held-out
test split. For Phase 5, run this under both guard settings:

- output-column-only guard (default),
- strict row+column guards (opt-in).

## Phase 3 masked-optimizer usage

```python
import numpy as np
from src.nn import MLP
from src.optim import MaskedAdam

rng = np.random.default_rng(0)
model = MLP(in_features=64, hidden_sizes=[128], out_features=10, masked=True, rng=rng)
optimizer = MaskedAdam(
	masked_params=model.masked_parameters(),
	dense_params=[layer.bias for layer in model.layers if layer.bias is not None],
	lr=1e-2,
)
```

`MaskedAdam` enforces:

- inactive weights stay exactly zero,
- per-element moments and age reset on prune/revive transitions,
- `dL/dW_eff` remains accessible via `MaskedParameter.effective_position_grad`.
