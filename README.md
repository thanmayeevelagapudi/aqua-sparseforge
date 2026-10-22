# AQUA Self-Pruning Network Challenge

NumPy-only implementation of an autodiff + training + pruning pipeline.

This repository is being built in the six-phase order agreed in planning.
Current status: **Phases 1-4 complete; Phase 5 evidence under final sign-off review**.

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
- `src/nn.py`: MLP components (Linear/MaskedLinear + activation wiring + sparse-aware forward path)
- `src/optim.py`: from-scratch Adam + masked-state Adam
- `src/train.py`: dense baseline training, profiling, and Phase 4 self-pruning loop
- `experiments/run_sweep.py`: Phase 5 Pareto sweep, paired statistics, and cost evidence generation
- `experiments/verify_claim.py`: machine-check verification for the committed Phase 5 quantitative claim
- `tests/test_autodiff.py`: required regression tests
- `tests/test_gradient_check.py`: finite-difference gradient checks
- `tests/test_phase1_mask_semantics.py`: masked-weight correctness tests
- `tests/test_optimizer.py`: optimizer correctness tests
- `tests/test_train.py`: training, reproducibility, and profiling tests
- `tests/test_pruning.py`: prune/revive state-reset and effective-gradient contracts
- `tests/test_phase4_pruning.py`: schedule, target-enforcement, guard, and Phase 4 integration tests
- `tests/test_sparse_inference.py`: sparse-path equivalence and sparse-cost-accounting tests
- `tests/test_phase5_sweep.py`: sweep pairing/statistics/schema helper tests

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

## Phase 5 evaluation and evidence

### Reproducible Pareto sweep command

```bash
python -m experiments.run_sweep --targets "0.0,0.5,0.75,0.9,0.95,0.97,0.98,0.99" --seeds "100,101,102,103,104,105,106,107,108,109" --epochs 20 --batch-size 32 --hidden-size 256 --lr 0.01 --test-ratio 0.1 --prune-start-epoch 2 --prune-end-epoch 20 --prune-frequency 1 --min-survivors-per-layer 1 --min-survivors-per-row 0 --min-survivors-per-col 0 --min-survivors-per-output-col 1 --latency-warmup-runs 3 --latency-timed-runs 10 --bootstrap-resamples 5000 --permutation-resamples 5000 --quiet
```

Claim verification command:

```bash
python -m experiments.verify_claim
```

### Committed Phase 5 artifacts

- `experiments/results/phase5_pareto_raw.csv`
- `experiments/results/phase5_paired_differences.csv`
- `experiments/results/phase5_pareto_plot.png`
- `experiments/results/phase5_summary.json`

### Sparse-aware inference notes

- Sparse-aware inference uses an explicit active-connection traversal path
	(`MLP.predict_logits_numpy_sparse`) and is numerically equivalent to masked
	dense forward pass within tolerance.
- Sweep result: maximum absolute dense-vs-sparse logit difference was
	`1.13686837721616e-13` (below `1e-10`).
- Cost accounting in CSV includes both:
	- theoretical linear-layer FLOP ratio (`theory_sparse_over_dense_flops`),
	- measured batch-eval latency including sparse-plan build
		(`dense_latency_batch_eval_median_ms`, `sparse_latency_batch_eval_total_median_ms`),
	- measured batch-eval sparse run-only latency with pre-built sparse plan
		(`sparse_latency_batch_eval_run_only_median_ms`),
	- measured batch-1 latency columns (`dense_latency_batch1_median_ms`,
		`sparse_latency_batch1_total_median_ms`,
		`sparse_latency_batch1_run_only_median_ms`).
- On this hardware, sparse-aware NumPy inference is slower than dense BLAS even
	at high sparsity. The dominant reason is Python-level loop/scatter overhead per
	active connection, while dense BLAS executes vectorized kernels in optimized C.
	Sparse plan build time (`sparse_csr_build_ms`) is separated from run-only
	latency to make this overhead transparent.

Examples from `phase5_pareto_raw.csv` means across both pruning methods:

- At 0% sparsity: dense eval latency is about `0.4273 ms`, sparse total is
	about `78.3924 ms`, sparse run-only is about `79.7035 ms`.
- At 99% sparsity: dense eval latency is about `0.3795 ms`, sparse total is
	about `1.2570 ms`, sparse run-only is about `1.0390 ms`.
- Batch-size-1 also remains slower for sparse (`~1.1817 ms` sparse total vs
	`~0.0298 ms` dense at 99% sparsity).

### Statistical method and limitation

- Taylor vs magnitude comparison is paired by seed and target.
- Comparison rows in `phase5_paired_differences.csv` require matched achieved
	sparsity and active counts.
- Reported uncertainty includes bootstrap CI, paired t-interval, and a paired
	sign-flip permutation p-value.
- The 0% target pairs are identical by construction and are not evidence of
	method separation.
- This 10-seed sweep shows a clear high-sparsity trade-off: mean accuracy is
	near flat through 95%, then drops at 99% (Taylor: `0.971667 -> 0.940556`,
	Magnitude: `0.971111 -> 0.885556`).

### Falsifiable quantitative claim

At achieved sparsity `0.9900232264` (`189` active of `18,944`), across `10`
paired seeds (`100-109`), Taylor achieved mean accuracy `0.940556` versus
`0.885556` for magnitude; the mean paired difference (Taylor - magnitude) was
`0.055000` with bootstrap 95% CI `[0.033889, 0.078333]` and paired t-interval
`[0.027186, 0.082814]`.

This claim is generated from committed artifacts and can be recomputed by:

```bash
python -m experiments.verify_claim
```

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
