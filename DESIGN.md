# Design Notes

This document tracks locked decisions and implementation contracts.

## Locked build order

1. Tensor and autodiff engine.
2. Baseline MLP and training loop.
3. Masking + optimizer state policy.
4. Saliency + pruning schedule and guards.
5. Evaluation and reporting.
6. Finalization and reproduction docs.

## Locked optimizer-state policy (for Phase 3)

- Masks define effective parameters: `W_eff = M * W`.
- Per-element Adam state consists of first moment, second moment, and age.
- Pruned entries are hard-clamped to zero and excluded from updates.
- Prune/revive transitions reset per-element Adam moments and age deterministically.
- Revival initialization uses zeros; revived entries begin with age zero and become
  age one on their first active optimization step.
- Effective-position gradients can be recorded separately from underlying
  gradients to support future regrowth logic while preserving frozen inactive
  updates (`dL/dW = M * dL/dW_eff`).

### Gradient-channel derivation

For masked weights, the forward path uses:

`W_eff = M * W`

Let `G_eff = dL/dW_eff`. Chain rule gives:

`dL/dW = dL/dW_eff * dW_eff/dW = G_eff * M`

So two gradient channels are intentionally distinguished:

- underlying-weight gradient (`dL/dW`): used for parameter updates and must be
  zero on inactive entries;
- effective-position gradient (`dL/dW_eff`): can be non-zero on inactive
  entries and is preserved for regrowth/scoring policies.

Implementation note:

- `MaskedParameter.effective_tensor()` uses a custom backward that captures
  `dL/dW_eff` directly from the intermediate effective tensor, then applies the
  mask before accumulating into the underlying parameter gradient.

### Transition semantics

- Transition events are recorded at mutation time (`prune`, `revive`, `set_mask`),
  not inferred later by mask snapshot diffing.
- `MaskedAdam` consumes transition counters at `step()` time and resets moments
  and age for exactly the changed entries.
- This avoids stale-state reuse when prune+revive happens between optimizer steps.

### Test-to-claim map (Phase 3)

| Claim | Test |
| --- | --- |
| Effective forward uses `W_eff` only | `tests/test_pruning.py::test_masked_linear_forward_uses_effective_weights_only` |
| Chain rule `dL/dW = M * dL/dW_eff` | `tests/test_pruning.py::test_masked_parameter_gradient_matches_effective_chain_rule` |
| Inactive weights remain hard-zero under updates | `tests/test_pruning.py::test_maskedadam_keeps_inactive_weights_and_grads_zero` |
| Prune then step resets moments/age | `tests/test_pruning.py::test_prune_then_revive_resets_state_and_restarts_age_from_one` |
| Prune+revive between steps still resets age | `tests/test_pruning.py::test_prune_then_revive_without_intervening_step_resets_age` |
| `dL/dW_eff` is computed (including inactive entries) | `tests/test_pruning.py::test_effective_position_grad_is_computed_for_inactive_connections` |
| Effective-grad buffer refreshes after pruning | `tests/test_pruning.py::test_effective_position_grad_refreshes_after_pruning` |
| Active updates use fresh gradient, not stale buffer | `tests/test_pruning.py::test_revived_active_update_uses_fresh_grad_not_stale_effective_buffer` |
| No double-managed tensor updates in mixed optimizer setup | `tests/test_pruning.py::test_maskedadam_rejects_double_managed_dense_tensor` |

## Locked saliency policy (for Phase 4)

- Use per-example first-order Taylor sensitivity aggregation:

  `score_ij = mean_b( abs(w_ij * g_bij) )`

- Distinguish this from:
  - batch-gradient proxy `abs(w * mean(g))`
  - diagonal-Fisher style approximations.

Implementation constraint for this repository:

- The current engine naturally produces batch-mean gradients for a scalar loss.
- To implement per-example `g_bij` without redesigning autodiff internals,
  Phase 4 will compute Taylor scores via deterministic microbatch size `1`
  gradient accumulation (or equivalently, a per-example loop), then aggregate
  `mean_b(abs(w_ij * g_bij))`.

The chosen score estimates sensitivity and is not a signed expected loss-change
estimator.

## Testing contract

- Do not claim a test passed unless it has been executed.
- Every phase must ship deterministic tests with explicit expected values or invariants.

## Phase 1 autodiff contracts

### Backward semantics

- Leaf tensors (`requires_grad=True` and no parents) accumulate gradients across
  separate `backward()` calls.
- Intermediate tensors reset their gradients at the start of each backward pass.
- Calling `zero_grad()` on a leaf resets that leaf's stored gradient to zeros.
- Non-scalar outputs must receive an explicit upstream gradient with matching
  shape.

### Deterministic graph traversal

- Parent links are stored in a deterministic ordered tuple with deduplication.
- Topological ordering uses iterative DFS (not Python recursion), preventing
  recursion-limit failures on deep graphs.
- Reproducibility tests compare identical runs across separate processes.
  This is intended to verify deterministic traversal and accumulation order, not
  to claim guaranteed bitwise identity across different hardware or BLAS setups.

### Numerical stability policies

- Cross-entropy uses stable log-softmax / log-sum-exp algebra, not
  `log(softmax + eps)`.
- Sigmoid uses a branch-stable implementation that avoids overflow in `exp`
  for large-magnitude inputs.

### Derivative edge cases

- Some mathematically valid forward operations can have undefined or infinite
  derivatives at specific points. Example: `d/dx sqrt(x) = 1/(2*sqrt(x))`, which
  is infinite at `x = 0`.
- Tests and gradient checks should avoid nondifferentiable or singular points
  unless the behavior is explicitly being characterized.

### Tensor data ownership

- `Tensor(np_array)` copies the input array into internal storage.
- External arrays and tensor storage are intentionally not aliased, avoiding
  accidental shared mutation between model parameters, masks, and caller-owned
  arrays.

## Phase 2 baseline contracts

### Architecture

- Dense baseline model is a one-hidden-layer MLP:
  `Linear(n_features -> hidden_size) -> activation -> Linear(hidden_size -> n_classes)`.
- Layer implementation is composable and parameterized through `Module`, `Linear`,
  and `MLP` classes.

### Initialization policy

- Hidden ReLU layers use He initialization with
  `std = sqrt(2 / fan_in)`.
- Output layer uses Xavier initialization.
- Biases start at zero.
- Rationale: He initialization better preserves forward signal variance under
  ReLU nonlinearities; Xavier is retained for the linear output layer.

### Optimization and stability

- Optimizer is from-scratch Adam with explicit bias correction.
- Each minibatch step enforces finite-loss and finite-gradient checks before
  applying updates.
- Training loop raises explicit errors for NaN/Inf loss, gradient, or parameter
  values.

### Dataset and reproducibility

- Phase 2 uses `sklearn.datasets.load_digits` (1,797 samples, 64 features,
  10 classes). Scikit-learn is used only for dataset loading.
- Deterministic seeds are used for:
  - dataset split,
  - train/validation split,
  - model initialization,
  - minibatch shuffling.
- Reproducibility tests run two independent trainings with the same config and
  assert identical metric histories.

### Practical compute budget

- Default baseline command budget:
  - 1,797 examples total (`load_digits`)
  - `hidden_size=256`
  - `epochs=30`
  - `batch_size=32`
- Why this budget:
  - enough optimization steps to reliably converge on `load_digits`,
  - short wall-clock runtime on CPU for repeated local validation,
  - small memory footprint, enabling deterministic reruns during debugging.

The training CLI also prints a budget estimate in terms of:

`steps_per_epoch × epochs × planned_runs × seconds_per_step`.

For the default baseline run (`load_digits`, 30 epochs, batch size 32), the
current measured estimate is approximately:

- `steps_per_epoch = 45`
- `steps_per_run = 1,350`
- `planned_runs = 50`
- `estimated_total_seconds ≈ 144s`

### Profiling evidence

- A representative training workload can be profiled with `cProfile` via
  `src.train.profile_training_step`.
- Profiling output is written to a user-selected text path and can be committed
  as evidence for bottleneck analysis in later phases.
- Profiling is run across multiple repeated steps and sorted by `tottime`
  (self time) so bottleneck conclusions are tied to actual measured execution,
  not assumptions. Saved output is sanitized to avoid machine-specific path
  leakage.
- In the current measured profile (`20` repeated steps on the default digits
  workload), top hotspots include:
  - `src/optim.py:Adam.step` (about `0.007s` self time)
  - `src/tensor.py` backward kernels (e.g., `_backward` around `0.003s` self time)
  These findings are based on the generated profile artifact, not estimates.
