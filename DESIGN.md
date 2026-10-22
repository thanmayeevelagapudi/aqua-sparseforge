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

- Baseline criterion: magnitude pruning using `|w_ij|`.
- Stronger criterion: first-order Taylor saliency using `|w_ij * g_ij|`.
- Production implementation in this repository uses minibatch-averaged
  first-order signal accumulated across training batches in an epoch:

  `score_ij = mean_t( abs(w_ij * g_ij^(t)) )`

  where `g_ij^(t)` is the effective-weight gradient from minibatch `t`.

- This implementation is a practical Taylor-style heuristic and does **not**
  claim universal superiority over magnitude pruning.

### First-order loss-change derivation

Consider removing one connection by setting `w_ij` to `0`.

Let `Delta w_ij = -w_ij` and write a first-order Taylor expansion:

`Delta L ~= (dL/dw_ij) * Delta w_ij = - (dL/dw_ij) * w_ij`

So the estimated absolute loss impact is:

`|Delta L| ~= |w_ij * dL/dw_ij|`

For masked training, gradients are taken with respect to effective weights,
and the same form applies with `dL/dw_eff,ij`.

Connections with smaller `|w_ij * g_ij|` are estimated to cause smaller local
loss change on removal, so they are pruned first.

### Pruning schedule and enforcement

- Schedule: cubic sparsity ramp from `0` to target sparsity:

  `s(e) = s_final * (1 - (1 - p)^3)`,

  where `p` is normalized progress between `prune_start_epoch` and
  `prune_end_epoch`.

- Why cubic: it prunes gently early (stability) and accelerates later once the
  model has adapted to earlier removals.

- Hard target enforcement: each prune step computes the exact active-parameter
  target from masks and prunes globally by saliency ranking until that active
  count is met, or reports an explicit limitation.

  Discrete enforcement uses:

  `active_target = floor((1 - s_target) * total_params)`

  which guarantees achieved mask sparsity is never below requested target.

- Layer-collapse guard: each masked layer keeps at least
  `min_survivors_per_layer` active connections.

- Row/column guards are opt-in (`min_survivors_per_row`,
  `min_survivors_per_col`, both default 0). The default structural guard is
  output-layer-only column connectivity (`min_survivors_per_output_col=1`) so
  each output class remains connected.

- Strict global row+column guards can impose a hard sparsity ceiling. For the
  `64x256 + 256x10` topology with both set to `1`, minimum active parameters
  are approximately `512` (about `97.3%` maximum sparsity).

- If guards prevent reaching the global target, the limitation is reported
  explicitly.

- Distinguish this from:
  - full-data first-order proxy `abs(w * mean(g))`
    (i.e., absolute of mean gradient)
  - diagonal-Fisher style approximations.

The implemented `mean(abs(w * g_t))` and `abs(w * mean(g_t))` are related but
not identical estimators; both should be treated as first-order proxies rather
than guaranteed dominance criteria.

The chosen score estimates sensitivity and is not a signed expected loss-change
estimator.

### Test-to-claim map (Phase 4)

| Claim | Test |
| --- | --- |
| Cubic schedule is monotonic and hits final target | `tests/test_phase4_pruning.py::test_cubic_schedule_is_monotonic_and_hits_target` |
| Global hard-target pruning reaches mask-count target when feasible | `tests/test_phase4_pruning.py::test_hard_target_pruning_reaches_requested_mask_count_when_feasible` |
| Discrete target enforcement never undershoots requested sparsity on 18,944 params | `tests/test_phase4_pruning.py::test_enforced_target_active_count_achieves_at_least_requested_sparsity` |
| Magnitude score equals absolute effective weights | `tests/test_phase4_pruning.py::test_compute_magnitude_scores_matches_absolute_effective_weights` |
| Taylor score equals abs(w_eff * grad_eff) and validates missing grad signal | `tests/test_phase4_pruning.py::test_compute_taylor_scores_matches_w_times_grad_and_requires_grad_signal` |
| Global pruning removes lowest-score connections first | `tests/test_phase4_pruning.py::test_apply_global_pruning_prunes_lowest_scores_first` |
| Taylor and magnitude can produce different rankings | `tests/test_phase4_pruning.py::test_taylor_and_magnitude_can_rank_connections_differently` |
| Per-layer guard prevents collapse and reports limitation | `tests/test_phase4_pruning.py::test_layer_guard_prevents_collapse_and_reports_limitation` |
| Row/column guard prevents dimension collapse | `tests/test_phase4_pruning.py::test_row_col_survivor_guards_prevent_dimension_collapse` |
| Output-layer-only column guard preserves class connectivity while allowing high sparsity | `tests/test_phase4_pruning.py::test_output_layer_column_guard_only_preserves_class_connectivity` |
| Achieved sparsity is mask-based (not near-zero counting) | `tests/test_phase4_pruning.py::test_sparsity_computed_from_masks_not_weight_magnitudes` |
| Self-pruning training progressively increases sparsity and keeps inactive weights zero | `tests/test_phase4_pruning.py::test_self_pruning_training_progressively_prunes_and_reports_target` |
| Prune-event active counts satisfy per-epoch floor target and schedule/frequency constraints | `tests/test_phase4_pruning.py::test_self_pruning_schedule_adheres_to_configured_target_curve` |
| Self-pruning is deterministic for a fixed seed | `tests/test_phase4_pruning.py::test_self_pruning_is_deterministic_for_fixed_seed` |
| Paired-seed summary emits expected shape on held-out split | `tests/test_phase4_pruning.py::test_paired_seed_comparison_reports_expected_shape` |
| Magnitude baseline path remains functional | `tests/test_phase4_pruning.py::test_self_pruning_magnitude_path_runs_and_prunes` |

## Phase 5 evaluation protocol

### Matched-sparsity comparison

- Phase 5 compares Taylor and magnitude using paired seeds under identical
  training/splitting configuration.
- Primary comparison rows require matched achieved sparsity and matched active
  parameter counts for each `(seed, target)` pair.
- If a pair does not match achieved sparsity/active count, it is recorded but
  excluded from paired-difference conclusions.

### Sparse-aware inference and cost accounting

- Sparse-aware inference uses explicit active-connection traversal
  (`predict_logits_numpy_sparse`) rather than dense matmul over zeroed weights.
- Latency reporting is split into:
  - sparse plan build cost (`sparse_csr_build_ms`),
  - sparse total latency including plan build,
  - sparse run-only latency with pre-built sparse plan,
  - corresponding dense baselines for both eval-batch and batch-size-1 paths.
- Numerical equivalence target: absolute dense-vs-sparse logit error <= `1e-10`
  (float64 path).
- Cost reporting includes both:
  - theoretical linear-layer arithmetic (`theory_sparse_over_dense_flops`), and
  - measured median wall-clock latency with warmup/repeats.

Theoretical arithmetic reduction is not interpreted as guaranteed speedup.
Measured latency is the performance source of truth on tested hardware.

### Statistical reporting

- Per-target paired difference is `acc_taylor - acc_magnitude`.
- Reported statistics: mean, standard deviation, bootstrap 95% CI, paired
  t-interval, and paired sign-flip permutation p-value.
- Small-seed sweeps can yield wide intervals; if CI includes zero, no
  superiority claim is made.

### Test-to-claim map (Phase 5)

| Claim | Test |
| --- | --- |
| Sparse-aware forward matches dense masked logits within strict tolerance | `tests/test_sparse_inference.py::test_sparse_predict_matches_masked_dense_forward` |
| Sparse-aware forward also matches dense path for unmasked models | `tests/test_sparse_inference.py::test_sparse_predict_matches_dense_for_unmasked_model` |
| Inference cost accounting reflects active-connection counts and reduced linear ops | `tests/test_sparse_inference.py::test_sparse_inference_cost_reflects_active_connections` |
| Sparse-aware path remains equivalent for batch-size-1 tanh/sigmoid cases | `tests/test_sparse_inference.py::test_sparse_predict_matches_masked_dense_for_batch_size_one` |
| Sparse-aware path handles all-pruned hidden layer without divergence | `tests/test_sparse_inference.py::test_sparse_predict_handles_all_pruned_hidden_layer` |
| Sparse-aware path remains equivalent on trained-size (64-256-10) ~95% sparse masks | `tests/test_sparse_inference.py::test_sparse_predict_matches_dense_for_trained_size_high_sparsity` |
| Paired-row matching logic marks matched/unmatched cases correctly | `tests/test_phase5_sweep.py::test_build_paired_rows_marks_matched_and_unmatched_pairs` |
| Paired uncertainty outputs include bootstrap, t-interval, and permutation keys | `tests/test_phase5_sweep.py::test_summarize_paired_statistics_reports_expected_keys` |
| Phase 5 CSV schema includes latency decomposition and sparse-cost columns | `tests/test_phase5_sweep.py::test_phase5_csv_schema_contains_latency_and_cost_columns` |

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
