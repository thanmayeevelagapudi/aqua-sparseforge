# Aqua Challenge

NumPy-only implementation of an autodiff + training + pruning pipeline.

This repository is being built in the six-phase order agreed in planning.
Current status: **Phase 1 complete (Tensor engine + core tests validated)**.

## Environment

- Python 3.10+
- NumPy core
- Pytest for validation
- Matplotlib and scikit-learn reserved for later phases

## Install

```bash
python -m pip install -r requirements.txt
```

## Run tests

```bash
python -m pytest -q
```

## Current scope

- `src/tensor.py`: reverse-mode autodiff tensor engine
- `src/ops.py`: softmax and cross-entropy helpers
- `src/pruning.py`: Phase 1 mask semantics helpers (`W_eff = M * W`)
- `tests/test_autodiff.py`: required regression tests
- `tests/test_gradient_check.py`: finite-difference gradient checks
- `tests/test_phase1_mask_semantics.py`: masked-weight correctness tests

Optimizer, pruning, and end-to-end training modules are intentionally deferred to
later phases.
