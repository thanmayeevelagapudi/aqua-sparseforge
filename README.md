# Aqua Challenge

NumPy-only implementation of an autodiff + training + pruning pipeline.

This repository is being built in the six-phase order agreed in planning.
Current status: **Phase 2 complete (dense baseline training + optimizer validated)**.

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

## Train Phase 2 baseline

```bash
python -m src.train --seed 42 --epochs 40 --batch-size 32 --samples-per-class 120 --hidden-size 32 --lr 0.01
```

## Profile a representative training step

```bash
python -m src.train --profile-step --profile-output experiments/results/phase2_profile.txt --quiet
```

## Current scope

- `src/tensor.py`: reverse-mode autodiff tensor engine
- `src/ops.py`: softmax and cross-entropy helpers
- `src/pruning.py`: Phase 1 mask semantics helpers (`W_eff = M * W`)
- `src/nn.py`: MLP components (Linear + activation wiring)
- `src/optim.py`: from-scratch Adam optimizer
- `src/train.py`: deterministic synthetic-dataset training loop and profiling
- `tests/test_autodiff.py`: required regression tests
- `tests/test_gradient_check.py`: finite-difference gradient checks
- `tests/test_phase1_mask_semantics.py`: masked-weight correctness tests
- `tests/test_optimizer.py`: optimizer correctness tests
- `tests/test_train.py`: training, reproducibility, and profiling tests

Structured pruning schedule and sparse evaluation are deferred to later phases.
