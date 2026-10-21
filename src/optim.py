"""Optimization algorithms for Tensor-based models."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from .pruning import MaskedParameter
from .tensor import Tensor


class Adam:
    """Adam optimizer implemented from scratch for Tensor parameters."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("betas must be in [0, 1)")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")

        self.params: List[Tensor] = [param for param in params if param.requires_grad]
        if not self.params:
            raise ValueError("Adam requires at least one trainable parameter")

        self.lr = float(lr)
        self.betas = (float(beta1), float(beta2))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.step_count = 0

        self._m = [np.zeros_like(param.data) for param in self.params]
        self._v = [np.zeros_like(param.data) for param in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        self.step_count += 1
        beta1, beta2 = self.betas

        beta1_correction = 1.0 - beta1**self.step_count
        beta2_correction = 1.0 - beta2**self.step_count

        for idx, param in enumerate(self.params):
            if param.grad is None:
                continue
            if param.grad.shape != param.data.shape:
                raise ValueError(
                    "parameter grad shape does not match parameter data shape"
                )
            if not np.isfinite(param.grad).all():
                raise ValueError("encountered non-finite gradient values")

            grad = param.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * param.data

            self._m[idx] = beta1 * self._m[idx] + (1.0 - beta1) * grad
            self._v[idx] = beta2 * self._v[idx] + (1.0 - beta2) * (grad * grad)

            m_hat = self._m[idx] / beta1_correction
            v_hat = self._v[idx] / beta2_correction

            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            param.data -= update


class MaskedAdam:
    """Adam variant for masked parameters with per-element age counters.

    Policy:
    - Inactive entries are always clamped to zero and never updated.
    - Pruned and revived entries reset first/second moments and age.
    - Revived entries start from zero with age 0; their first update uses
      age 1 bias correction on the next optimizer step.
    """

    def __init__(
        self,
        masked_params: Iterable[MaskedParameter],
        *,
        dense_params: Iterable[Tensor] | None = None,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("betas must be in [0, 1)")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")

        self.masked_params = list(masked_params)
        if not self.masked_params:
            raise ValueError("MaskedAdam requires at least one masked parameter")

        masked_tensor_ids = set()
        for masked_param in self.masked_params:
            tensor_id = id(masked_param.tensor)
            if tensor_id in masked_tensor_ids:
                raise ValueError(
                    "duplicate masked parameter tensor detected in masked_params"
                )
            masked_tensor_ids.add(tensor_id)

        self.dense_params = []
        dense_seen = set()
        for dense_param in dense_params or []:
            if not dense_param.requires_grad:
                continue
            tensor_id = id(dense_param)
            if tensor_id in masked_tensor_ids:
                raise ValueError(
                    "dense_params contains a tensor already managed by masked_params"
                )
            if tensor_id in dense_seen:
                raise ValueError("duplicate dense parameter tensor detected")
            dense_seen.add(tensor_id)
            self.dense_params.append(dense_param)

        self.lr = float(lr)
        self.betas = (float(beta1), float(beta2))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.step_count = 0

        self._masked_index = {
            id(masked_param): idx for idx, masked_param in enumerate(self.masked_params)
        }
        self._seen_transition_counts = [
            masked_param.transition_counts for masked_param in self.masked_params
        ]

        self._m_masked = [np.zeros_like(masked_param.tensor.data) for masked_param in self.masked_params]
        self._v_masked = [np.zeros_like(masked_param.tensor.data) for masked_param in self.masked_params]
        self._age_masked = [
            np.zeros_like(masked_param.tensor.data, dtype=np.int64)
            for masked_param in self.masked_params
        ]

        self._m_dense = [np.zeros_like(param.data) for param in self.dense_params]
        self._v_dense = [np.zeros_like(param.data) for param in self.dense_params]

    def zero_grad(self) -> None:
        for masked_param in self.masked_params:
            masked_param.tensor.zero_grad()
            masked_param.clear_effective_position_grad()
        for dense_param in self.dense_params:
            dense_param.zero_grad()

    def set_effective_position_grad(
        self,
        masked_param: MaskedParameter,
        grad: np.ndarray,
    ) -> None:
        masked_param.set_effective_position_grad(grad)

    def state_for(self, masked_param: MaskedParameter) -> dict:
        if id(masked_param) not in self._masked_index:
            raise ValueError("masked parameter is not managed by this optimizer")
        idx = self._masked_index[id(masked_param)]
        return {
            "m": self._m_masked[idx].copy(),
            "v": self._v_masked[idx].copy(),
            "age": self._age_masked[idx].copy(),
            "mask": masked_param.mask.copy(),
        }

    def _sync_transition_resets(self, idx: int, masked_param: MaskedParameter) -> None:
        seen_counts = self._seen_transition_counts[idx]
        current_counts = masked_param.transition_counts
        changed = current_counts != seen_counts

        if np.any(changed):
            self._m_masked[idx][changed] = 0.0
            self._v_masked[idx][changed] = 0.0
            self._age_masked[idx][changed] = 0

        self._seen_transition_counts[idx] = current_counts

    def _masked_step(self) -> None:
        beta1, beta2 = self.betas

        for idx, masked_param in enumerate(self.masked_params):
            self._sync_transition_resets(idx, masked_param)

            tensor = masked_param.tensor
            mask = masked_param.mask
            active = mask == 1.0
            inactive = ~active

            if tensor.grad is None:
                underlying_grad = np.zeros_like(tensor.data)
            else:
                if tensor.grad.shape != tensor.data.shape:
                    raise ValueError(
                        "parameter grad shape does not match parameter data shape"
                    )
                if not np.isfinite(tensor.grad).all():
                    raise ValueError("encountered non-finite gradient values")
                underlying_grad = np.array(tensor.grad, dtype=np.float64, copy=True)

            if masked_param.effective_position_grad is None:
                effective_position_grad = np.zeros_like(tensor.data)
                effective_position_grad[active] = underlying_grad[active]
            else:
                provided_effective_grad = np.array(
                    masked_param.effective_position_grad,
                    dtype=np.float64,
                    copy=True,
                )
                if provided_effective_grad.shape != tensor.data.shape:
                    raise ValueError(
                        "effective_position_grad shape does not match parameter shape"
                    )
                if not np.isfinite(provided_effective_grad).all():
                    raise ValueError("encountered non-finite effective-position gradient")
                effective_position_grad = provided_effective_grad

            grad_for_update = np.zeros_like(tensor.data)
            grad_for_update[active] = underlying_grad[active]
            if self.weight_decay != 0.0:
                grad_for_update[active] += self.weight_decay * tensor.data[active]

            m = self._m_masked[idx]
            v = self._v_masked[idx]
            age = self._age_masked[idx]

            age[active] += 1
            age[inactive] = 0

            m[active] = beta1 * m[active] + (1.0 - beta1) * grad_for_update[active]
            v[active] = beta2 * v[active] + (1.0 - beta2) * (
                grad_for_update[active] * grad_for_update[active]
            )
            m[inactive] = 0.0
            v[inactive] = 0.0

            if np.any(active):
                active_age = age[active].astype(np.float64)
                denom_m = 1.0 - np.power(beta1, active_age)
                denom_v = 1.0 - np.power(beta2, active_age)
                m_hat = m[active] / denom_m
                v_hat = v[active] / denom_v
                update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                tensor.data[active] -= update

            tensor.data[inactive] = 0.0
            masked_param.set_effective_position_grad(effective_position_grad)

    def _dense_step(self) -> None:
        if not self.dense_params:
            return

        beta1, beta2 = self.betas
        beta1_correction = 1.0 - beta1**self.step_count
        beta2_correction = 1.0 - beta2**self.step_count

        for idx, param in enumerate(self.dense_params):
            if param.grad is None:
                continue
            if param.grad.shape != param.data.shape:
                raise ValueError("parameter grad shape does not match parameter data shape")
            if not np.isfinite(param.grad).all():
                raise ValueError("encountered non-finite gradient values")

            grad = np.array(param.grad, dtype=np.float64, copy=True)
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * param.data

            self._m_dense[idx] = beta1 * self._m_dense[idx] + (1.0 - beta1) * grad
            self._v_dense[idx] = beta2 * self._v_dense[idx] + (1.0 - beta2) * (grad * grad)

            m_hat = self._m_dense[idx] / beta1_correction
            v_hat = self._v_dense[idx] / beta2_correction
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            param.data -= update

    def step(self) -> None:
        self.step_count += 1
        self._masked_step()
        self._dense_step()
