"""Optimization algorithms for Tensor-based models."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

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
				raise ValueError("parameter grad shape does not match parameter data shape")
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
