"""Neural network modules built on the local Tensor autodiff engine."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .tensor import Tensor


class Module:
	"""Base class for composable model components."""

	def forward(self, x: Tensor) -> Tensor:
		raise NotImplementedError

	def __call__(self, x: Tensor) -> Tensor:
		return self.forward(x)

	def parameters(self) -> List[Tensor]:
		return []


class Linear(Module):
	"""Fully-connected affine layer."""

	def __init__(
		self,
		in_features: int,
		out_features: int,
		*,
		use_bias: bool = True,
		rng: np.random.Generator | None = None,
	) -> None:
		if in_features <= 0 or out_features <= 0:
			raise ValueError("in_features and out_features must be positive")

		self.in_features = in_features
		self.out_features = out_features
		self.use_bias = use_bias

		rng = rng if rng is not None else np.random.default_rng()

		# Xavier uniform keeps activation variance stable for affine layers.
		limit = np.sqrt(6.0 / (in_features + out_features))
		weight_data = rng.uniform(
			low=-limit,
			high=limit,
			size=(in_features, out_features),
		).astype(np.float64)
		self.weight = Tensor(weight_data, requires_grad=True)

		if use_bias:
			self.bias = Tensor(np.zeros((out_features,), dtype=np.float64), requires_grad=True)
		else:
			self.bias = None

	def forward(self, x: Tensor) -> Tensor:
		output = x @ self.weight
		if self.bias is not None:
			output = output + self.bias
		return output

	def parameters(self) -> List[Tensor]:
		params = [self.weight]
		if self.bias is not None:
			params.append(self.bias)
		return params


class MLP(Module):
	"""Simple multi-layer perceptron classifier."""

	def __init__(
		self,
		in_features: int,
		hidden_sizes: Sequence[int],
		out_features: int,
		*,
		activation: str = "relu",
		rng: np.random.Generator | None = None,
	) -> None:
		if out_features <= 0:
			raise ValueError("out_features must be positive")
		if any(size <= 0 for size in hidden_sizes):
			raise ValueError("hidden_sizes must contain positive values")

		self.activation = activation
		self.layers: List[Linear] = []

		layer_sizes = [in_features, *hidden_sizes, out_features]
		rng = rng if rng is not None else np.random.default_rng()

		for idx in range(len(layer_sizes) - 1):
			layer_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
			layer_rng = np.random.default_rng(layer_seed)
			self.layers.append(
				Linear(
					layer_sizes[idx],
					layer_sizes[idx + 1],
					use_bias=True,
					rng=layer_rng,
				)
			)

	def _activation(self, tensor: Tensor) -> Tensor:
		if self.activation == "relu":
			return tensor.relu()
		if self.activation == "tanh":
			return tensor.tanh()
		if self.activation == "sigmoid":
			return tensor.sigmoid()
		raise ValueError(f"unsupported activation: {self.activation}")

	def forward(self, x: Tensor) -> Tensor:
		out = x
		for idx, layer in enumerate(self.layers):
			out = layer(out)
			if idx < len(self.layers) - 1:
				out = self._activation(out)
		return out

	def parameters(self) -> List[Tensor]:
		params: List[Tensor] = []
		for layer in self.layers:
			params.extend(layer.parameters())
		return params


def flatten_parameters(modules: Iterable[Module]) -> List[Tensor]:
	"""Collect parameters from multiple modules."""
	params: List[Tensor] = []
	for module in modules:
		params.extend(module.parameters())
	return params
