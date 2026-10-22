"""Neural network modules built on the local Tensor autodiff engine."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np

from .pruning import MaskedParameter
from .tensor import Tensor


class Module:
    """Base class for composable model components."""

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        return []


class Sequential(Module):
    """Compose modules left-to-right."""

    def __init__(self, *modules: Module) -> None:
        self.modules = list(modules)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for module in self.modules:
            out = module(out)
        return out

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for module in self.modules:
            params.extend(module.parameters())
        return params


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Linear(Module):
    """Fully-connected affine layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        rng: np.random.Generator,
        init_mode: str = "xavier",
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if rng is None:
            raise ValueError("rng must be provided for deterministic initialization")

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        if init_mode == "xavier":
            limit = np.sqrt(6.0 / (in_features + out_features))
            weight_data = rng.uniform(
                low=-limit,
                high=limit,
                size=(in_features, out_features),
            ).astype(np.float64)
        elif init_mode == "he":
            std = np.sqrt(2.0 / in_features)
            weight_data = rng.normal(
                loc=0.0,
                scale=std,
                size=(in_features, out_features),
            ).astype(np.float64)
        else:
            raise ValueError("init_mode must be either 'xavier' or 'he'")
        self.weight = Tensor(weight_data, requires_grad=True)

        if use_bias:
            self.bias = Tensor(
                np.zeros((out_features,), dtype=np.float64), requires_grad=True
            )
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


class MaskedLinear(Linear):
    """Linear layer with explicit binary mask over weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        rng: np.random.Generator,
        init_mode: str = "xavier",
        mask: np.ndarray | None = None,
        name: str = "",
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            rng=rng,
            init_mode=init_mode,
        )
        initial_mask = (
            np.ones_like(self.weight.data, dtype=np.float64)
            if mask is None
            else np.array(mask, dtype=np.float64, copy=True)
        )
        self.masked_weight = MaskedParameter(
            tensor=self.weight,
            mask=initial_mask,
            name=name,
        )

    def forward(self, x: Tensor) -> Tensor:
        output = x @ self.masked_weight.effective_tensor()
        if self.bias is not None:
            output = output + self.bias
        return output

    def masked_parameters(self) -> List[MaskedParameter]:
        return [self.masked_weight]


class MLP(Module):
    """Simple multi-layer perceptron classifier."""

    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int],
        out_features: int,
        *,
        activation: str = "relu",
        masked: bool = False,
        rng: np.random.Generator,
    ) -> None:
        if out_features <= 0:
            raise ValueError("out_features must be positive")
        if any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive values")
        if rng is None:
            raise ValueError("rng must be provided for deterministic initialization")

        self.activation = activation
        self.masked = masked
        self.layers: List[Linear] = []
        self._activation_module = self._build_activation_module(activation)

        layer_sizes = [in_features, *hidden_sizes, out_features]

        for idx in range(len(layer_sizes) - 1):
            layer_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint32))
            layer_rng = np.random.default_rng(layer_seed)
            is_hidden_transition = idx < len(layer_sizes) - 2
            if activation == "relu" and is_hidden_transition:
                init_mode = "he"
            else:
                init_mode = "xavier"
            if masked:
                layer = MaskedLinear(
                    layer_sizes[idx],
                    layer_sizes[idx + 1],
                    use_bias=True,
                    rng=layer_rng,
                    init_mode=init_mode,
                    name=f"mlp_layer_{idx}.weight",
                )
            else:
                layer = Linear(
                    layer_sizes[idx],
                    layer_sizes[idx + 1],
                    use_bias=True,
                    rng=layer_rng,
                    init_mode=init_mode,
                )
            self.layers.append(layer)

    def _build_activation_module(self, activation: str) -> Module:
        if activation == "relu":
            return ReLU()
        if activation == "tanh":
            return Tanh()
        if activation == "sigmoid":
            return Sigmoid()
        raise ValueError(f"unsupported activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for idx, layer in enumerate(self.layers):
            out = layer(out)
            if idx < len(self.layers) - 1:
                out = self._activation_module(out)
        return out

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def masked_parameters(self) -> List[MaskedParameter]:
        masked: List[MaskedParameter] = []
        for layer in self.layers:
            getter = getattr(layer, "masked_parameters", None)
            if callable(getter):
                masked.extend(getter())
        return masked

    def _apply_activation_numpy(self, values: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(values, 0.0)
        if self.activation == "tanh":
            return np.tanh(values)
        if self.activation == "sigmoid":
            nonnegative = values >= 0.0
            result = np.empty_like(values)
            exp_neg = np.exp(-values[nonnegative])
            result[nonnegative] = 1.0 / (1.0 + exp_neg)
            exp_pos = np.exp(values[~nonnegative])
            result[~nonnegative] = exp_pos / (1.0 + exp_pos)
            return result
        raise ValueError(f"unsupported activation: {self.activation}")

    def _build_sparse_layer_specs(self) -> List[Dict[str, np.ndarray | int | None]]:
        specs: List[Dict[str, np.ndarray | int | None]] = []
        for layer in self.layers:
            if isinstance(layer, MaskedLinear):
                weight_data = layer.masked_weight.effective_numpy()
            else:
                weight_data = np.array(layer.weight.data, dtype=np.float64, copy=True)

            in_features, out_features = weight_data.shape
            row_indices, col_indices = np.nonzero(weight_data)

            if row_indices.size == 0:
                row_ptr = np.zeros((in_features + 1,), dtype=np.int64)
                col_idx = np.zeros((0,), dtype=np.int64)
                values = np.zeros((0,), dtype=np.float64)
            else:
                row_counts = np.bincount(row_indices, minlength=in_features)
                row_ptr = np.zeros((in_features + 1,), dtype=np.int64)
                row_ptr[1:] = np.cumsum(row_counts)
                col_idx = np.asarray(col_indices, dtype=np.int64)
                values = np.asarray(weight_data[row_indices, col_indices], dtype=np.float64)

            bias_data = None
            if layer.bias is not None:
                bias_data = np.asarray(layer.bias.data, dtype=np.float64)

            specs.append(
                {
                    "in_features": int(in_features),
                    "out_features": int(out_features),
                    "row_ptr": row_ptr,
                    "col_idx": col_idx,
                    "values": values,
                    "bias": bias_data,
                }
            )
        return specs

    def _sparse_linear_forward_numpy(
        self,
        x: np.ndarray,
        *,
        out_features: int,
        row_ptr: np.ndarray,
        col_idx: np.ndarray,
        values: np.ndarray,
        bias: np.ndarray | None,
    ) -> np.ndarray:
        out = np.zeros((x.shape[0], out_features), dtype=np.float64)
        for row in range(x.shape[1]):
            start = int(row_ptr[row])
            end = int(row_ptr[row + 1])
            if start == end:
                continue
            x_col = x[:, row]
            for idx in range(start, end):
                out[:, int(col_idx[idx])] += x_col * float(values[idx])
        if bias is not None:
            out = out + bias
        return out

    def predict_logits_numpy(self, x: np.ndarray) -> np.ndarray:
        """Compute logits without building an autograd graph."""
        out = np.array(x, dtype=np.float64, copy=True)
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, MaskedLinear):
                weight_data = layer.masked_weight.effective_numpy()
            else:
                weight_data = layer.weight.data

            out = out @ weight_data
            if layer.bias is not None:
                out = out + layer.bias.data
            if idx < len(self.layers) - 1:
                out = self._apply_activation_numpy(out)
        return out

    def predict_logits_numpy_sparse(self, x: np.ndarray) -> np.ndarray:
        """Sparse-aware inference that skips arithmetic on removed connections."""
        layer_specs = self.build_sparse_inference_plan()
        return self.predict_logits_numpy_sparse_with_plan(x, layer_specs=layer_specs)

    def build_sparse_inference_plan(self) -> List[Dict[str, np.ndarray | int | None]]:
        """Build a reusable sparse layer representation for inference benchmarking."""
        return self._build_sparse_layer_specs()

    def predict_logits_numpy_sparse_with_plan(
        self,
        x: np.ndarray,
        *,
        layer_specs: Sequence[Dict[str, np.ndarray | int | None]],
    ) -> np.ndarray:
        """Sparse-aware inference using a pre-built sparse layer representation."""
        out = np.array(x, dtype=np.float64, copy=True)

        for idx, spec in enumerate(layer_specs):
            out = self._sparse_linear_forward_numpy(
                out,
                out_features=int(spec["out_features"]),
                row_ptr=np.asarray(spec["row_ptr"], dtype=np.int64),
                col_idx=np.asarray(spec["col_idx"], dtype=np.int64),
                values=np.asarray(spec["values"], dtype=np.float64),
                bias=spec["bias"] if isinstance(spec["bias"], np.ndarray) else None,
            )
            if idx < len(layer_specs) - 1:
                out = self._apply_activation_numpy(out)
        return out

    def estimate_inference_cost(self, *, batch_size: int, sparse: bool) -> Dict[str, int]:
        """Estimate linear-operation cost for one forward pass.

        This accounts for multiply-adds and bias additions for linear layers.
        Activation op counts are intentionally excluded from this estimate.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        linear_mac = 0
        linear_flops = 0
        active_weights = 0
        total_weights = 0

        for layer in self.layers:
            in_features, out_features = layer.weight.data.shape
            layer_total = int(in_features * out_features)
            total_weights += layer_total

            if sparse and isinstance(layer, MaskedLinear):
                layer_active = int(np.count_nonzero(layer.masked_weight.mask))
            else:
                layer_active = layer_total

            active_weights += layer_active
            layer_mac = int(batch_size * layer_active)
            linear_mac += layer_mac

            bias_adds = int(batch_size * out_features) if layer.bias is not None else 0
            linear_flops += int((2 * layer_mac) + bias_adds)

        return {
            "batch_size": int(batch_size),
            "linear_mac": int(linear_mac),
            "linear_flops": int(linear_flops),
            "active_weights": int(active_weights),
            "total_weights": int(total_weights),
        }


def flatten_parameters(modules: Iterable[Module]) -> List[Tensor]:
    """Collect parameters from multiple modules."""
    params: List[Tensor] = []
    for module in modules:
        params.extend(module.parameters())
    return params


def flatten_masked_parameters(modules: Iterable[Module]) -> List[MaskedParameter]:
    """Collect masked parameters from modules that expose masked_parameters()."""
    masked: List[MaskedParameter] = []
    for module in modules:
        getter = getattr(module, "masked_parameters", None)
        if callable(getter):
            masked.extend(getter())
    return masked
