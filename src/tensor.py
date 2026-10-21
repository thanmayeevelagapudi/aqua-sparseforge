"""Minimal reverse-mode autodiff tensor engine built on NumPy."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[float, int, Sequence[float], np.ndarray]


def _to_numpy(data: ArrayLike) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data.astype(np.float64, copy=True)
    return np.array(data, dtype=np.float64, copy=True)


def _ordered_unique_tensors(children: Tuple["Tensor", ...]) -> Tuple["Tensor", ...]:
    """Deduplicate tensor parents while preserving first-seen order."""
    ordered = []
    seen_ids = set()
    for child in children:
        child_id = id(child)
        if child_id in seen_ids:
            continue
        seen_ids.add(child_id)
        ordered.append(child)
    return tuple(ordered)


def _promote_matmul_operand(
    data: np.ndarray,
    *,
    is_left_operand: bool,
) -> Tuple[np.ndarray, bool]:
    """Promote 1D vectors to 2D matrices following NumPy matmul semantics."""
    if data.ndim == 1:
        if is_left_operand:
            return data[np.newaxis, :], True
        return data[:, np.newaxis], True
    return data, False


def _promote_matmul_output_grad(
    grad: np.ndarray,
    *,
    left_was_vector: bool,
    right_was_vector: bool,
) -> np.ndarray:
    """Promote output gradient to matrix form used by matmul derivatives."""
    if left_was_vector and right_was_vector:
        return np.asarray(grad, dtype=np.float64).reshape(1, 1)

    promoted = grad
    if left_was_vector:
        promoted = np.expand_dims(promoted, axis=-2)
    if right_was_vector:
        promoted = np.expand_dims(promoted, axis=-1)
    return promoted


def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce a broadcasted gradient back to the original tensor shape."""
    if grad.shape == target_shape:
        return grad

    reduced = grad
    while reduced.ndim > len(target_shape):
        reduced = reduced.sum(axis=0)

    for axis, size in enumerate(target_shape):
        if size == 1 and reduced.shape[axis] != 1:
            reduced = reduced.sum(axis=axis, keepdims=True)

    return reduced.reshape(target_shape)


class Tensor:
    """Autodiff tensor supporting reverse-mode backpropagation."""

    __array_priority__ = 1000

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        self.data = _to_numpy(data)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = (
            np.zeros_like(self.data) if requires_grad else None
        )
        self._backward = lambda: None
        self._prev = _ordered_unique_tensors(_children)
        self._op = _op

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @staticmethod
    def _ensure_tensor(other: Union["Tensor", ArrayLike]) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def is_leaf(self) -> bool:
        return len(self._prev) == 0

    def item(self) -> float:
        if self.data.size != 1:
            raise ValueError("Tensor.item() only works for scalar tensors")
        return float(self.data.reshape(-1)[0])

    def __repr__(self) -> str:
        return (
            f"Tensor(data={self.data}, grad={self.grad}, "
            f"requires_grad={self.requires_grad})"
        )

    def __add__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other = self._ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="add",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self + (-self._ensure_tensor(other))

    def __rsub__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self._ensure_tensor(other) - self

    def __mul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other = self._ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = other.data * out.grad
                self.grad += _unbroadcast(grad_self, self.data.shape)
            if other.requires_grad:
                grad_other = self.data * out.grad
                other.grad += _unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self * other

    def __truediv__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other = self._ensure_tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="div",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = out.grad / other.data
                self.grad += _unbroadcast(grad_self, self.data.shape)
            if other.requires_grad:
                grad_other = -out.grad * self.data / (other.data**2)
                other.grad += _unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self._ensure_tensor(other) / self

    def __pow__(self, power: float) -> "Tensor":
        if not isinstance(power, (float, int)):
            raise TypeError("power must be a numeric scalar")

        out_data = self.data**power
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="pow",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += (power * (self.data ** (power - 1.0))) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other = self._ensure_tensor(other)

        if self.data.ndim == 0 or other.data.ndim == 0:
            raise ValueError("matmul operands must have at least 1 dimension")

        out = Tensor(
            np.matmul(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul",
        )

        def _backward() -> None:
            if out.grad is None:
                return

            self_matrix, self_was_vector = _promote_matmul_operand(
                self.data,
                is_left_operand=True,
            )
            other_matrix, other_was_vector = _promote_matmul_operand(
                other.data,
                is_left_operand=False,
            )
            grad_matrix = _promote_matmul_output_grad(
                out.grad,
                left_was_vector=self_was_vector,
                right_was_vector=other_was_vector,
            )

            if self.requires_grad:
                grad_self = np.matmul(grad_matrix, np.swapaxes(other_matrix, -1, -2))
                grad_self = _unbroadcast(grad_self, self_matrix.shape)
                if self_was_vector:
                    grad_self = np.squeeze(grad_self, axis=-2)
                self.grad += grad_self.reshape(self.data.shape)

            if other.requires_grad:
                grad_other = np.matmul(np.swapaxes(self_matrix, -1, -2), grad_matrix)
                grad_other = _unbroadcast(grad_other, other_matrix.shape)
                if other_was_vector:
                    grad_other = np.squeeze(grad_other, axis=-1)
                other.grad += grad_other.reshape(other.data.shape)

        out._backward = _backward
        return out

    def __rmatmul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self._ensure_tensor(other) @ self

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return

            grad = out.grad
            if axis is None:
                grad_input = np.ones_like(self.data) * grad
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                if not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, axis=ax)
                grad_input = np.broadcast_to(grad, self.data.shape)

            self.grad += grad_input

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            denom = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
            denom = int(np.prod([self.data.shape[a] for a in axes]))

        return self.sum(axis=axis, keepdims=keepdims) / float(denom)

    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> "Tensor":
        out = Tensor(
            np.transpose(self.data, axes=axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            if axes is None:
                inverse_axes = None
            else:
                inverse_axes = np.argsort(axes)
            self.grad += np.transpose(out.grad, axes=inverse_axes)

        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def exp(self) -> "Tensor":
        out_data = np.exp(self.data)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out_data * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        out_data = np.tanh(self.data)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="tanh",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += (1.0 - out_data**2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        out_data = np.empty_like(self.data)
        nonnegative = self.data >= 0.0

        exp_neg = np.exp(-self.data[nonnegative])
        out_data[nonnegative] = 1.0 / (1.0 + exp_neg)

        exp_pos = np.exp(self.data[~nonnegative])
        out_data[~nonnegative] = exp_pos / (1.0 + exp_pos)

        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sigmoid",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += (out_data * (1.0 - out_data)) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out_data = np.maximum(self.data, 0.0)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += (self.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def backward(self, grad: Optional[ArrayLike] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("cannot call backward() on a non-differentiable tensor")

        topo = []
        visited_ids = set()
        stack = [(self, False)]

        while stack:
            node, expanded = stack.pop()
            node_id = id(node)

            if expanded:
                topo.append(node)
                continue

            if node_id in visited_ids:
                continue

            visited_ids.add(node_id)
            stack.append((node, True))

            for parent in reversed(node._prev):
                if id(parent) not in visited_ids:
                    stack.append((parent, False))

        for node in topo:
            if node.requires_grad:
                if node.is_leaf():
                    if node.grad is None:
                        node.grad = np.zeros_like(node.data)
                else:
                    node.grad = np.zeros_like(node.data)

        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be provided for non-scalar outputs")
            seed_grad = np.ones_like(self.data)
        else:
            seed_grad = _to_numpy(grad)
            if seed_grad.shape != self.data.shape:
                raise ValueError(
                    f"grad shape {seed_grad.shape} does not match output shape {self.data.shape}"
                )

        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += seed_grad

        for node in reversed(topo):
            node._backward()
