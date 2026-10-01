"""Normalized delta/LMS updates; evidence ownership belongs to the replay runner."""

import math
from numbers import Integral
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from .protocols import ReleasedEvidence, _key


def _size_rate(slots: int, rate: float) -> None:
    if isinstance(slots, bool) or not isinstance(slots, Integral) or slots < 1:
        raise ValueError("slots must be a positive integer")
    if isinstance(rate, bool) or not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("rate must be finite and in (0, 1]")


def delta_write(values: Tensor, key: Tensor, target: Tensor, rate: float) -> Tensor:
    """Pure differentiable update: [...,slots,channels], [...,slots], [...,channels].

    All tensors must share dtype/device and exactly matching leading dimensions.
    Finite inputs and representable intermediate squared key norms are required;
    stateful backends validate norms and proposals before committing. This
    function neither clips residuals nor mutates inputs.
    """
    if values.ndim < 2 or key.ndim != values.ndim - 1 or target.ndim != values.ndim - 1:
        raise ValueError(
            "Expected values [...,slots,channels], key [...,slots], target [...,channels]"
        )
    if (
        values.shape[-2] < 1
        or values.shape[-1] < 1
        or key.shape != values.shape[:-1]
        or target.shape != (*values.shape[:-2], values.shape[-1])
    ):
        raise ValueError("Delta update shape mismatch")
    if not values.is_floating_point() or values.dtype != key.dtype or values.dtype != target.dtype:
        raise ValueError("Delta update requires one shared floating dtype")
    if values.device != key.device or values.device != target.device:
        raise ValueError("Delta update requires one shared device")
    if isinstance(rate, bool) or not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("rate must be finite and in (0, 1]")
    prediction = torch.einsum("...n,...nd->...d", key, values)
    floor = max(1e-12, torch.finfo(values.dtype).tiny)
    denominator = key.square().sum(-1, keepdim=True).clamp_min(floor)
    return values + (rate / denominator).unsqueeze(-1) * key.unsqueeze(-1) * (
        target - prediction
    ).unsqueeze(-2)


class SparseResidualMemory:
    """Backward-compatible NumPy scalar backend; one instance per horizon."""

    def __init__(
        self, slots: int, rate: float, permutation: NDArray[np.int64] | None = None
    ) -> None:
        _size_rate(slots, rate)
        self.values: NDArray[np.float64] = np.zeros(slots, dtype=np.float64)
        self.rate = float(rate)
        self.writes = 0
        self.last_label_hour: int | None = None
        self.permutation: NDArray[np.int64] | None = None
        if permutation is not None:
            candidate = np.asarray(permutation)
            if (
                candidate.shape != (slots,)
                or not np.issubdtype(candidate.dtype, np.integer)
                or not np.array_equal(np.sort(candidate), np.arange(slots))
            ):
                raise ValueError("Permutation must be an integral bijection")
            self.permutation = np.frombuffer(candidate.astype(np.int64).tobytes(), dtype=np.int64)

    def read(self, key: Any) -> float:
        result = float(np.dot(self.values, _key(key, self.values.size)))
        if not math.isfinite(result):
            raise FloatingPointError("Memory read is not finite")
        return result

    def write(self, evidence: ReleasedEvidence) -> None:
        if evidence.ticket.address_version != 0:
            raise ValueError("Unsupported address_version")
        key = _key(evidence.ticket.read_key, self.values.size)
        if self.permutation is not None:
            key = key[self.permutation]
        with np.errstate(over="ignore", invalid="ignore"):
            squared_norm = float(key @ key)
        if not math.isfinite(squared_norm):
            raise FloatingPointError("Key squared norm is not finite")
        residual = float(np.clip(evidence.target - evidence.ticket.base_prediction, -12, 12))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            proposal = self.values + self.rate * key * (residual - np.dot(self.values, key)) / max(
                squared_norm, 1e-12
            )
        if not np.isfinite(proposal).all():
            raise FloatingPointError("Memory proposal is not finite")
        self.values[:] = proposal
        self.writes += 1
        self.last_label_hour = evidence.ticket.available_hour

    def snapshot(self) -> dict[str, Any]:
        return {
            "values": self.values.copy(),
            "writes": self.writes,
            "rate": self.rate,
            "last_label_hour": self.last_label_hour,
        }


class ResidualMemory(nn.Module):  # type: ignore[misc]
    """Native Torch scalar state; one instance per horizon, default FP64.

    Tensor keys require an exact dtype/device match. NumPy ticket keys are
    explicitly copied into this backend's configured dtype/device. Stateful
    writes use clipped target-minus-base residuals and have no autograd graph.
    """

    def __init__(
        self,
        slots: int,
        rate: float,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        _size_rate(slots, rate)
        if not dtype.is_floating_point:
            raise ValueError("Memory requires a floating dtype")
        self.register_buffer("values", torch.zeros(slots, dtype=dtype, device=device))
        self.register_buffer("learning_rate", torch.tensor(rate, dtype=dtype, device=device))
        self.register_buffer("writes", torch.zeros((), dtype=torch.int64, device=device))
        self.register_buffer(
            "last_label_hour", torch.full((), -1, dtype=torch.int64, device=device)
        )

    @property
    def rate(self) -> float:
        return float(self.learning_rate.item())

    def _read_key(self, key: Any) -> Tensor:
        if isinstance(key, Tensor):
            if key.dtype != self.values.dtype or key.device != self.values.device:
                raise ValueError("Tensor key dtype/device must match memory")
            result = key
        else:
            result = torch.tensor(_key(key), dtype=self.values.dtype, device=self.values.device)
        if result.shape != self.values.shape or not torch.isfinite(result).all():
            raise ValueError("A finite key with matching address width is required")
        return result

    def read(self, key: Any) -> Tensor:
        result = torch.dot(self.values, self._read_key(key))
        if not torch.isfinite(result):
            raise FloatingPointError("Memory read is not finite")
        return result

    def write(self, evidence: ReleasedEvidence) -> None:
        if evidence.ticket.address_version != 0:
            raise ValueError("Unsupported address_version")
        hour = evidence.ticket.available_hour
        limits = torch.iinfo(torch.int64)
        if not limits.min <= hour <= limits.max or int(self.writes.item()) >= limits.max:
            raise ValueError("Evidence hour/write count exceeds native metadata range")
        key = self._read_key(evidence.ticket.read_key)
        if not torch.isfinite(key.square().sum()):
            raise FloatingPointError("Key squared norm is not finite")
        residual = max(-12.0, min(12.0, evidence.target - evidence.ticket.base_prediction))
        target = self.values.new_tensor([residual])
        with torch.no_grad():
            proposal = delta_write(self.values[:, None], key, target, self.rate).squeeze(-1)
            if not torch.isfinite(proposal).all():
                raise FloatingPointError("Memory proposal is not finite")
            self.values.copy_(proposal)
            self.writes.add_(1)
            self.last_label_hour.fill_(hour)

    def snapshot(self) -> dict[str, Any]:
        hour = int(self.last_label_hour.item())
        return {
            "values": self.values.detach().clone(),
            "writes": int(self.writes.item()),
            "rate": self.rate,
            "last_label_hour": None if int(self.writes.item()) == 0 else hour,
        }

    def config(self) -> dict[str, Any]:
        return {
            "slots": self.values.numel(),
            "rate": self.rate,
            "dtype": str(self.values.dtype).removeprefix("torch."),
        }
