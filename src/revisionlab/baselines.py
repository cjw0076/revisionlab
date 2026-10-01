"""Existing recursive least squares and explicit negative-control baselines."""

import math
from dataclasses import replace

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from .memory import ResidualMemory
from .protocols import ReleasedEvidence


def _key(values: Tensor, key: Tensor | NDArray[np.float64]) -> Tensor:
    if isinstance(key, Tensor):
        if key.dtype != values.dtype or key.device != values.device:
            raise ValueError("Tensor keys must match memory dtype and device")
        result = key
    else:
        result = torch.as_tensor(np.array(key, copy=True), dtype=values.dtype, device=values.device)
    if result.shape != values.shape or not bool(torch.isfinite(result).all()):
        raise ValueError("Key must be a finite vector matching memory width")
    return result


class RLS_Corrector(nn.Module):  # type: ignore[misc]
    """FP64-default RLS residual corrector, reproducing the v13 baseline.

    Initial covariance is 100 I. Forgetting lambda is in (0,1]; observed
    residuals are clipped to [-12,12]. Low-level writes do not deduplicate:
    use DelayedReplay for owner-bound, exactly-once evidence handling.
    """

    def __init__(self, slots: int, forget: float = 0.999, *, dtype: torch.dtype = torch.float64):
        super().__init__()
        if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
            raise ValueError("slots must be a positive integer")
        if not math.isfinite(forget) or not 0 < forget <= 1:
            raise ValueError("forget must be finite and in (0,1]")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be float32 or float64")
        self.register_buffer("forgetting_factor", torch.tensor(forget, dtype=dtype))
        self.register_buffer("values", torch.zeros(slots, dtype=dtype))
        self.register_buffer("P", 100 * torch.eye(slots, dtype=dtype))
        self.register_buffer("writes", torch.zeros((), dtype=torch.int64))
        self.register_buffer("last_label_hour", torch.full((), -1, dtype=torch.int64))

    def read(self, key: Tensor | NDArray[np.float64]) -> Tensor:
        result = torch.dot(self.values, _key(self.values, key))
        if not bool(torch.isfinite(result)):
            raise FloatingPointError("RLS read is non-finite")
        return result

    @property
    def forget(self) -> float:
        return float(self.forgetting_factor)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def write(self, evidence: ReleasedEvidence) -> None:
        if evidence.ticket.address_version != 0:
            raise ValueError("Only address version 0 is supported")
        if int(self.writes) >= torch.iinfo(torch.int64).max:
            raise OverflowError("RLS write counter exhausted; write not committed")
        key = _key(self.values, evidence.ticket.read_key)
        residual = evidence.target - evidence.ticket.base_prediction
        if not math.isfinite(residual):
            raise ValueError("Residual must be finite")
        target = max(-12.0, min(12.0, residual))
        px = self.P @ key
        denominator = self.forget + torch.dot(key, px)
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
            raise FloatingPointError("RLS denominator is not finite and positive")
        gain = px / denominator
        values = self.values + gain * (target - self.read(key))
        covariance = (self.P - torch.outer(gain, px)) / self.forget
        covariance = (covariance + covariance.T) * 0.5
        if not bool(torch.isfinite(values).all() and torch.isfinite(covariance).all()):
            raise FloatingPointError("RLS candidate state is non-finite; write not committed")
        self.values.copy_(values)
        self.P.copy_(covariance)
        self.writes.add_(1)
        self.last_label_hour.fill_(evidence.ticket.available_hour)

    def config(self) -> dict[str, object]:
        return {
            "slots": self.values.numel(),
            "forget": self.forget,
            "dtype": str(self.values.dtype).removeprefix("torch."),
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "values": self.values.detach().clone(),
            "P": self.P.detach().clone(),
            "writes": int(self.writes),
            "last_label_hour": None if int(self.writes) == 0 else int(self.last_label_hour),
            "forget": self.forget,
        }


class NoWrite(ResidualMemory):
    """Zero-correction control: accept mature evidence but never modify state."""

    def __init__(self, slots: int, *, dtype: torch.dtype = torch.float64):
        super().__init__(slots, 1.0, dtype=dtype)

    def read(self, key: Tensor | NDArray[np.float64]) -> Tensor:
        _key(self.values, key)
        return torch.zeros((), dtype=self.values.dtype, device=self.values.device)

    def write(self, evidence: ReleasedEvidence) -> None:
        if evidence.ticket.address_version != 0:
            raise ValueError("Only address version 0 is supported")
        _key(self.values, evidence.ticket.read_key)
        if not math.isfinite(evidence.target - evidence.ticket.base_prediction):
            raise ValueError("Residual must be finite")

    def config(self) -> dict[str, object]:
        return {
            "slots": self.values.numel(),
            "dtype": str(self.values.dtype).removeprefix("torch."),
        }


class ShuffledWrite(ResidualMemory):
    """Negative control: permute write addresses, keep read addresses unchanged."""

    def __init__(
        self, slots: int, rate: float = 0.05, *, seed: int = 0, dtype: torch.dtype = torch.float64
    ):
        if slots < 2:
            raise ValueError("Shuffled-write requires at least two slots")
        super().__init__(slots, rate, dtype=dtype)
        self.seed = seed
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(slots, generator=generator)
        if torch.equal(permutation, torch.arange(slots)):
            permutation = permutation.roll(1)
        self.register_buffer("permutation", permutation)

    def write(self, evidence: ReleasedEvidence) -> None:
        key = _key(self.values, evidence.ticket.read_key)
        shuffled = key[self.permutation].detach().cpu().numpy()
        super().write(replace(evidence, ticket=replace(evidence.ticket, read_key=shuffled)))

    def config(self) -> dict[str, object]:
        return {**super().config(), "seed": self.seed}
