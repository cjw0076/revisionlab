"""Immutable issued forecasts and causally released evidence."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import numpy as np
import torch
from numpy.typing import NDArray


def _tick(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integral tick")
    if not -(2**63) <= value < 2**63:
        raise ValueError(f"{name} must fit the signed int64 tick range")
    return int(value)


def _scalar(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite scalar")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(array)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _key(value: Any, slots: int | None = None) -> NDArray[np.float64]:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu", dtype=torch.float64).numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError("A finite nonempty one-dimensional key is required")
    if slots is not None and result.size != slots:
        raise ValueError("Address width mismatch")
    return result


@dataclass(frozen=True)
class ForecastTicket:
    issue_hour: int
    horizon: int
    base_prediction: float
    read_key: NDArray[np.float64]
    prediction: float | None = None
    ticket_id: str = field(default_factory=lambda: str(uuid4()))
    address_version: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "issue_hour", _tick(self.issue_hour, "issue_hour"))
        horizon = _tick(self.horizon, "horizon")
        if horizon < 1:
            raise ValueError("Horizon must be positive")
        object.__setattr__(self, "horizon", horizon)
        _tick(self.issue_hour + horizon, "available_hour")
        object.__setattr__(
            self, "base_prediction", _scalar(self.base_prediction, "base_prediction")
        )
        if self.prediction is not None:
            object.__setattr__(self, "prediction", _scalar(self.prediction, "prediction"))
        version = _tick(self.address_version, "address_version")
        if version < 0:
            raise ValueError("address_version must be nonnegative")
        object.__setattr__(self, "address_version", version)
        if not isinstance(self.ticket_id, str) or not self.ticket_id:
            raise ValueError("ticket_id must be a nonempty string")
        # Immutable bytes backing prevents callers from re-enabling NumPy writes.
        object.__setattr__(self, "read_key", _key(self.read_key).tobytes())

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        if name == "read_key" and isinstance(value, bytes):
            # Return a fresh view: NumPy shape/dtype metadata remains mutable even
            # when its data backing is immutable. Never expose our stored object.
            return np.frombuffer(value, dtype=np.float64)
        return value

    @property
    def available_hour(self) -> int:
        return self.issue_hour + self.horizon


@dataclass(frozen=True)
class ReleasedEvidence:
    ticket: ForecastTicket
    target: float
    now_hour: int

    def __post_init__(self) -> None:
        if not isinstance(self.ticket, ForecastTicket):
            raise ValueError("Evidence requires a ForecastTicket")
        now = _tick(self.now_hour, "now_hour")
        if now < self.ticket.available_hour:
            raise ValueError("Outcome has not matured")
        object.__setattr__(self, "now_hour", now)
        object.__setattr__(self, "target", _scalar(self.target, "target"))


@dataclass(frozen=True)
class BudgetReport:
    value_bytes: int
    metadata_bytes: int
    pending_bytes: int
    seconds: float


@runtime_checkable
class MemoryProtocol(Protocol):
    def read(self, key: Any) -> Any: ...

    def write(self, evidence: ReleasedEvidence) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...
