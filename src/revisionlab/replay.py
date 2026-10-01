"""Owner-bound delayed evidence replay; pending state contains no future labels."""

import math
import sys
import uuid
from numbers import Integral
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .protocols import ForecastTicket, ReleasedEvidence


class ReplayLedger:
    """One horizon and address version per ledger, with retained audit IDs.

    Settled IDs are retained without a bound to guarantee lifetime exactly-once
    release through checkpoints. There is no automatic retention limit: callers
    must rotate ledgers explicitly if accepting a shorter deduplication window.
    ``budget`` exposes growing audit payload plus Python container overhead.
    """

    def __init__(self, horizon: int, *, owner_id: str | None = None):
        if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        self.horizon: int = int(horizon)
        self.owner_id: str = str(uuid.uuid4()) if owner_id is None else owner_id
        if not isinstance(self.owner_id, str) or not self.owner_id or ":" in self.owner_id:
            raise ValueError("owner_id must be a nonempty string without colon")
        self.pending: dict[str, ForecastTicket] = {}
        self.settled_ids: set[str] = set()
        self.clock: int | None = None
        self.squared_error: float = 0.0
        self.base_squared_error: float = 0.0
        self.releases: int = 0

    def _time(self, hour: int) -> None:
        if isinstance(hour, bool) or not isinstance(hour, Integral):
            raise ValueError("Event time must be an integer hour")
        if not -(2**63) <= hour < 2**63:
            raise ValueError("Event time must fit int64")
        if self.clock is not None and hour < self.clock:
            raise ValueError("Event time must be monotonic; release before issuing at that tick")

    def issue(self, ticket: ForecastTicket) -> None:
        self._time(ticket.issue_hour)
        if ticket.horizon != self.horizon or ticket.address_version != 0:
            raise ValueError("Ticket horizon or address version does not match ledger")
        if not ticket.ticket_id.startswith(self.owner_id + ":"):
            raise ValueError("Foreign ticket owner")
        if ticket.ticket_id in self.pending or ticket.ticket_id in self.settled_ids:
            raise ValueError("Ticket ID has already been issued")
        if ticket.prediction is None or not math.isfinite(ticket.prediction):
            raise ValueError("Ledger requires the actual finite issued prediction")
        self.pending[ticket.ticket_id] = ticket
        self.clock = ticket.issue_hour

    def validate_release(self, ticket_id: str, target: float, now_hour: int) -> ReleasedEvidence:
        self._time(now_hour)
        if ticket_id in self.settled_ids:
            raise ValueError("Ticket has already been released")
        if ticket_id not in self.pending:
            raise ValueError("Unknown or foreign ticket ID")
        ticket = self.pending[ticket_id]
        if ticket.address_version != 0 or ticket.horizon != self.horizon:
            raise ValueError("Stored ticket horizon or address version does not match ledger")
        evidence = ReleasedEvidence(ticket, target, now_hour)
        assert ticket.prediction is not None
        try:
            squared_error = (ticket.prediction - evidence.target) ** 2
            base_error = (ticket.base_prediction - evidence.target) ** 2
        except OverflowError as error:
            raise ValueError("Squared error must be finite") from error
        if not math.isfinite(
            squared_error + self.squared_error + base_error + self.base_squared_error
        ):
            raise ValueError("Squared error must be finite")
        return evidence

    def _settle(self, evidence: ReleasedEvidence) -> float:
        ticket = evidence.ticket
        canonical = self.validate_release(ticket.ticket_id, evidence.target, evidence.now_hour)
        if canonical.ticket is not ticket:
            raise ValueError("Release must use the canonical pending ticket")
        assert ticket.prediction is not None
        loss = (ticket.prediction - evidence.target) ** 2
        self.squared_error += loss
        self.base_squared_error += (ticket.base_prediction - evidence.target) ** 2
        self.releases += 1
        del self.pending[ticket.ticket_id]
        self.settled_ids.add(ticket.ticket_id)
        self.clock = evidence.now_hour
        return loss

    @property
    def mse(self) -> float | None:
        return self.squared_error / self.releases if self.releases else None

    @property
    def base_mse(self) -> float | None:
        return self.base_squared_error / self.releases if self.releases else None

    def budget(self) -> dict[str, int]:
        return {
            "pending_tickets": len(self.pending),
            "settled_ids": len(self.settled_ids),
            "pending_key_bytes": sum(ticket.read_key.nbytes for ticket in self.pending.values()),
            "audit_id_payload_bytes": sum(len(value.encode()) for value in self.settled_ids),
            "audit_python_bytes": sys.getsizeof(self.settled_ids)
            + sum(sys.getsizeof(value) for value in self.settled_ids),
        }

    def state(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "owner_id": self.owner_id,
            "clock": self.clock,
            "squared_error": self.squared_error,
            "base_squared_error": self.base_squared_error,
            "releases": self.releases,
            "settled_ids": sorted(self.settled_ids),
            "pending": [
                {
                    "issue_hour": ticket.issue_hour,
                    "horizon": ticket.horizon,
                    "base_prediction": ticket.base_prediction,
                    "read_key": torch.tensor(np.array(ticket.read_key, copy=True)),
                    "prediction": ticket.prediction,
                    "ticket_id": ticket.ticket_id,
                    "address_version": ticket.address_version,
                }
                for ticket in self.pending.values()
            ],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "ReplayLedger":
        ledger = cls(state["horizon"], owner_id=state["owner_id"])
        settled = state["settled_ids"]
        if len(set(settled)) != len(settled) or any(
            not isinstance(value, str) or not value.startswith(ledger.owner_id + ":")
            for value in settled
        ):
            raise ValueError("Invalid settled ticket IDs")
        ledger.settled_ids = set(settled)
        for data in sorted(state["pending"], key=lambda item: item["issue_hour"]):
            ledger.issue(ForecastTicket(**data))
        clock = state["clock"]
        if clock is not None:
            ledger._time(clock)
        elif ledger.pending or ledger.settled_ids:
            raise ValueError("Nonempty ledger requires a clock")
        ledger.clock = clock
        count = state["releases"]
        if isinstance(count, bool) or not isinstance(count, int) or count != len(settled):
            raise ValueError("Release count must match settled audit IDs")
        ledger.releases = count
        for name in ("squared_error", "base_squared_error"):
            value = state[name]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid accumulated error")
            setattr(ledger, name, float(value))
        return ledger


class DelayedReplay:
    """Sequential single-process replay: revise only after an issued outcome matures.

    Calls must be serialized by the caller; this class is not thread-safe.
    """

    def __init__(self, memory: Any, horizon: int, *, correction_clip: float = 3.0):
        if not math.isfinite(correction_clip) or correction_clip <= 0:
            raise ValueError("correction_clip must be finite and positive")
        self.memory = memory
        self.ledger = ReplayLedger(horizon)
        self.correction_clip = float(correction_clip)

    def issue(
        self, issue_hour: int, base_prediction: float, read_key: NDArray[np.float64] | torch.Tensor
    ) -> ForecastTicket:
        self.ledger._time(issue_hour)
        correction = float(self.memory.read(read_key))
        if not math.isfinite(correction):
            raise ValueError("Memory correction must be finite")
        correction = max(-self.correction_clip, min(self.correction_clip, correction))
        ticket = ForecastTicket(
            issue_hour,
            self.ledger.horizon,
            base_prediction,
            read_key,
            prediction=base_prediction + correction,
            ticket_id=self.ledger.owner_id + ":" + str(uuid.uuid4()),
            address_version=0,
        )
        self.ledger.issue(ticket)
        return ticket

    def release(self, ticket_id: str, target: float, now_hour: int) -> float:
        evidence = self.ledger.validate_release(ticket_id, target, now_hour)
        self.memory.write(evidence)
        return self.ledger._settle(evidence)
