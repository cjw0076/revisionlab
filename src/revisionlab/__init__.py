"""Delayed evidence and residual-memory experiments, native to PyTorch."""

from .diagnostics import aggregate_alignment, credit_alignment
from .memory import ResidualMemory, SparseResidualMemory, delta_write
from .protocols import BudgetReport, ForecastTicket, MemoryProtocol, ReleasedEvidence

__version__ = "0.1.0"

__all__ = [
    "BudgetReport",
    "ForecastTicket",
    "MemoryProtocol",
    "ReleasedEvidence",
    "ResidualMemory",
    "SparseResidualMemory",
    "aggregate_alignment",
    "credit_alignment",
    "delta_write",
]
