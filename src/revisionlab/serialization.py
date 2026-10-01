"""Atomic, weights-only checkpoints with explicit model and ledger validation."""

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .baselines import NoWrite, RLS_Corrector, ShuffledWrite
from .memory import ResidualMemory
from .replay import DelayedReplay, ReplayLedger


def save_checkpoint(path: str | Path, runner: DelayedReplay) -> None:
    if type(runner.memory) not in (ResidualMemory, RLS_Corrector, NoWrite, ShuffledWrite):
        raise ValueError("Checkpoint supports native RevisionLab models only")
    if runner.memory.values.dtype not in (torch.float32, torch.float64):
        raise ValueError("Checkpoint dtype must be float32 or float64")
    if isinstance(runner.memory, ResidualMemory) and not 0 < runner.memory.rate <= 1:
        raise ValueError("Checkpoint learning rate must be in (0,1]")
    if isinstance(runner.memory, RLS_Corrector) and not 0 < runner.memory.forget <= 1:
        raise ValueError("Checkpoint forgetting factor must be in (0,1]")
    for name, value in runner.memory.state_dict().items():
        if value.is_floating_point() and value.dtype != runner.memory.values.dtype:
            raise ValueError(f"Checkpoint tensor dtype mismatch: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Checkpoint tensor is non-finite: {name}")
    payload = {
        "format_version": 1,
        "model_type": type(runner.memory).__name__,
        "model_config": runner.memory.config(),
        "model_state": {
            name: value.detach().cpu().clone() for name, value in runner.memory.state_dict().items()
        },
        "ledger": runner.ledger.state(),
        "correction_clip": runner.correction_clip,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | Path) -> DelayedReplay:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    factories: dict[str, type[nn.Module]] = {
        "ResidualMemory": ResidualMemory,
        "RLS_Corrector": RLS_Corrector,
        "NoWrite": NoWrite,
        "ShuffledWrite": ShuffledWrite,
    }
    model_type = payload.get("model_type")
    if model_type not in factories:
        raise ValueError("Unknown checkpoint model type")
    config = dict(payload["model_config"])
    dtype_name = config.pop("dtype", None)
    if dtype_name not in ("float32", "float64"):
        raise ValueError("Checkpoint dtype must be float32 or float64")
    dtype = torch.float32 if dtype_name == "float32" else torch.float64
    memory = factories[model_type](**config, dtype=dtype)
    expected = memory.state_dict()
    saved = payload["model_state"]
    if not isinstance(saved, dict) or set(saved) != set(expected):
        raise ValueError("Checkpoint model state keys mismatch")
    for name, value in saved.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError("Checkpoint state must contain tensors")
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"Checkpoint tensor shape/dtype mismatch: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Checkpoint tensor is non-finite: {name}")
    if int(saved["writes"]) < 0:
        raise ValueError("Checkpoint write counter cannot be negative")
    if isinstance(memory, ResidualMemory):
        if not 0 < float(saved["learning_rate"]) <= 1:
            raise ValueError("Checkpoint learning rate must be in (0,1]")
        if not torch.equal(saved["learning_rate"], expected["learning_rate"]):
            raise ValueError("Checkpoint learning rate and config mismatch")
    if isinstance(memory, RLS_Corrector):
        if not 0 < float(saved["forgetting_factor"]) <= 1:
            raise ValueError("Checkpoint forgetting factor must be in (0,1]")
        if not torch.equal(saved["forgetting_factor"], expected["forgetting_factor"]):
            raise ValueError("Checkpoint forgetting factor and config mismatch")
    if isinstance(memory, NoWrite) and (bool(saved["values"].any()) or int(saved["writes"]) != 0):
        raise ValueError("No-write checkpoint must preserve zero state")
    if isinstance(memory, ShuffledWrite):
        permutation = saved["permutation"]
        if not torch.equal(permutation.sort().values, torch.arange(memory.values.numel())):
            raise ValueError("Checkpoint write permutation must be bijective")
        if not torch.equal(permutation, expected["permutation"]):
            raise ValueError("Checkpoint write permutation and seed config mismatch")
    memory.load_state_dict(saved, strict=True)
    ledger = ReplayLedger.from_state(payload["ledger"])
    for ticket in ledger.pending.values():
        if ticket.read_key.size != memory.values.numel():
            raise ValueError("Pending ticket address width does not match checkpoint model")
    runner = DelayedReplay(memory, ledger.horizon, correction_clip=payload["correction_clip"])
    runner.ledger = ledger
    return runner
