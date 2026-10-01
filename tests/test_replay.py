import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

from revisionlab.baselines import NoWrite, RLS_Corrector, ShuffledWrite
from revisionlab.memory import ResidualMemory
from revisionlab.replay import DelayedReplay, ReplayLedger
from revisionlab.serialization import load_checkpoint, save_checkpoint
from revisionlab.smoke import synthetic_check


def test_predictions_are_issue_time_values_and_no_pending_targets():
    runner = DelayedReplay(ResidualMemory(2, 1.0), 2)
    first = runner.issue(0, 10.0, np.array([1.0, 0.0]))
    second = runner.issue(1, 10.0, np.array([1.0, 0.0]))
    assert first.prediction == second.prediction == 10
    assert "target" not in vars(runner.ledger.pending[first.ticket_id])
    assert runner.release(first.ticket_id, 12.0, 2) == 4
    assert runner.release(second.ticket_id, 12.0, 3) == 4  # no historical recomputation
    third = runner.issue(3, 10.0, np.array([1.0, 0.0]))
    assert third.prediction == 12
    assert runner.ledger.mse == 4
    assert runner.ledger.base_mse == 4
    assert runner.ledger.budget()["settled_ids"] == 2
    assert runner.ledger.budget()["audit_id_payload_bytes"] > 0


def test_exactly_once_owned_release_and_maturity():
    runner = DelayedReplay(ResidualMemory(2, 0.5), 2)
    ticket = runner.issue(0, 0, np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="matured"):
        runner.release(ticket.ticket_id, 2, 1)
    with pytest.raises(ValueError, match="foreign"):
        runner.release("another-owner:ticket", 2, 2)
    runner.release(ticket.ticket_id, 2, 2)
    before = runner.memory.values.clone()
    with pytest.raises(ValueError, match="already"):
        runner.release(ticket.ticket_id, 2, 2)
    torch.testing.assert_close(before, runner.memory.values)
    assert runner.ledger.releases == 1


def test_ledger_rejects_duplicate_issue_and_mismatching_contract():
    runner = DelayedReplay(ResidualMemory(2, 1), 2)
    ticket = runner.issue(0, 0, np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="already"):
        runner.ledger.issue(ticket)
    with pytest.raises(ValueError, match="horizon"):
        runner.ledger.issue(replace(ticket, horizon=3, ticket_id=runner.ledger.owner_id + ":new"))
    with pytest.raises(ValueError, match="version"):
        runner.ledger.issue(
            replace(ticket, address_version=1, ticket_id=runner.ledger.owner_id + ":new")
        )


def test_monotonic_events_and_arbitrary_ticket_order_at_same_tick():
    runner = DelayedReplay(ResidualMemory(2, 1), 1)
    first = runner.issue(0, 0, np.array([1.0, 0.0]))
    second = runner.issue(0, 0, np.array([0.0, 1.0]))
    runner.release(second.ticket_id, 2, 1)
    runner.release(first.ticket_id, 1, 1)
    runner.issue(1, 0, np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="monotonic"):
        runner.issue(0, 0, np.array([1.0, 0.0]))


def test_failed_write_stays_pending_and_retryable(monkeypatch):
    runner = DelayedReplay(ResidualMemory(2, 1), 1)
    ticket = runner.issue(0, 0, np.array([1.0, 0.0]))
    write = runner.memory.write

    def fail(evidence):
        raise FloatingPointError("Injected rejected write")

    monkeypatch.setattr(runner.memory, "write", fail)
    with pytest.raises(FloatingPointError):
        runner.release(ticket.ticket_id, 2, 1)
    assert ticket.ticket_id in runner.ledger.pending
    assert ticket.ticket_id not in runner.ledger.settled_ids
    assert runner.ledger.releases == 0
    assert runner.ledger.clock == 0
    monkeypatch.setattr(runner.memory, "write", write)
    assert runner.release(ticket.ticket_id, 2, 1) == 4


@pytest.mark.parametrize("factory", [ResidualMemory, RLS_Corrector, NoWrite, ShuffledWrite])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_checkpoint_resume_pending_and_audit_parity(tmp_path, factory, dtype):
    memory = factory(3, dtype=dtype) if factory != ResidualMemory else factory(3, 0.5, dtype=dtype)
    runner = DelayedReplay(memory, 2)
    first = runner.issue(0, 0, np.array([1.0, 0.0, 0.0]))
    second = runner.issue(1, 0, np.array([0.0, 1.0, 0.0]))
    runner.release(first.ticket_id, 2, 2)
    path = tmp_path / "state.pt"
    save_checkpoint(path, runner)
    resumed = load_checkpoint(path)
    assert resumed.ledger.owner_id == runner.ledger.owner_id
    assert resumed.memory.values.dtype == dtype
    assert resumed.ledger.settled_ids == runner.ledger.settled_ids
    assert resumed.release(second.ticket_id, -1, 3) == runner.release(second.ticket_id, -1, 3)
    for name, value in runner.memory.state_dict().items():
        torch.testing.assert_close(value, resumed.memory.state_dict()[name])
    key = np.array([1.0, 0.0, 0.0])
    assert resumed.issue(3, 5, key).prediction == runner.issue(3, 5, key).prediction
    assert resumed.ledger.mse == runner.ledger.mse
    with pytest.raises(ValueError, match="already"):
        resumed.release(first.ticket_id, 2, 3)
    payload = torch.load(path, weights_only=True)
    assert isinstance(payload, dict)
    assert isinstance(payload["ledger"]["pending"][0]["read_key"], torch.Tensor)


def test_checkpoint_rejects_dtype_cast_and_invalid_audit(tmp_path):
    runner = DelayedReplay(ResidualMemory(2, 0.5), 1)
    runner.issue(0, 0, np.array([1.0, 0.0]))
    path = tmp_path / "state.pt"
    save_checkpoint(path, runner)
    payload = torch.load(path, weights_only=True)
    payload["model_state"]["values"] = payload["model_state"]["values"].float()
    torch.save(payload, path)
    with pytest.raises(ValueError, match="dtype"):
        load_checkpoint(path)
    state = runner.ledger.state()
    state["clock"] = -1
    with pytest.raises(ValueError, match="monotonic"):
        ReplayLedger.from_state(state)


def test_atomic_checkpoint_failure_keeps_previous_file(tmp_path, monkeypatch):
    runner = DelayedReplay(ResidualMemory(2, 0.5), 1)
    path = tmp_path / "state.pt"
    save_checkpoint(path, runner)
    original = path.read_bytes()

    def fail_save(payload, stream):
        stream.write(b"incomplete")
        raise OSError("Injected failed checkpoint")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError):
        save_checkpoint(path, runner)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("factory", [ResidualMemory, NoWrite, ShuffledWrite])
def test_checkpoint_rejects_rate_buffer_config_mismatch(tmp_path, factory):
    memory = factory(3, 0.5) if factory == ResidualMemory else factory(3)
    runner = DelayedReplay(memory, 1)
    path = tmp_path / "state.pt"
    save_checkpoint(path, runner)
    payload = torch.load(path, weights_only=True)
    payload["model_state"]["learning_rate"].fill_(0.7)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="learning rate.*config"):
        load_checkpoint(path)


def test_unsupported_checkpoint_dtype_preserves_existing_file(tmp_path):
    runner = DelayedReplay(ResidualMemory(2, 0.5), 1)
    path = tmp_path / "state.pt"
    save_checkpoint(path, runner)
    original = path.read_bytes()
    runner.memory.to(dtype=torch.float16)
    with pytest.raises(ValueError, match="dtype"):
        save_checkpoint(path, runner)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("identity", [True, False])
def test_shuffled_checkpoint_rejects_permutation_seed_mismatch(tmp_path, identity):
    runner = DelayedReplay(ShuffledWrite(4, seed=7), 1)
    path = tmp_path / "state.pt"
    save_checkpoint(path, runner)
    payload = torch.load(path, weights_only=True)
    original = payload["model_state"]["permutation"]
    replacement = torch.arange(4) if identity else original.roll(1)
    assert not torch.equal(replacement, original)
    payload["model_state"]["permutation"] = replacement
    torch.save(payload, path)
    with pytest.raises(ValueError, match="permutation.*config"):
        load_checkpoint(path)


def test_private_settle_repeated_call_keeps_metrics_intact():
    runner = DelayedReplay(ResidualMemory(2, 0.5), 1)
    ticket = runner.issue(0, 0, np.array([1.0, 0.0]))
    evidence = runner.ledger.validate_release(ticket.ticket_id, 2, 1)
    runner.release(ticket.ticket_id, 2, 1)
    with pytest.raises(ValueError, match="already"):
        runner.ledger._settle(evidence)
    assert runner.ledger.squared_error == 4
    assert runner.ledger.releases == 1


def test_numpy_ticks_and_checkpoint_clock_bounds():
    runner = DelayedReplay(ResidualMemory(2, 0.5), np.int64(1))
    ticket = runner.issue(np.int64(0), 0, np.array([1.0, 0.0]))
    runner.release(ticket.ticket_id, 1, np.int64(1))
    state = runner.ledger.state()
    state["clock"] = 2**63
    with pytest.raises(ValueError, match="int64"):
        ReplayLedger.from_state(state)
    with pytest.raises(ValueError, match="owner"):
        ReplayLedger(1, owner_id="")


def test_synthetic_controls_and_cli_smoke():
    report = synthetic_check(128, 0, 3)
    assert "synthetic" in report["scope"]
    methods = report["methods"]
    assert methods["no-write"]["corrected_mse"] == methods["no-write"]["base_mse"]
    assert methods["residual"]["corrected_mse"] < methods["residual"]["base_mse"]
    assert methods["rls"]["corrected_mse"] < methods["rls"]["base_mse"]
    assert report == synthetic_check(128, 0, 3)
    process = subprocess.run(
        [sys.executable, "-m", "revisionlab.smoke", "--steps", "8", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(process.stdout)["steps"] == 8


@pytest.mark.parametrize("argument", ["--steps", "--horizon"])
def test_cli_rejects_nonpositive_inputs(argument):
    process = subprocess.run(
        [sys.executable, "-m", "revisionlab.smoke", argument, "0"], capture_output=True, text=True
    )
    assert process.returncode == 2
    assert "positive" in process.stderr
