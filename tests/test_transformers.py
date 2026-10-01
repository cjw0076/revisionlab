"""Offline coverage of the optional real Transformers regression boundary."""

import socket

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
from transformers import BertConfig, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

from revisionlab import ResidualMemory
from revisionlab.integrations.transformers import issue_regression
from revisionlab.replay import DelayedReplay
from revisionlab.serialization import load_checkpoint, save_checkpoint


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject_connection(*args, **kwargs):
        raise AssertionError("Transformers integration tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", reject_connection)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_connection)
    monkeypatch.setattr(socket, "create_connection", reject_connection)


@pytest.fixture
def regression_model():
    torch.manual_seed(7)
    return BertForSequenceClassification(
        BertConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=16,
            num_labels=1,
            problem_type="regression",
        )
    ).eval()


def test_real_model_output_delayed_release_preserves_model_and_pending_values(regression_model):
    model = regression_model
    inputs = torch.tensor([[1, 4, 2], [1, 6, 2]])
    output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), return_dict=True)
    assert isinstance(output, SequenceClassifierOutput)
    assert output.logits.requires_grad
    original_logits = output.logits.detach().clone()
    model_state = {name: value.clone() for name, value in model.state_dict().items()}
    runner = DelayedReplay(ResidualMemory(2, 0.5), horizon=2)
    memory_state = {name: value.clone() for name, value in runner.memory.state_dict().items()}
    key = np.array([1.0, 0.0])
    ticket = issue_regression(runner, output, issue_hour=0, read_key=key, sample_index=1)
    baseline = original_logits[1, 0].item()
    assert ticket.base_prediction == ticket.prediction == baseline
    assert "target" not in vars(ticket)
    assert output.logits.requires_grad
    torch.testing.assert_close(output.logits, original_logits)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, model_state[name])
    for name, value in runner.memory.state_dict().items():
        torch.testing.assert_close(value, memory_state[name])
    assert all(parameter.grad is None for parameter in model.parameters())

    # Later mutation of caller-owned objects cannot rewrite a past forecast or address.
    key[:] = [0.0, 1.0]
    with torch.no_grad():
        output.logits.fill_(99)
    assert ticket.base_prediction == ticket.prediction == baseline
    np.testing.assert_array_equal(ticket.read_key, [1.0, 0.0])

    with pytest.raises(ValueError, match="matured"):
        runner.release(ticket.ticket_id, baseline + 2, 1)
    other = DelayedReplay(ResidualMemory(2, 0.5), horizon=2)
    with pytest.raises(ValueError, match="foreign"):
        other.release(ticket.ticket_id, baseline + 2, 2)
    assert runner.ledger.clock == 0
    assert int(runner.memory.writes) == 0
    assert runner.release(ticket.ticket_id, baseline + 2, 2) == 4
    next_ticket = issue_regression(
        runner,
        SequenceClassifierOutput(logits=original_logits),
        issue_hour=2,
        read_key=np.array([1.0, 0.0]),
        sample_index=1,
    )
    assert next_ticket.prediction == baseline + 1
    assert runner.ledger.pending[next_ticket.ticket_id] is next_ticket
    with pytest.raises(ValueError, match="already"):
        runner.release(ticket.ticket_id, baseline + 2, 2)
    assert int(runner.memory.writes) == 1
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, model_state[name])
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_preserves_selected_scalar_dtype_and_ignores_loss(dtype):
    scalar = 1.0000000000000002 if dtype == torch.float64 else 1.125
    logits = torch.tensor([[scalar], [float("nan")]], dtype=dtype, requires_grad=True)
    output = SequenceClassifierOutput(loss=torch.tensor(float("nan")), logits=logits)
    runner = DelayedReplay(ResidualMemory(1, 0.5), 1)
    ticket = issue_regression(
        runner, output, issue_hour=0, read_key=np.array([1.0]), sample_index=np.int64(0)
    )
    assert ticket.base_prediction == scalar
    assert logits.requires_grad
    assert ticket.prediction == scalar


@pytest.mark.parametrize(
    ("output", "sample_index", "error", "message"),
    [
        ({"logits": torch.zeros(1, 1)}, 0, TypeError, "SequenceClassifierOutput"),
        (
            TokenClassifierOutput(logits=torch.zeros(1, 2, 1)),
            0,
            TypeError,
            "SequenceClassifierOutput",
        ),
        (SequenceClassifierOutput(), 0, TypeError, "torch.Tensor"),
        (SequenceClassifierOutput(logits=np.zeros((1, 1))), 0, TypeError, "torch.Tensor"),
        (
            SequenceClassifierOutput(logits=torch.zeros(1, 1, dtype=torch.int64)),
            0,
            ValueError,
            "dtype",
        ),
        (
            SequenceClassifierOutput(logits=torch.zeros(1, 1, dtype=torch.complex64)),
            0,
            ValueError,
            "dtype",
        ),
        (SequenceClassifierOutput(logits=torch.zeros(0, 1)), 0, ValueError, "shape"),
        (SequenceClassifierOutput(logits=torch.zeros(1)), 0, ValueError, "shape"),
        (SequenceClassifierOutput(logits=torch.zeros(1, 2)), 0, ValueError, "shape"),
        (SequenceClassifierOutput(logits=torch.zeros(1, 2, 1)), 0, ValueError, "shape"),
        (SequenceClassifierOutput(logits=torch.zeros(1, 1)), True, ValueError, "integer"),
        (SequenceClassifierOutput(logits=torch.zeros(1, 1)), 0.0, ValueError, "integer"),
        (SequenceClassifierOutput(logits=torch.zeros(1, 1)), -1, ValueError, "outside"),
        (SequenceClassifierOutput(logits=torch.zeros(1, 1)), 1, ValueError, "outside"),
        (SequenceClassifierOutput(logits=torch.tensor([[float("nan")]])), 0, ValueError, "finite"),
        (SequenceClassifierOutput(logits=torch.tensor([[float("inf")]])), 0, ValueError, "finite"),
    ],
)
def test_invalid_outputs_and_rows_fail_before_issue(output, sample_index, error, message):
    runner = DelayedReplay(ResidualMemory(2, 0.5), 1)
    before = {name: value.clone() for name, value in runner.memory.state_dict().items()}
    with pytest.raises(error, match=message):
        issue_regression(
            runner, output, issue_hour=0, read_key=np.array([1.0, 0.0]), sample_index=sample_index
        )
    assert not runner.ledger.pending
    assert not runner.ledger.settled_ids
    assert runner.ledger.clock is None
    for name, value in runner.memory.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_row_is_required_and_existing_key_contract_is_preserved():
    runner = DelayedReplay(ResidualMemory(2, 0.5), 1)
    output = SequenceClassifierOutput(logits=torch.tensor([[1.0]]))
    with pytest.raises(TypeError, match="sample_index"):
        issue_regression(runner, output, issue_hour=0, read_key=np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="dtype/device"):
        issue_regression(
            runner,
            output,
            issue_hour=0,
            read_key=torch.ones(2, dtype=torch.float32),
            sample_index=0,
        )
    assert not runner.ledger.pending
    assert runner.ledger.clock is None
    ticket = issue_regression(
        runner,
        output,
        issue_hour=0,
        read_key=torch.tensor([1.0, 0.0], dtype=torch.float64),
        sample_index=0,
    )
    assert ticket.prediction == 1


def test_configured_correction_clip_is_delegated_to_runner():
    runner = DelayedReplay(ResidualMemory(1, 0.5), 1, correction_clip=0.25)
    runner.memory.values.fill_(4.0)
    ticket = issue_regression(
        runner,
        SequenceClassifierOutput(logits=torch.tensor([[10.0]])),
        issue_hour=0,
        read_key=np.array([1.0]),
        sample_index=0,
    )
    assert ticket.base_prediction == 10
    assert ticket.prediction == 10.25
    assert int(runner.memory.writes) == 0


def test_native_checkpoint_resumes_pending_and_settled_tickets(tmp_path):
    runner = DelayedReplay(ResidualMemory(2, 0.5), 2)
    output = SequenceClassifierOutput(logits=torch.tensor([[10.0]]))
    key = np.array([1.0, 0.0])
    settled = issue_regression(runner, output, issue_hour=0, read_key=key, sample_index=0)
    pending = issue_regression(runner, output, issue_hour=1, read_key=key, sample_index=0)
    runner.release(settled.ticket_id, 12, 2)
    path = tmp_path / "replay.pt"
    save_checkpoint(path, runner)
    resumed = load_checkpoint(path)
    assert resumed.ledger.owner_id == runner.ledger.owner_id
    assert resumed.ledger.clock == 2
    assert resumed.release(pending.ticket_id, 12, 3) == runner.release(pending.ticket_id, 12, 3)
    with pytest.raises(ValueError, match="already"):
        resumed.release(settled.ticket_id, 12, 3)
    next_resumed = issue_regression(resumed, output, issue_hour=3, read_key=key, sample_index=0)
    next_original = issue_regression(runner, output, issue_hour=3, read_key=key, sample_index=0)
    assert next_resumed.prediction == next_original.prediction == 11.5
    torch.testing.assert_close(resumed.memory.values, runner.memory.values)
