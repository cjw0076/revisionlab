import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch

from revisionlab import (
    ForecastTicket,
    MemoryProtocol,
    ReleasedEvidence,
    ResidualMemory,
    SparseResidualMemory,
    aggregate_alignment,
    credit_alignment,
    delta_write,
)


def ticket(key=None, **kwargs):
    return ForecastTicket(100, 6, 1.0, np.array([0.25, 0.75]) if key is None else key, **kwargs)


def test_ticket_copies_and_truly_freezes_key():
    key = np.array([1.0, 0.0])
    forecast = ticket(key, prediction=1.2, ticket_id="known")
    key[0] = 0
    assert forecast.read_key[0] == 1
    assert forecast.prediction == 1.2
    assert forecast.ticket_id == "known"
    with pytest.raises(ValueError):
        forecast.read_key[0] = 3
    with pytest.raises(ValueError):
        forecast.read_key.setflags(write=True)
    source = torch.tensor([0.1, 0.9], requires_grad=True)
    converted = ticket(source)
    with torch.no_grad():
        source.fill_(0)
    assert converted.read_key.sum() == pytest.approx(1)
    assert ticket().ticket_id != ticket().ticket_id


def test_ticket_key_metadata_cannot_change_stored_semantics():
    forecast = ticket(np.array([0.25, 0.75]))
    exposed = forecast.read_key
    exposed.dtype = np.int64
    exposed.shape = (1, 2)
    assert forecast.read_key.dtype == np.float64
    assert forecast.read_key.shape == (2,)
    np.testing.assert_array_equal(forecast.read_key, [0.25, 0.75])
    assert forecast.read_key is not forecast.read_key
    with pytest.raises(FrozenInstanceError):
        forecast.read_key = np.ones(2)
    shuffled = replace(forecast, read_key=forecast.read_key[::-1])
    np.testing.assert_array_equal(shuffled.read_key, [0.75, 0.25])
    assert shuffled.ticket_id == forecast.ticket_id


@pytest.mark.parametrize("field", ["issue_hour", "horizon", "address_version"])
@pytest.mark.parametrize("invalid", [True, 1.5, float("nan")])
def test_integral_ticket_ticks(field, invalid):
    arguments = dict(issue_hour=100, horizon=6, base_prediction=1.0, read_key=np.ones(2))
    arguments[field] = invalid
    with pytest.raises(ValueError, match="integral"):
        ForecastTicket(**arguments)


@pytest.mark.parametrize("key", [[], [[1, 2]], [float("nan"), 1], [float("inf")]])
def test_invalid_keys(key):
    with pytest.raises(ValueError, match="finite nonempty"):
        ticket(key)


@pytest.mark.parametrize("field", ["base_prediction", "prediction"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), [1], True])
def test_finite_scalar_predictions(field, value):
    arguments = dict(issue_hour=0, horizon=1, base_prediction=0.0, read_key=np.ones(2))
    arguments[field] = value
    with pytest.raises(ValueError, match="finite scalar"):
        ForecastTicket(**arguments)


def test_evidence_maturity_and_scalar_target():
    forecast = ticket()
    with pytest.raises(ValueError, match="matured"):
        ReleasedEvidence(forecast, 3, 105)
    assert ReleasedEvidence(forecast, 3, np.int64(106)).now_hour == 106
    for now in (True, 106.0):
        with pytest.raises(ValueError, match="integral"):
            ReleasedEvidence(forecast, 3, now)
    for target in (float("nan"), float("inf"), [3]):
        with pytest.raises(ValueError, match="finite scalar"):
            ReleasedEvidence(forecast, target, 106)


@pytest.mark.parametrize("backend", [SparseResidualMemory, ResidualMemory])
def test_normalized_effect_and_snapshot_no_alias(backend):
    memory = backend(2, 0.1)
    assert isinstance(memory, MemoryProtocol)
    evidence = ReleasedEvidence(ticket(), 3, 110)
    memory.write(evidence)
    assert float(memory.read(evidence.ticket.read_key)) == pytest.approx(0.2)
    snapshot = memory.snapshot()
    assert snapshot["writes"] == 1
    assert snapshot["last_label_hour"] == 106
    snapshot["values"][0] = 999
    assert float(memory.values[0]) != 999
    # The backend has no duplicate ledger: ownership is the runner's concern.
    memory.write(evidence)
    assert memory.snapshot()["writes"] == 2


@pytest.mark.parametrize("backend", [SparseResidualMemory, ResidualMemory])
def test_write_validation_is_atomic(backend):
    memory = backend(2, 0.1)
    before = memory.snapshot()
    for forecast in (ticket(np.ones(3)), ticket(address_version=1)):
        with pytest.raises(ValueError):
            memory.write(ReleasedEvidence(forecast, 3, 106))
        np.testing.assert_array_equal(memory.snapshot()["values"], before["values"])
        assert memory.snapshot()["writes"] == before["writes"]
        assert memory.snapshot()["last_label_hour"] == before["last_label_hour"]
    if isinstance(memory, ResidualMemory):
        memory.values.fill_(1e308)
    else:
        memory.values.fill(1e308)
    before = memory.snapshot()
    with pytest.raises(FloatingPointError, match="proposal"):
        memory.write(ReleasedEvidence(ticket(np.array([2.0, 2.0])), 3, 106))
    np.testing.assert_array_equal(memory.snapshot()["values"], before["values"])
    assert memory.snapshot()["writes"] == 0


@pytest.mark.parametrize("backend", [SparseResidualMemory, ResidualMemory])
def test_overflow_key_norm_rejects_without_partial_write(backend):
    memory = backend(1, 1.0)
    memory.write(ReleasedEvidence(ForecastTicket(0, 1, 0.0, np.ones(1)), 0.25, 1))
    before = memory.snapshot()
    overflow = ForecastTicket(1, 1, 0.0, np.array([1e200]))
    with pytest.raises(FloatingPointError, match="squared norm"):
        memory.write(ReleasedEvidence(overflow, 1.0, 2))
    np.testing.assert_array_equal(memory.snapshot()["values"], before["values"])
    assert memory.snapshot()["writes"] == before["writes"]
    assert memory.snapshot()["last_label_hour"] == before["last_label_hour"]


def test_native_metadata_overflow_is_atomic():
    memory = ResidualMemory(2, 0.1)
    for issue in (2**63, 2**63 - 1):
        with pytest.raises(ValueError, match="int64 tick range"):
            ForecastTicket(issue, 1, 0.0, np.ones(2))
    memory.writes.fill_(2**63 - 1)
    with pytest.raises(ValueError, match="metadata range"):
        memory.write(ReleasedEvidence(ticket(), 3, 106))
    assert torch.count_nonzero(memory.values) == 0
    assert memory.writes.item() == 2**63 - 1


def test_negative_tick_snapshot_and_bfloat_ticket():
    memory = ResidualMemory(2, 0.1)
    forecast = ForecastTicket(-2, 1, 0.0, torch.ones(2, dtype=torch.bfloat16))
    memory.write(ReleasedEvidence(forecast, 1, -1))
    assert memory.snapshot()["last_label_hour"] == -1


@pytest.mark.parametrize(
    "slots,rate", [(True, 0.1), (2.5, 0.1), (0, 0.1), (2, 0), (2, 1.1), (2, math.nan)]
)
def test_invalid_state_configuration(slots, rate):
    for backend in (SparseResidualMemory, ResidualMemory):
        with pytest.raises(ValueError):
            backend(slots, rate)


@pytest.mark.parametrize("permutation", [[0, 0], [0.0, 1.0], [True, False], [[0, 1]]])
def test_strict_permutation(permutation):
    with pytest.raises(ValueError, match="bijection"):
        SparseResidualMemory(2, 0.1, np.asarray(permutation))


def test_numpy_native_pure_parity_and_clipping():
    numpy_memory = SparseResidualMemory(2, 0.05)
    native = ResidualMemory(2, 0.05)
    generator = np.random.default_rng(7)
    for index in range(10):
        key = generator.random(2)
        forecast = ForecastTicket(index, 2, 1.0, key)
        target = float(generator.normal() * 30)
        evidence = ReleasedEvidence(forecast, target, index + 2)
        before = native.values.clone()
        pure = delta_write(
            before[:, None],
            torch.tensor(key, dtype=torch.float64),
            torch.tensor([np.clip(target - 1, -12, 12)], dtype=torch.float64),
            0.05,
        )
        numpy_memory.write(evidence)
        native.write(evidence)
        np.testing.assert_allclose(
            native.values.numpy(), numpy_memory.values, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(native.values, pure[:, 0])


def test_native_tensor_dtype_shape_and_state_roundtrip(tmp_path):
    memory = ResidualMemory(2, 0.1)
    assert memory.values.dtype == torch.float64
    assert memory.read(torch.ones(2, dtype=torch.float64)).ndim == 0
    with pytest.raises(ValueError, match="dtype/device"):
        memory.read(torch.ones(2, dtype=torch.float32))
    with pytest.raises(ValueError, match="address width"):
        memory.read(torch.ones(3, dtype=torch.float64))
    memory.write(ReleasedEvidence(ticket(), 3, 106))
    path = tmp_path / "native.pt"
    torch.save(memory.state_dict(), path)
    restored = ResidualMemory(2, 0.7)
    restored.load_state_dict(torch.load(path, weights_only=True))
    for key, value in memory.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    assert restored.snapshot()["last_label_hour"] == 106
    assert restored.rate == 0.1
    assert not memory.values.requires_grad
    assert memory.values.grad_fn is None


def test_delta_gradcheck_no_mutation_and_zero_key():
    torch.manual_seed(8)
    values = torch.randn(2, 3, 2, dtype=torch.double, requires_grad=True)
    key = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    target = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
    before = values.detach().clone()
    assert torch.autograd.gradcheck(
        lambda v, k, t: delta_write(v, k, t, 0.02), (values, key, target)
    )
    torch.testing.assert_close(values, before)
    torch.testing.assert_close(delta_write(values, torch.zeros_like(key), target, 0.1), values)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "rate", "integral", "empty"])
def test_delta_contract_checks(invalid):
    values = torch.zeros(2, 1)
    key = torch.ones(2)
    target = torch.ones(1)
    rate = 0.1
    if invalid == "shape":
        key = torch.ones(1, 2)
    elif invalid == "dtype":
        key = key.double()
    elif invalid == "rate":
        rate = math.inf
    elif invalid == "integral":
        values, key, target = values.long(), key.long(), target.long()
    else:
        values, key = torch.zeros(0, 1), torch.zeros(0)
    with pytest.raises(ValueError):
        delta_write(values, key, target, rate)


def test_delta_compile_fullgraph_forward_backward():
    compiled = torch.compile(delta_write, backend="aot_eager", fullgraph=True)
    values = torch.randn(2, 3, 2, requires_grad=True)
    key = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 2, requires_grad=True)
    actual = compiled(values, key, target, 0.1)
    expected = delta_write(values, key, target, 0.1)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), (values, key, target))
    expected_grad = torch.autograd.grad(expected.sum(), (values, key, target))
    for first, second in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(first, second)


def test_alignment_pooled_formula_undefined_and_stability():
    updates = np.array([[1.0, 0], [0, 10]])
    gradients = np.array([[1.0, 0], [0, -10]])
    assert aggregate_alignment(updates, gradients)["rho"] == pytest.approx(-99 / 101)
    assert credit_alignment([0, 0], [1, 2]) == {"defined": False, "cosine": None}
    for scale in (1e-300, 1e300):
        assert credit_alignment(np.array([2, 4]) * scale, np.array([1, 2]) * scale)[
            "cosine"
        ] == pytest.approx(1)
    tensor = torch.tensor([2.0, 4], requires_grad=True)
    assert credit_alignment(tensor, tensor)["cosine"] == pytest.approx(1)
    for first, second in (([1, 2], [1]), ([math.nan], [1]), ([math.inf], [1])):
        with pytest.raises(ValueError):
            credit_alignment(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_native_and_diagnostics():
    memory = ResidualMemory(2, 0.1, device="cuda")
    key = torch.tensor([0.25, 0.75], dtype=torch.float64, device="cuda", requires_grad=True)
    forecast = ticket(key)
    memory.write(ReleasedEvidence(forecast, 3, 106))
    assert float(memory.read(key)) == pytest.approx(0.2)
    assert credit_alignment(key, key)["cosine"] == pytest.approx(1)
    with pytest.raises(ValueError, match="dtype/device"):
        memory.read(key.cpu())
