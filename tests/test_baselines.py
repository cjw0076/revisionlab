import numpy as np
import pytest
import torch

from revisionlab.baselines import NoWrite, RLS_Corrector, ShuffledWrite
from revisionlab.protocols import ForecastTicket, ReleasedEvidence


def evidence(key, target, index=0, version=0):
    ticket = ForecastTicket(index, 1, 2.0, np.asarray(key), address_version=version)
    return ReleasedEvidence(ticket, target + 2.0, index + 1)


@pytest.mark.parametrize("forget", [0.9, 0.999, 1.0])
def test_rls_matches_independent_numpy_sequence(forget):
    memory = RLS_Corrector(3, forget)
    weights = np.zeros(3)
    covariance = np.eye(3) * 100
    sequence = [
        ([1, 0, 0], 4),
        ([0.5, 1, -0.2], -3),
        ([0, 1, 1], 100),
        ([1, -1, 2], -100),
        ([0, 0, 0], 2),
    ]
    for index, (key, residual) in enumerate(sequence):
        x = np.array(key, dtype=np.float64)
        projected = covariance @ x
        gain = projected / (forget + x @ projected)
        weights = weights + gain * (np.clip(residual, -12, 12) - x @ weights)
        covariance = (covariance - np.outer(gain, projected)) / forget
        covariance = (covariance + covariance.T) / 2
        memory.write(evidence(x, residual, index))
        np.testing.assert_allclose(memory.values.numpy(), weights, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(memory.P.numpy(), covariance, rtol=1e-12, atol=1e-12)
    assert int(memory.writes) == len(sequence)
    assert memory.values.dtype == torch.float64


@pytest.mark.parametrize("forget", [0, -1, 1.01, float("nan"), float("inf")])
def test_invalid_forgetting(forget):
    with pytest.raises(ValueError):
        RLS_Corrector(2, forget)


def test_failed_rls_write_is_atomic():
    memory = RLS_Corrector(2)
    memory.P.fill_(1e308)
    before = {name: tensor.clone() for name, tensor in memory.state_dict().items()}
    with pytest.raises(FloatingPointError):
        memory.write(evidence([1e100, 1e100], 2))
    for name, tensor in memory.state_dict().items():
        torch.testing.assert_close(before[name], tensor)


def test_rls_write_counter_overflow_is_atomic():
    memory = RLS_Corrector(2)
    memory.writes.fill_(torch.iinfo(torch.int64).max)
    before = memory.values.clone()
    with pytest.raises(OverflowError):
        memory.write(evidence([1, 0], 2))
    torch.testing.assert_close(before, memory.values)
    assert int(memory.writes) == torch.iinfo(torch.int64).max


@pytest.mark.parametrize("factory", [RLS_Corrector, NoWrite, ShuffledWrite])
def test_baselines_reject_invalid_address_version(factory):
    memory = factory(2)
    with pytest.raises(ValueError, match="version"):
        memory.write(evidence([1, 0], 2, version=1))
    assert int(memory.writes) == 0


def test_no_write_preserves_zero_exactly():
    memory = NoWrite(3)
    for index in range(10):
        memory.write(evidence([1, 2, 3], 5, index))
        assert memory.read(np.array([1.0, 2.0, 3.0])).item() == 0
    assert torch.equal(memory.values, torch.zeros(3, dtype=torch.float64))
    assert int(memory.writes) == 0


def test_shuffled_write_is_seeded_and_reads_original_address():
    rng = torch.random.get_rng_state().clone()
    first = ShuffledWrite(4, rate=1.0, seed=7)
    second = ShuffledWrite(4, rate=1.0, seed=7)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(first.permutation, second.permutation)
    assert not torch.equal(first.permutation, torch.arange(4))
    key = np.array([1.0, 0.0, 0.0, 0.0])
    first.write(evidence(key, 4))
    shuffled = torch.as_tensor(key)[first.permutation]
    torch.testing.assert_close(first.values, shuffled * 4)
    assert float(first.read(key)) == float(first.values[0])


def test_rls_dtype_contract_and_state_dict_forgetting():
    first = RLS_Corrector(2, 0.9)
    second = RLS_Corrector(2, 1)
    second.load_state_dict(first.state_dict())
    assert second.forget == pytest.approx(0.9)
    with pytest.raises(ValueError, match="dtype"):
        first.read(torch.ones(2, dtype=torch.float32))


def test_negative_tick_snapshot_does_not_confuse_sentinel():
    memory = RLS_Corrector(2)
    memory.write(evidence([1, 0], 2, index=-2))
    assert memory.snapshot()["last_label_hour"] == -1
