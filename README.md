# RevisionLab

**PyTorch-native delayed-feedback memory contracts, controls, and diagnostics.**

RevisionLab records a prediction when it is issued, waits for its outcome to mature, and applies a residual-state update to the memory that owned the prediction. It provides a small substrate for testing delayed correction: immutable tickets, explicit memory interfaces, replay, negative controls, and resumable state.

v0.1 focuses on fixed-capacity residual memory and RLS correction. It does not implement Cosmos v14 state splitting or the v10 learned-refinement toy. It makes no claim to support every neural architecture. [archcredit](https://github.com/cjw0076/archcredit) remains a separate architecture/credit benchmark library; RevisionLab does not depend on it.

## Install and check

With Python 3.10 or newer, install [v0.1.0 from PyPI](https://pypi.org/project/revisionlab/0.1.0/):

```powershell
python -m pip install revisionlab==0.1.0
revisionlab-check
```

The check runs a small synthetic delayed-feedback example. It does not download weather data or replay the historical experiment.

For development from a checkout:

```powershell
python -m pip install -e ".[dev]"
```

## Minimal API

```python
import numpy as np
from revisionlab import ResidualMemory
from revisionlab.replay import DelayedReplay

runner = DelayedReplay(ResidualMemory(2, rate=0.1), horizon=2)
ticket = runner.issue(0, 1.0, np.array([1.0, 0.0]))
error = runner.release(ticket.ticket_id, target=2.0, now_hour=2)
assert ticket.prediction == 1.0
assert error == 1.0
assert runner.memory.read(np.array([1.0, 0.0])) == 0.1
```

See [examples/delayed_forecast.py](examples/delayed_forecast.py) for a complete replay example. `revisionlab.serialization.save_checkpoint(path, runner)` and `load_checkpoint(path)` preserve delayed-feedback runners using the four native built-in correctors. Custom correctors and the NumPy reference are not supported by these checkpoint helpers in v0.1.

## Contracts

| Boundary | Purpose |
| --- | --- |
| Prediction ticket | Keeps prediction-time key, baseline, actual forecast, maturity, and owner identity. |
| Released evidence | Rejects targets that have not matured and invalid timestamp/target fields. |
| Memory `read`/`write`/`snapshot` | Reads a correction, applies valid delayed evidence, and copies state. |
| `DelayedReplay` | Owns tickets, evaluates the issued forecast, and settles feedback exactly once. |
| Checkpoint | Preserves memory, replay owner, pending tickets, and settled identities for resume. |
| Local alignment | Compares a correction update with a local residual-loss reference direction. |

The NumPy reference and PyTorch-native implementations use fixed-capacity state. Negative controls are explicitly named; shuffled writes corrupt write addressing rather than demonstrating a new learning rule. A no-write control must preserve zero correction and perform no learning.

`ResidualMemory(slots, rate)` is the native normalized-delta corrector. `revisionlab.baselines` provides `RLS_Corrector(slots, forget)`, `NoWrite(slots)`, and `ShuffledWrite(slots, rate, seed=...)`. See [the method manifest](docs/methods.md) for equations and clipping. Low-level memory `write` applies each valid call; **exactly-once settlement is a `DelayedReplay` guarantee**, not a guarantee of direct memory writes. Tickets retain prediction-time keys and forecasts, and the runner rejects duplicate/foreign settlement.

Memory values have fixed capacity. Replay retains settled ticket identities for lifetime deduplication, so its total audit ledger grows with settled forecasts; report that overhead alongside pending tickets. Do not call the entire runner constant-memory.

## What the evidence means

Recovered v13 artifacts report the following historical mean joint MSE:

| Historical policy | Mean joint MSE |
| --- | ---: |
| Frozen baseline | 2.122014 |
| Native live residual | 2.092176 |
| Live RLS | 2.070044 |

These are prior-run receipts for one site and the first half of 2026, across five weight seeds. The residual method improved on the frozen baseline in that recorded setting, while RLS had the lower mean MSE. This release has not rerun the full weather experiment, established superiority over RLS, or demonstrated generalization to other climates. See [provenance](docs/provenance.md) and [historical evidence](docs/evidence/README.md).

The original shuffled-write policy used a fixed cyclic permutation. The release's seeded `ShuffledWrite` control is a distinct policy and does not reproduce that historical control automatically.

Local residual alignment `rho` is not full-model BPTT alignment. The delta primitive is differentiable, but that alone does not prove correct global credit assignment or causal memory capacity. Zero-norm directions have undefined alignment.

## Development and contribution

```powershell
python -m ruff check .
python -m ruff format --check .
python -m mypy src/revisionlab
python -m pytest
python -m build
python -m twine check dist/*
```

[CONTRIBUTING.md](CONTRIBUTING.md) explains the five-minute corrector recipe. CI covers Python 3.10/3.11/3.12, lint/format, types, tests, `aot_eager` compile/serialization/gradcheck, and wheel installation outside the checkout. A CI definition is not a claim that hosted jobs have already passed; current receipts belong in [STATE.md](STATE.md).

See [release steps](docs/releasing.md) for TestPyPI/PyPI handoff and [the source audit](docs/source-audit.md) for adversarial regression requirements. A manual-dispatch Trusted Publisher pipeline is provided; publisher registration and verified upload status belong in [STATE.md](STATE.md). Licensed under [MIT](LICENSE); participation follows the [Code of Conduct](CODE_OF_CONDUCT.md).

Local v0.1.0 validation: **95 tests passed, 1 CUDA hardware skip**; lint/format, strict package types, and separate reviews passed. See [verification evidence](docs/verification.md) and [release notes](docs/release-notes-v0.1.0.md). Hosted and publication receipts are recorded separately.

Research proposal: [chaotic data and delayed correction](docs/research-chaotic-data.md) separates an unverified Cosmos hypothesis from current fixed-memory capabilities and outlines falsifiable comparisons.
