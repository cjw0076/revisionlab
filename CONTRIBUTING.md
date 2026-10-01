# Contributing

Keep delayed-feedback behavior explicit. A corrector should be easy to replace without changing ticket issuance, maturity checks, or how forecasts are evaluated.

## Setup

Use Python 3.10 or newer:

```powershell
python -m pip install -e ".[dev]"
revisionlab-check
```

## Add a corrector in five minutes

1. Implement `read(key)`, `write(evidence)`, and `snapshot()` using the memory protocol.
2. Return a finite scalar correction from `read`; check key shape and finite values before updating.
3. Make `write` atomic: validate a candidate before committing it, and preserve state/counters if it fails.
4. Return copied state from `snapshot`.
5. Supply your corrector to generic `DelayedReplay` and test an issued ticket, exact maturity, duplicate release, and foreign owner.

Start with [examples/delayed_forecast.py](examples/delayed_forecast.py). Import `DelayedReplay` from `revisionlab.replay`; its constructor accepts your memory plus a horizon. `issue(issue_hour, base_prediction, read_key)` returns a ticket, and `release(ticket.ticket_id, target, now_hour)` evaluates the issued forecast before settling its update. Low-level correctors may apply repeated `write` calls; the runner owns exactly-once enforcement. Checkpoint helpers live in `revisionlab.serialization`.

This runnable custom bias corrector ignores the key and learns only a global residual offset:

```python
import math
import numpy as np
from revisionlab.replay import DelayedReplay


class BiasCorrector:
    def __init__(self):
        self.value = 0.0

    def read(self, key):
        return self.value

    def write(self, evidence):
        residual = evidence.target - evidence.ticket.base_prediction
        candidate = self.value + 0.1 * (residual - self.value)
        if not math.isfinite(candidate):
            raise ValueError("Nonfinite candidate")
        self.value = candidate

    def snapshot(self):
        return {"value": self.value}


runner = DelayedReplay(BiasCorrector(), horizon=2)
ticket = runner.issue(0, 1.0, np.array([1.0]))
assert runner.release(ticket.ticket_id, target=2.0, now_hour=2) == 1.0
assert runner.memory.snapshot() == {"value": 0.1}
```

Built-in checkpoint helpers support only `ResidualMemory`, `RLS_Corrector`, `NoWrite`, and `ShuffledWrite`. They reject custom correctors; `snapshot()` alone is not a resumable checkpoint contract. Supporting your custom corrector's persistence is separate work, not a five-minute registration promise.

Use a negative control name that describes what it does. No-write and shuffled-write controls must not be described as trained alternatives or evidence of improved memory. Keep prediction-time keys and issued forecasts fixed when feedback arrives; do not recompute past forecasts using current state.

The runner retains settled IDs for lifetime duplicate detection. Include that growing ledger and pending tickets in memory accounting instead of reporting only the fixed corrector state.

For a new rule, document its equation, forgetting/rate settings, clipping, initialization, state budget, and supported inputs. Distinguish a local residual update from full-model gradients. Compare identical inputs, baseline predictions, and released labels while scoring each policy's own actual issued forecast.

## Checks

```powershell
python -m ruff check .
python -m ruff format --check .
python -m mypy src/revisionlab
python -m pytest
python -m build
python -m twine check dist/*
```

Tests should expose causal leakage, failed-write mutation, duplicate settlement, and restore differences. Passing the original happy-path suite is insufficient: [the source audit](docs/source-audit.md) records failures found after 23 original tests passed. A separate reviewer evaluates author changes; authors do not self-approve.

Do not add raw weather data, large checkpoints, credentials, or generated environments to the package. Keep historical receipts separate from newly executed results. Follow [the Code of Conduct](CODE_OF_CONDUCT.md).
