"""A synthetic delayed-residual smoke check, not forecasting validation."""

import argparse
import json
from collections import deque

import numpy as np
import torch

from .baselines import NoWrite, RLS_Corrector, ShuffledWrite
from .memory import ResidualMemory
from .replay import DelayedReplay


def synthetic_check(steps: int = 128, seed: int = 0, horizon: int = 3) -> dict[str, object]:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    generator = np.random.default_rng(seed)
    slots = 8
    addresses = generator.integers(slots, size=steps)
    noise = generator.normal(0, 0.03, size=steps)
    residuals = np.linspace(-2, 2, slots)[addresses] + noise
    methods: dict[str, ResidualMemory | RLS_Corrector] = {
        "residual": ResidualMemory(slots, 0.5),
        "rls": RLS_Corrector(slots, 0.999),
        "no-write": NoWrite(slots),
        "shuffled-write": ShuffledWrite(slots, 0.5, seed=seed),
    }
    results: dict[str, object] = {}
    for name, memory in methods.items():
        runner = DelayedReplay(memory, horizon)
        pending: deque[tuple[int, str, int]] = deque()
        for now in range(steps):
            # This evaluator queue owns targets; runner pending state contains
            # only issued tickets and never a not-yet-mature target.
            while pending and pending[0][0] <= now:
                _, ticket_id, index = pending.popleft()
                runner.release(ticket_id, float(residuals[index]), now)
            key = np.zeros(slots)
            key[addresses[now]] = 1
            ticket = runner.issue(now, 0.0, key)
            pending.append((ticket.available_hour, ticket.ticket_id, now))
        while pending:
            available, ticket_id, index = pending.popleft()
            runner.release(ticket_id, float(residuals[index]), available)
        corrected_mse = runner.ledger.mse
        base_mse = runner.ledger.base_mse
        assert corrected_mse is not None and base_mse is not None
        results[name] = {
            "base_mse": base_mse,
            "corrected_mse": corrected_mse,
            "mse_improvement": base_mse - corrected_mse,
            "releases": runner.ledger.releases,
            "model_tensor_bytes": sum(
                value.numel() * value.element_size() for value in memory.state_dict().values()
            ),
            "ledger": runner.ledger.budget(),
        }
    return {
        "scope": "synthetic stationary address-residual smoke; not real-world forecasting evidence",
        "steps": steps,
        "seed": seed,
        "horizon": horizon,
        "dtype": str(torch.float64),
        "methods": results,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="revisionlab-check")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable synthetic results"
    )
    args = parser.parse_args(argv)
    try:
        report = synthetic_check(args.steps, args.seed, args.horizon)
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(report["scope"])
        methods = report["methods"]
        assert isinstance(methods, dict)
        for name, row in methods.items():
            print(
                f"{name}: base MSE={row['base_mse']:.6f}; corrected MSE={row['corrected_mse']:.6f}"
            )


if __name__ == "__main__":
    main()
