"""Mature previous evidence before issuing the next forecast at the same tick."""

import numpy as np

from revisionlab import ResidualMemory
from revisionlab.replay import DelayedReplay


def main() -> None:
    runner = DelayedReplay(ResidualMemory(2, 0.5), horizon=2)
    first = runner.issue(0, base_prediction=10.0, read_key=np.array([1.0, 0.0]))
    print("Issued:", first.prediction)
    runner.issue(1, base_prediction=10.0, read_key=np.array([0.0, 1.0]))
    loss = runner.release(first.ticket_id, target=12.0, now_hour=2)
    next_ticket = runner.issue(2, base_prediction=10.0, read_key=np.array([1.0, 0.0]))
    print("Actual issued squared error:", loss)
    print("Next corrected forecast:", next_ticket.prediction)
    print("Unreleased tickets:", len(runner.ledger.pending))


if __name__ == "__main__":
    main()
