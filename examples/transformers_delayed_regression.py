"""Offline contract demonstration with a tiny, randomly initialized BERT regressor.

Run from a checkout installed with ``python -m pip install -e '.[transformers]'``.
No tokenizer, pretrained weights, dataset, or Hub download is used.
"""

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification

from revisionlab import ResidualMemory
from revisionlab.integrations.transformers import issue_regression
from revisionlab.replay import DelayedReplay


def main() -> None:
    torch.manual_seed(7)
    config = BertConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=16,
        num_labels=1,
        problem_type="regression",
    )
    model = BertForSequenceClassification(config).eval()
    inputs = torch.tensor([[1, 4, 2], [1, 6, 2]])
    with torch.no_grad():
        output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), return_dict=True)

    runner = DelayedReplay(ResidualMemory(2, rate=0.5), horizon=2)
    # Caller-defined fixed memory address; it is independent of BERT's hidden states.
    key = np.array([1.0, 0.0], dtype=np.float64)
    first = issue_regression(runner, output, issue_hour=0, read_key=key, sample_index=0)
    assert first.prediction == first.base_prediction
    assert int(runner.memory.writes) == 0  # Issuing a forecast never learns from a target.

    # Synthetic target arrives two ticks later. Release before issuing at the same tick.
    target = first.base_prediction + 2.0
    loss = runner.release(first.ticket_id, target=target, now_hour=2)
    next_ticket = issue_regression(runner, output, issue_hour=2, read_key=key, sample_index=0)
    assert loss == 4.0
    assert next_ticket.prediction == next_ticket.base_prediction + 1.0
    assert int(runner.memory.writes) == 1
    print(f"Issued baseline: {first.base_prediction:.6f}")
    print(f"Matured squared error: {loss:.6f}")
    print(f"Next corrected forecast: {next_ticket.prediction:.6f}")
    print("Synthetic contract example; no model training or performance claim.")


if __name__ == "__main__":
    main()
