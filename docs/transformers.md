# Delayed forecasts from Transformers regression outputs

This optional integration connects one scalar model output to the existing delayed-feedback runner. It accepts a Hugging Face [`SequenceClassifierOutput`](https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput), whose `logits` represent regression predictions when the model has `num_labels=1`. Use `return_dict=True` so the output is an object with named fields. A tuple's first entry can be a loss when labels are present, so the adapter deliberately reads `.logits`.

The adapter is available in the repository checkout. Published `revisionlab==0.1.0` predates this integration; installing that release will not provide this module or optional extra. From the repository root:

```powershell
python -m pip install -e ".[transformers]"
python examples/transformers_delayed_regression.py
```

The extra installs `transformers>=4.57.6,<6` and `torch>=2.5`. Core dependencies remain `torch>=2.2` and `numpy>=1.24`; core imports do not load Transformers. The adapter is imported explicitly from its optional module.

## API

```python
from revisionlab.integrations.transformers import issue_regression

ticket = issue_regression(
    runner,  # existing DelayedReplay
    output,  # SequenceClassifierOutput
    issue_hour=0,  # integer event tick, same contract as runner.issue
    read_key=key,  # explicit caller-owned memory address
    sample_index=0,  # required batch row
)
```

`output.logits` must be a floating Torch tensor shaped `[batch, 1]` with a nonempty batch. Multiclass logits, token outputs, tuple/dict outputs, missing logits, integer/complex tensors, out-of-range rows, Boolean/noninteger rows, and nonfinite selected values are rejected before issuing a ticket. Only the selected scalar is validated for finiteness; other batch rows are not issued by this call. To issue multiple forecasts, call once per intended row with its own explicit key and event time, respecting the runner's sequential event ordering.

The scalar is detached and copied to a Python float without first casting to float32, preserving float64 and bfloat16 represented values. A GPU scalar requires a host synchronization. The adapter does not mutate output tensors, model parameters, or gradients. It ignores `output.loss` and never receives a target at issue time. Supplying labels to the model may calculate a loss, but it does not update RevisionLab state; forecasting before target arrival should normally omit labels.

`read_key` is a one-dimensional NumPy float64 array or Torch tensor with the chosen memory's address width. Prefer an explicit NumPy float64 key for simple use, for example `np.array([1.0, 0.0], dtype=np.float64)`. Native correctors copy NumPy keys into their configured dtype/device. Tensor keys must already match the memory's exact dtype/device (default native memory is CPU float64); the adapter does not cast them. Existing backend validation remains authoritative. A key is a memory address, not an automatically pooled hidden state, token identifier, or learned model representation.

The adapter delegates to `runner.issue`: it reads the current correction, applies the configured correction clipping, and freezes the baseline, issued prediction, address, maturity, and owner in a `ForecastTicket`. Issuing does not write residual memory. The runner is sequential and not thread-safe; serialize calls.

## Matured feedback and resume

```python
# runner = DelayedReplay(ResidualMemory(2, rate=0.5), horizon=2)
# ticket was issued at tick 0 with key [1, 0]
loss = runner.release(ticket.ticket_id, target=observed_target, now_hour=2)
next_ticket = issue_regression(runner, next_output, issue_hour=2, read_key=key, sample_index=0)
```

Release matured outcomes before issuing new forecasts at the same tick. `release` evaluates the recorded issued prediction, updates its owning memory, and settles exactly once. Existing early/foreign/duplicate-release checks apply unchanged. The delayed update uses the saved issue-time key even if you later mutate the original array or model output. Targets must share the model prediction's units and are supplied by the application when they become available.

Existing `revisionlab.serialization.save_checkpoint(path, runner)` and `load_checkpoint(path)` preserve supported native memory, owner, clock, pending tickets, and settled IDs. The adapter stores no additional state. Save and restore the Transformers model separately if needed; RevisionLab's checkpoint does not serialize model parameters or its outputs. After loading the runner, use a fresh output with the same `issue_regression` function. Settled-ID audit storage still grows with releases.

## Offline example and validation

[The example](../examples/transformers_delayed_regression.py) uses [`BertConfig` and `BertForSequenceClassification`](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification), a fixed seed, hardcoded input IDs, a caller-defined address, and a synthetic target that arrives later. It constructs the model from a tiny local configuration, sets `num_labels=1` and `problem_type="regression"`, and uses evaluation mode. There is no `from_pretrained`, tokenizer, dataset, or Hub download. It asserts issue-time state preservation and the expected next residual correction. This is a contract demonstration with a randomly initialized model, not a regression accuracy result.

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python -m pytest tests/test_transformers.py tests/test_transformers_optional.py
python -m mypy src/revisionlab examples/transformers_delayed_regression.py
python examples/transformers_delayed_regression.py
```

Core test runs may skip the real-model test module when Transformers is absent. Dedicated integration CI installs and explicitly imports Transformers before running those tests, so dependency absence cannot produce a false green result. The tests deny socket connections while exercising the model and adapter. CI checks 4.57.6 and 5.18.0; that version matrix is a check definition until run receipts are recorded. Offline environment variables follow [Hugging Face's offline guidance](https://huggingface.co/docs/transformers/installation#offline-mode).

## Scope

This is inference-time scalar residual correction with delayed evidence through existing `MemoryProtocol` implementations and `DelayedReplay`. It provides no Trainer callback, gradient updates to BERT, causal-language-model/token correction, automatic address learning, model training benchmark, Cosmos state expansion, or full-model credit-assignment claim. Existing local residual alignment diagnostics retain their original meaning. [archcredit](https://github.com/cjw0076/archcredit) remains a separate project.
