# Original source audit and release hardening

The recovered original passed all 23 contract tests in an independent local audit. Adversarial probes then exposed behavior outside those happy paths. Passing the original suite therefore did not establish release safety.

| Original behavior | Required release contract |
| --- | --- |
| NumPy ticket write flag could be reenabled | Prediction-time keys remain immutable even when accessed by callers. |
| NaN baseline accepted | Reject nonfinite baseline values before issuing evidence. |
| NaN/infinite timestamp bypassed maturity | Require valid integer timestamps and enforce exact maturity. |
| Same evidence wrote twice (`0.1` became `0.19`) | Owner-bound ticket identity and exactly-once settlement. |
| Failed write corrupted state before raising | Validate candidate state before a single atomic commit. |
| Snapshot could not resume | Versioned checkpoint restores memory, owner, pending tickets, and settled identities. |

Release regression tests must cover these cases, including failed-write preservation of counters and clocks, foreign-owner evidence, duplicate feedback, delayed labels across checkpoint boundaries, and resumed/uninterrupted equivalence. Current test receipts and implementation status belong in [STATE.md](../STATE.md).

Exactly-once enforcement belongs to the owner-bound `DelayedReplay` ledger. Low-level corrector `write` intentionally applies each valid call and has no duplicate-delivery guarantee by itself. The runner's settled-ID ledger is retained for lifetime deduplication, so total audit storage grows with issued/settled forecasts even when corrector values have fixed capacity.

The original compile test used `backend="eager"`; v0.1 acceptance adds CPU `aot_eager` smoke for the differentiable primitive. Neither establishes Inductor/GPU performance. The original local alignment tests checked cosine arithmetic and local residual directions, not global BPTT equivalence.

## Release review gates

A separate review of the new release found additional validation cases: checkpoint rate configuration must agree with its stored learning-rate buffer; unsupported checkpoint dtypes must be rejected before replacing a previous file; and finite but enormous keys must not overflow the norm and silently bypass an update. Release regression tests now cover these failures, previous-file/state preservation, and consistency between a shuffled control's seed and stored permutation. Final validation and separate review receipts are recorded in [STATE.md](../STATE.md), separately from the original 23-test audit and historical weather receipts.
