# v0.1 method and replay manifest

## Residual memory

`ResidualMemory` stores one scalar value per address slot, initially zero. Its read is the inner product of the current values and key. For a matured ticket, the update target is `clip(target - base_prediction, -12, 12)`.

For value vector `v`, key `k`, residual target `r`, and learning rate `eta`, the normalized delta/LMS rule is:

```text
v_next = v + eta * k * (r - dot(k, v)) / max(dot(k, k), epsilon)
```

The learning rate must lie in `(0, 1]`. Stateful writes validate the proposed state before commit. The NumPy `SparseResidualMemory` preserves the scalar reference. The native backend defaults to float64; tensor keys must match its configured dtype/device, while NumPy ticket keys are explicitly copied into that backend.

The functional `delta_write(values, key, target, rate)` supports channel/batch dimensions, returns a new tensor, and does not clip the target or mutate input values. It is the differentiable primitive. Stateful correction updates run without constructing an autograd graph; this separation does not imply full-model credit assignment.

## Recursive least squares

`RLS_Corrector` starts with zero coefficients and covariance `P = 100 I`, default forgetting factor `lambda = 0.999`, and float64 state. It uses the same clipped residual target as native memory:

```text
gain = P @ k / (lambda + dot(k, P @ k))
v_next = v + gain * (r - dot(k, v))
P_next = (P - outer(gain, P @ k)) / lambda
```

The implementation symmetrizes the proposed covariance and checks finite candidates before commit. `lambda` must lie in `(0, 1]`. RLS maintains quadratic covariance state; include it when comparing resource budgets to fixed-slot delta memory.

## Controls and replay

`NoWrite` produces zero correction and performs no state updates. `ShuffledWrite` keeps read addresses intact but applies a seeded, nonidentity permutation to write addresses. This is a negative control, distinct from the original historical fixed cyclic permutation.

`DelayedReplay` clips the read correction to `[-3, 3]` by default and stores the actual issued prediction, baseline, key, owner/ticket identity, and horizon. At release it evaluates that issued forecast against the matured target, then applies a write and settles the ticket once. Release matured evidence before issuing the next forecast at the same clock tick. Event times must be monotonic integers; a label may arrive later than its maturity but never earlier.

Exactly-once release is enforced by the runner ledger. Direct low-level writes do not deduplicate. The runner retains settled IDs for its lifetime; corrector capacity is fixed, but the whole audit ledger is not constant-memory.

Checkpoint helpers support the four native built-ins only, restore on CPU, and preserve pending/settled ticket identity, owner, clock, memory, and evaluation totals. A custom `read`/`write`/`snapshot` implementation can use generic replay but is not automatically supported by checkpoint loading.

## Diagnostic scope

`credit_alignment` reports a single update/reference cosine; `aggregate_alignment` pools dot products and squared norms rather than averaging cosines. Zero norms are undefined. In the v13 residual experiment, the reference was a local residual squared-error direction. Alignment near one is expected algebraically for normalized delta updates against that reference; it is not evidence of full-model BPTT agreement or delayed credit superiority.
