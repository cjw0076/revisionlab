# Source and result provenance

RevisionLab v0.1 derives from the recovered v13 delayed-credit reference at:

```text
labserver1_ts:/home/jaewon/workspaces/jaewon/mosaic/cosmos_model/v13_delayed_credit_20261001
```

The local read-only reference is the sibling `revisionlab-v13-reference` directory. Recovery verified byte-identical source/results before release refactoring. [The SHA-256 manifest](evidence/v13-source-sha256.json) records the recovered ten files: four package source files, the original contract test, online runner/verifier, and three result/selection JSON files.

The three copied historical JSON files match the source receipts:

| File | SHA-256 |
| --- | --- |
| `SUMMARY.json` | `19e17f6b90f2b06b2a999f500ac6c2a4a280e57eaafd0818b75255ab29b13301` |
| `INDEPENDENT_VERIFICATION.json` | `1893c592433cf1aa524e52ff357718a849bab5901303c586fa8aae3983890076` |
| `SELECTION_LOCK.json` | `627734a171542d53ad4c8dd5c80b13efcddfae6916e4f10bdf5b6eab6bbada0b` |

Release source has intentional safety and API changes. These hashes identify the recovered original, not the new release source. Raw weather data and model checkpoints are not included.

## Historical findings

The recovered `SUMMARY.json` reports frozen mean joint MSE `2.1220140728718335`, native-live `2.0921760203138478`, and RLS-live `2.0700443533561574`. Native improvement over frozen was approximately 1.406%, conditional on one site and one half-year. RLS had the lower mean MSE; this is not evidence of native residual superiority over RLS.

The recovered independent verifier reports maximum prediction/state error `1.7763568394002505e-15` against a separate replay recurrence. This is a prior-run receipt. The release work checked source contracts and artifacts; it did not execute a new full weather replay.

Five weight seeds are not five independent climates. Local residual cosine/alignment is not full-model BPTT alignment, and the historical v13 run does not validate v14 state splitting or v10 learned refinement.

The original shuffled-write control applied a fixed cyclic permutation (`np.roll(arange, 1)`); the release's seeded-permutation `ShuffledWrite` control is a different policy. Results for one must not be relabeled as results for the other.

## Name and scope

On 2026-10-01, the `revisionlab` PyPI JSON endpoint returned HTTP 404, while GitHub `revisionlab in:name` returned 18 repositories. The user chose RevisionLab as the release priority. The search is an availability snapshot, not name reservation or trademark clearance.

The package is separate from archcredit and has no dependency on it. CREDO and Cosmos v14 remain deferred research directions.
