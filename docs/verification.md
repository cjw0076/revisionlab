# v0.1.0 verification

Recorded on 2026-10-01. The release was checked locally with Windows 11, Python 3.11.9, PyTorch 2.14.0+cpu, and NumPy 1.26.4.

| Check | Result |
| --- | --- |
| Tagged-release local tests | 95 passed, 1 CUDA hardware skip, 196.79 seconds. |
| Hosted matrix | All Python 3.10/3.11/3.12 jobs passed; each 95 passed and 1 CUDA hardware skip, including lint/types/build/Twine/installed-wheel checks. |
| Lint and formatting | Ruff passed; 21 files formatted. |
| Types | Strict mypy passed for 8 package source files. |
| Independent code review | APPROVE, zero remaining issues after hardening/regression fixes. |
| Independent scientific documentation review | ACCEPT. |
| Examples | README/custom-corrector snippets and delayed forecast example passed. |
| CLI | Literal installed `revisionlab-check` passed with finite synthetic results. |
| Compile/autograd | CPU `aot_eager` smoke and differentiable delta `gradcheck` passed in the suite. |
| Serialization | Checkpoint/replay resume and malformed-state regressions passed in the suite. |
| Packaging | Wheel/source builds, Twine metadata 2.4 checks, and outside-checkout installed-wheel import/CLI checks passed. |

Authoring and reviews ran in separate lanes. The local hardware skip is disclosed; these results do not establish CUDA or Inductor support/performance. Compile smoke concerns the functional delta primitive, not a claim that the whole Python ticket runner is compiled.

## Published source and artifacts

[v0.1.0](https://github.com/cjw0076/revisionlab/releases/tag/v0.1.0) is published as a research prerelease with wheel, source distribution, and `SHA256SUMS`. The tagged source is [`58567cc162f2d290be10c8e26ca251bbf6d3ccbc`](https://github.com/cjw0076/revisionlab/commit/58567cc162f2d290be10c8e26ca251bbf6d3ccbc); [CI run 36822956187](https://github.com/cjw0076/revisionlab/actions/runs/36822956187) passed all three jobs for that release source.

| Published artifact | SHA-256 |
| --- | --- |
| `revisionlab-0.1.0-py3-none-any.whl` | `fd9500032333fd2cdb1a990e39f9c1d4c5e8b7556036eb79f167277dc73704e8` |
| `revisionlab-0.1.0.tar.gz` | `e8529a4c07ae9cb3672fe96168bb7ab71bdd205ddb428289285b1e54690d121f` |

These receipts identify the tested source and published artifact bytes. This postrelease report does not change those artifacts or imply that a later documentation commit belongs to the original tagged snapshot.

The initial hosted matrix exposed NumPy typing compatibility gaps. The released fix uses each runner's actual Python version for strict mypy checks and explicitly annotates float64 NumPy state. The successful hosted matrix above verifies those fixes.

## Trusted publishing pipeline receipt

The publishing execution uses workflow-main revision [`fd98444cd47d00975cbef79b6e5e9be597cfe86e`](https://github.com/cjw0076/revisionlab/commit/fd98444cd47d00975cbef79b6e5e9be597cfe86e). Packaged source remains tagged revision `58567cc162f2d290be10c8e26ca251bbf6d3ccbc`; the workflow commit is not the packaged source commit. Both GitHub publishing environments are configured with main-only deployment restrictions.

- Current local suite: **121 passed, 1 CUDA hardware skip, 41.59 seconds**, including 26 release-verifier tests.
- Ruff, strict mypy for 8 package files, Twine, original published-artifact hashes, and fresh installed-wheel CLI/four native checkpoint checks passed. Separate independent monitor approval was collected.
- [CI 36826566815](https://github.com/cjw0076/revisionlab/actions/runs/36826566815) passed all Python 3.10/3.11/3.12 jobs, each with 121 passed and 1 CUDA hardware skip. [Manual validation 36826580022](https://github.com/cjw0076/revisionlab/actions/runs/36826580022) succeeded in 58 seconds with fresh dependency/wheel, four checkpoint-resume, and CLI checks; this was validation mode, not publication.

[The historical TestPyPI attempt](https://github.com/cjw0076/revisionlab/actions/runs/36826767156) passed validation but failed OIDC exchange with `invalid-publisher` before matching registration existed. No upload occurred in that attempt.

[TestPyPI run 36836398900](https://github.com/cjw0076/revisionlab/actions/runs/36836398900) and [production PyPI run 36836731138](https://github.com/cjw0076/revisionlab/actions/runs/36836731138) both passed validation, publication, and index verification. [TestPyPI v0.1.0](https://test.pypi.org/project/revisionlab/0.1.0/) and [PyPI v0.1.0](https://pypi.org/project/revisionlab/0.1.0/) serve the original wheel/source distribution. Independent index/byte downloads matched both SHA-256 values above and confirmed non-yanked files. Original source, version, and GitHub artifact bytes remain preserved.

The independent monitor cryptographically verified all four publication attestations (wheel/source on both indexes) with official `pypi-attestations==0.0.30`; each verification returned `OK` and exit 0. Public provenance matched repository `cjw0076/revisionlab`, workflow `publish.yml`, the respective `testpypi`/`pypi` environment, and signed artifact filename/digest. This proves signed publication identity and bytes, not an independently reproducible build; workflow-main `fd98444` did not rebuild the packaged source `58567cc`. See [the public receipt](../artifacts/publication-v0.1.0.json).

## Unreleased Transformers integration branch

Branch `codex/hf-delayed-regression` is based on main `7f1de7c0b57dab194a7bf420d4f30312e7ccbdac`. These branch checks are separate from the immutable v0.1.0 publication receipts above. The branch adds one optional scalar-regression-output adapter, a runnable offline tiny BERT example, and API documentation. Existing delayed replay, memory, checkpoints, scientific claims, and archcredit's role are unchanged.

| Branch check | Observed result |
| --- | --- |
| Fresh isolated core environment | Transformers confirmed absent; 123 passed, 2 skipped, 30.26 seconds. Skips are CUDA hardware and the optional real-model test module. |
| Core types | Strict mypy passed for 10 package source files without Transformers installed. |
| Preliminary existing environment | Transformers 4.57.1: 24 integration/import-isolation tests passed in 84.52 seconds; offline example passed; strict package/example types passed for 11 files. This is not selected-version matrix evidence. |
| Selected Transformers 5.18.0 | Isolated Python 3.11 / CPU PyTorch 2.14.1: 24 offline integration/import-isolation tests passed in 26.34 seconds; strict package/example types passed for 11 files; offline example passed. |
| Selected Transformers 4.57.6 | Same isolated runtime: 24 offline integration/import-isolation tests passed in 30.11 seconds; strict package/example types passed for 11 files; offline example and `pip check` passed. |
| Packaging | Fresh isolated branch build and Twine passed; wheel contains adapter modules and preserves torch/numpy-only core requirements. Non-editable candidate-wheel installation passed outside-checkout site-packages/source-path assertions and the offline example (exit 0). |
| Release preservation | Original published wheel/source hashes and v0.1.0 tag unchanged, independently confirmed. No new artifact upload occurred. |

The actual model tests deny socket connections and exercise frozen issue-time values/keys, memory/model immutability at issue, mature/foreign/duplicate release, dtype preservation, malformed-output rejection before issue, correction clipping, native checkpoint resume, and optional-import isolation. The example uses a random model and synthetic target; passing is contract evidence, not model accuracy or training evidence.

Dedicated offline CI jobs require explicit Transformers imports before tests, preventing a missing dependency from silently skipping that lane. [Draft PR #1](https://github.com/cjw0076/revisionlab/pull/1) has [successful PR CI 36855556931](https://github.com/cjw0076/revisionlab/actions/runs/36855556931) for code commit `859210483920c28bf5d548569bdbd31cfdb4b4ce`: all five jobs passed. Python 3.10/3.11/3.12 each recorded 123 passed and 2 skips (3.12 additionally recorded 2 warnings); Transformers 4.57.6 recorded 24 passed in 6.00 seconds and 5.18.0 recorded 24 passed in 4.74 seconds. [Push CI 36855537282](https://github.com/cjw0076/revisionlab/actions/runs/36855537282) also passed. Later documentation-only receipt commits are not included in these tested-code claims. Independent review returned APPROVE with zero blockers.

PyPI 0.1.0 predates this adapter and extra. Installation instructions for this code use the repository checkout, without replacing the published files.

## Evidence limits

`revisionlab-check` is a synthetic stationary address/residual smoke. It confirms executable delayed correction and controls, not real-world forecasting quality. The v13 metrics and independent replay JSON are byte-preserved historical receipts; this release did not run a new full weather replay.

Local residual alignment is not full-model BPTT alignment. Fixed-capacity memory does not imply constant total runner storage: settled ticket IDs grow with the lifetime audit ledger. Exactly-once settlement is a `DelayedReplay` contract; low-level corrector writes apply every valid call. Checkpoint helpers support the four native built-ins, not arbitrary custom correctors.

Package publication does not establish model training, a pretrained checkpoint, CREDO implementation, v10 learned refinement, or Cosmos v14 splitting. The [chaotic-data proposal](research-chaotic-data.md) received independent critic ACCEPT as an experimental proposal only. See [provenance](provenance.md), [method manifest](methods.md), and [source audit](source-audit.md).
