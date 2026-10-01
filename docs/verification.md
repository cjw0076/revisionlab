# v0.1.0 verification

Recorded on 2026-10-01. The release was checked locally with Windows 11, Python 3.11.9, PyTorch 2.14.0+cpu, and NumPy 1.26.4.

| Check | Result |
| --- | --- |
| Final local tests | 95 passed, 1 CUDA hardware skip, 196.79 seconds. |
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

## Evidence limits

`revisionlab-check` is a synthetic stationary address/residual smoke. It confirms executable delayed correction and controls, not real-world forecasting quality. The v13 metrics and independent replay JSON are byte-preserved historical receipts; this release did not run a new full weather replay.

Local residual alignment is not full-model BPTT alignment. Fixed-capacity memory does not imply constant total runner storage: settled ticket IDs grow with the lifetime audit ledger. Exactly-once settlement is a `DelayedReplay` contract; low-level corrector writes apply every valid call. Checkpoint helpers support the four native built-ins, not arbitrary custom correctors.

No PyPI publication, model training, pretrained checkpoint, CREDO implementation, v10 learned refinement, or Cosmos v14 split implementation is claimed. See [provenance](provenance.md), [method manifest](methods.md), and [source audit](source-audit.md).
