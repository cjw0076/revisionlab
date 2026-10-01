# RevisionLab state

Updated: 2026-10-01 (Asia/Seoul).

## Completed local v0.1.0

- Release priority: delayed-feedback memory/state contracts, controls, diagnostics, and small reproducible examples. archcredit remains separate; CREDO and Cosmos v14/v10 refinement are deferred.
- Implemented native ResidualMemory/RLS, NumPy reference, differentiable delta primitive, immutable tickets, owner-bound exactly-once replay, negative controls, and built-in checkpoint resume.
- Native checkpoints preserve pending/settled ticket identities, owner, clock, model state, and issued-forecast evaluation. Custom correctors work through generic replay but are not supported by built-in checkpoint helpers.
- CLI, typed packaging, Python 3.10/3.11/3.12 CI, contributor recipe, MIT license, release guide, and provenance/evidence documentation are complete.
- Recovered v13 historical artifacts are byte-preserved; source/result hashes and audit findings are recorded in [provenance](docs/provenance.md). This release did not rerun the full weather experiment.

## Verification

- Tagged-release local suite: **95 passed, 1 CUDA hardware skip, 196.79 seconds**. Current publishing-pipeline local suite: **121 passed, 1 CUDA hardware skip, 41.59 seconds**, including 26 release-verifier tests.
- Ruff lint/format passed for 21 formatted files; strict mypy passed for 8 source files.
- Separate independent code review: APPROVE with zero remaining issues. Scientific documentation review: ACCEPT.
- Regression tests cover maturity, immutable keys, foreign/duplicate settlement, atomic failed writes, checkpoint/resume, rate/config consistency, unsupported checkpoint dtypes, norm overflow, and seeded permutation consistency.
- CPU `aot_eager`/gradcheck, examples, literal installed CLI, build/Twine metadata 2.4, and outside-checkout installed-wheel checks passed. See [verification](docs/verification.md).

## Distribution receipts and next work

- [v0.1.0 research prerelease](https://github.com/cjw0076/revisionlab/releases/tag/v0.1.0) published with wheel, source distribution, and `SHA256SUMS`. Tagged source: `58567cc162f2d290be10c8e26ca251bbf6d3ccbc`.
- [Hosted CI run 36822956187](https://github.com/cjw0076/revisionlab/actions/runs/36822956187): all three Python 3.10/3.11/3.12 jobs passed; each recorded 95 passed and 1 CUDA hardware skip, including lint/types/build/Twine/installed-wheel checks. NumPy typing compatibility fixes are included.
- Published artifact hashes and source receipts are recorded in [verification](docs/verification.md). This postrelease state records the tested/tagged snapshot; it does not claim a later documentation commit was part of that snapshot.
- Current publishing execution uses workflow-main revision `fd98444cd47d00975cbef79b6e5e9be597cfe86e`; both GitHub publishing environments restrict deployment to main. The packaged source remains tagged revision `58567cc162f2d290be10c8e26ca251bbf6d3ccbc`. Local release-verifier tests, Ruff/strict mypy (8 package files), Twine, original artifact hash checks, and fresh installed-wheel CLI/four built-in checkpoint checks passed, with separate monitor approval.
- [CI 36826566815](https://github.com/cjw0076/revisionlab/actions/runs/36826566815): all Python 3.10/3.11/3.12 jobs passed, each 121 tests with 1 CUDA hardware skip. [Manual validation 36826580022](https://github.com/cjw0076/revisionlab/actions/runs/36826580022) passed in 58 seconds, including fresh dependencies/wheel, four checkpoint resumes, and CLI.
- Historical [TestPyPI attempt 36826767156](https://github.com/cjw0076/revisionlab/actions/runs/36826767156) failed OIDC exchange with `invalid-publisher` before account-side registration; no upload occurred in that attempt.
- [TestPyPI run 36836398900](https://github.com/cjw0076/revisionlab/actions/runs/36836398900) passed validation, publication, and index verification. [TestPyPI v0.1.0](https://test.pypi.org/project/revisionlab/0.1.0/) serves both original artifact hashes.
- [Production PyPI run 36836731138](https://github.com/cjw0076/revisionlab/actions/runs/36836731138) passed validation, publication, and index verification. [PyPI v0.1.0](https://pypi.org/project/revisionlab/0.1.0/) serves both original artifact hashes, independently downloaded and verified on both indexes.
- Independent monitor verified both files are non-yanked and cryptographically verified all four TestPyPI/PyPI publication attestations with `pypi-attestations==0.0.30`; publisher repository/workflow/environment and signed filename/digest matched. This verifies publication identity and bytes, not reproducible builds or historical research claims. See [public receipt](artifacts/publication-v0.1.0.json).
- The workflow is pinned to v0.1.0; future versions require reviewed new source/tag and hashes. Do not silently rerun an existing upload or replace published files.
- Next research step is controlled replay/generalization evidence across sites/regimes and matched correction budgets; v14 splitting requires separate preregistration and pending-ticket semantics.
- The [chaotic-data hypothesis](docs/research-chaotic-data.md) received independent critic ACCEPT as a proposal only; it has not been implemented or validated by new training.

Local residual alignment is not global BPTT alignment. The replay audit ledger grows with settled ticket IDs; fixed corrector capacity does not mean constant whole-runner memory. CPU smoke does not establish CUDA/Inductor compatibility or performance.

## Optional Transformers integration (unreleased)

- Branch `codex/hf-delayed-regression` starts from main `7f1de7c0b57dab194a7bf420d4f30312e7ccbdac`. The new `revisionlab.integrations.transformers.issue_regression` selects one `[batch, 1]` regression logit and delegates to existing `DelayedReplay.issue` with an explicit caller-defined memory key. Targets still enter only through matured `release`; no model training, automatic address learning, or Trainer callback was added.
- Transformers is isolated in a repository-only optional extra and module. Core runtime dependencies and imports remain independent. The runnable tiny random BERT example uses no Hub, tokenizer, pretrained weights, or data download. The API guide is [docs/transformers.md](docs/transformers.md).
- Fresh isolated core environment, confirmed without Transformers: **123 passed, 2 skipped, 30.26 seconds** (CUDA hardware and optional integration module); strict mypy passed for 10 package source files. Preliminary existing-environment Transformers 4.57.1 checks: **24 passed, 84.52 seconds**, offline example and strict package/example types passed.
- Selected Transformers 5.18.0 in isolated Python 3.11 with CPU PyTorch 2.14.1: **24 passed, 26.34 seconds**; strict package/example types (11 files) and the offline example passed.
- Selected Transformers 4.57.6 in the isolated environment: **24 passed, 30.11 seconds**; strict package/example types (11 files), the offline example, and dependency consistency (`pip check`) passed.
- New branch build/Twine and wheel metadata/module checks passed in isolated output storage. The candidate wheel was installed without editable mode; outside-checkout imports resolved to site-packages rather than source, and the offline example passed. Original published wheel/source hashes and v0.1.0 tag remain unchanged, independently confirmed. Published PyPI 0.1.0 contains no integration module or extra; this branch has not been published.
- [Draft PR #1](https://github.com/cjw0076/revisionlab/pull/1) and [PR CI 36855556931](https://github.com/cjw0076/revisionlab/actions/runs/36855556931) test code commit `859210483920c28bf5d548569bdbd31cfdb4b4ce`: all five jobs passed. Python 3.10/3.11/3.12 each recorded 123 passed and 2 skips (3.12 also recorded 2 warnings); Transformers 4.57.6 recorded 24 passed in 6.00 seconds and 5.18.0 recorded 24 passed in 4.74 seconds. [Push CI 36855537282](https://github.com/cjw0076/revisionlab/actions/runs/36855537282) also passed. Later documentation-only receipt commits are not included in these tested-code claims.
- Independent review: APPROVE, zero blockers. See [verification](docs/verification.md) for separate release, local branch, and hosted integration receipts.
