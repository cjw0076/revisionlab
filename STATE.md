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
- Trusted publishing pipeline is configured on main revision `32f98abd9f9b806c3c4fc5e1384ef87fcdf290b8`; both GitHub publishing environments restrict deployment to main. Local release-verifier tests, Ruff/strict mypy (8 package files), Twine, original artifact hash checks, and fresh installed-wheel CLI/four built-in checkpoint checks passed, with separate monitor approval.
- [CI 36826566815](https://github.com/cjw0076/revisionlab/actions/runs/36826566815): all Python 3.10/3.11/3.12 jobs passed, each 121 tests with 1 CUDA hardware skip. [Manual validation 36826580022](https://github.com/cjw0076/revisionlab/actions/runs/36826580022) passed in 58 seconds, including fresh dependencies/wheel, four checkpoint resumes, and CLI.
- [TestPyPI publication attempt 36826767156](https://github.com/cjw0076/revisionlab/actions/runs/36826767156): validation passed; upload failed during OIDC exchange with `invalid-publisher` (valid token, no matching publisher). No upload occurred; index verification and PyPI publication were skipped. Matching registration requires repository `cjw0076/revisionlab`, workflow `publish.yml`, ref `main`, and environment `testpypi`.
- TestPyPI/PyPI project APIs remain 404. Account-side publisher registration is required before upload; no successful TestPyPI/PyPI publication or software/runtime defect is claimed.
- Next research step is controlled replay/generalization evidence across sites/regimes and matched correction budgets; v14 splitting requires separate preregistration and pending-ticket semantics.
- The [chaotic-data hypothesis](docs/research-chaotic-data.md) received independent critic ACCEPT as a proposal only; it has not been implemented or validated by new training.

Local residual alignment is not global BPTT alignment. The replay audit ledger grows with settled ticket IDs; fixed corrector capacity does not mean constant whole-runner memory. CPU smoke does not establish CUDA/Inductor compatibility or performance.
