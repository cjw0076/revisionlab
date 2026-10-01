# RevisionLab state

Updated: 2026-10-01 (Asia/Seoul).

## Completed local v0.1.0

- Release priority: delayed-feedback memory/state contracts, controls, diagnostics, and small reproducible examples. archcredit remains separate; CREDO and Cosmos v14/v10 refinement are deferred.
- Implemented native ResidualMemory/RLS, NumPy reference, differentiable delta primitive, immutable tickets, owner-bound exactly-once replay, negative controls, and built-in checkpoint resume.
- Native checkpoints preserve pending/settled ticket identities, owner, clock, model state, and issued-forecast evaluation. Custom correctors work through generic replay but are not supported by built-in checkpoint helpers.
- CLI, typed packaging, Python 3.10/3.11/3.12 CI, contributor recipe, MIT license, release guide, and provenance/evidence documentation are complete.
- Recovered v13 historical artifacts are byte-preserved; source/result hashes and audit findings are recorded in [provenance](docs/provenance.md). This release did not rerun the full weather experiment.

## Verification

- Final local suite: **95 passed, 1 CUDA hardware skip, 196.79 seconds**.
- Ruff lint/format passed for 21 formatted files; strict mypy passed for 8 source files.
- Separate independent code review: APPROVE with zero remaining issues. Scientific documentation review: ACCEPT.
- Regression tests cover maturity, immutable keys, foreign/duplicate settlement, atomic failed writes, checkpoint/resume, rate/config consistency, unsupported checkpoint dtypes, norm overflow, and seeded permutation consistency.
- CPU `aot_eager`/gradcheck, examples, literal installed CLI, build/Twine metadata 2.4, and outside-checkout installed-wheel checks passed. See [verification](docs/verification.md).

## Distribution receipts and next work

- GitHub repository created: https://github.com/cjw0076/revisionlab. Hosted workflow receipts belong in [Actions](https://github.com/cjw0076/revisionlab/actions/workflows/ci.yml); publication assets belong at [v0.1.0](https://github.com/cjw0076/revisionlab/releases/tag/v0.1.0) once published. Local validation does not claim those hosted/publication steps have already succeeded.
- No PyPI upload or model training/pretrained release is recorded. Next distribution step is approved TestPyPI/PyPI handoff with trusted-publisher configuration.
- Next research step is controlled replay/generalization evidence across sites/regimes and matched correction budgets; v14 splitting requires separate preregistration and pending-ticket semantics.

Local residual alignment is not global BPTT alignment. The replay audit ledger grows with settled ticket IDs; fixed corrector capacity does not mean constant whole-runner memory. CPU smoke does not establish CUDA/Inductor compatibility or performance.
