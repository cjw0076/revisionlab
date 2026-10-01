# RevisionLab v0.1.0

First research release of **PyTorch-native delayed-feedback memory contracts, controls, and diagnostics**.

## Included

- Immutable prediction-time tickets, finite maturity validation, owner-bound exactly-once replay settlement, and scoring of each actual issued forecast.
- Fixed-capacity native normalized-delta residual memory and RLS correction, with explicit no-write and seeded shuffled-write controls.
- NumPy reference backend, differentiable functional delta primitive, and single/pooled local residual-alignment diagnostics.
- Native-built-in checkpoints preserving memory, pending tickets, settled IDs, clock, and evaluation state across delayed-feedback resume.
- `revisionlab-check` synthetic CLI, small contributor example, MIT licensing, typed package, wheel/source distribution, and Python 3.10/3.11/3.12 CI.
- Byte-preserved historical v13 JSON receipts with a ten-file source hash manifest and clear scientific scope.

## Verification

Final local suite: **95 passed, 1 CUDA hardware skip** in 196.79 seconds on Windows 11, Python 3.11.9, PyTorch 2.14.0+cpu, and NumPy 1.26.4. Ruff lint/format passed; strict mypy passed for 8 source files. CPU `aot_eager`, autograd gradcheck, adversarial contracts, and serialization/resume checks passed. Wheel/source builds, Twine metadata checks, and installed-wheel checks outside the checkout passed. Separate code review returned APPROVE with no remaining issues; scientific documentation review returned ACCEPT.

See [the verification report](https://github.com/cjw0076/revisionlab/blob/v0.1.0/docs/verification.md) and [hosted workflow receipts](https://github.com/cjw0076/revisionlab/actions/workflows/ci.yml) for scope and runtime evidence.

## Limits

This is infrastructure and a synthetic smoke, not a new full weather replay or model-training release. Historical native-live MSE improved on the recorded frozen baseline while RLS had a lower mean MSE. Local alignment does not establish full-model BPTT agreement or global delayed-credit correctness.

Low-level writes do not deduplicate; the replay ledger provides exactly-once release and retains a growing settled-ID audit. Checkpoint helpers support only the four native built-ins. CPU compile smoke does not establish CUDA/Inductor compatibility or performance.

RevisionLab is separate from archcredit. CREDO, v10 learned refinement, and Cosmos v14 state splitting are deferred. No PyPI upload or pretrained model publication is included. TestPyPI/PyPI require a separately configured and approved trusted-publisher handoff.
