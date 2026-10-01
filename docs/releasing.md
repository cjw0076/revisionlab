# Release handoff

## v0.1 checks

Before distribution, require green lint/format, strict package types, tests including adversarial regressions, compile/gradcheck/serialization smoke, and independent review. Build wheel and source distribution, run Twine metadata checks, and install the wheel outside the source checkout. Confirm the imported package comes from the installed wheel, then run `revisionlab-check`.

Keep result scope explicit: historical receipts are not a newly reproduced full weather experiment. No trained model, global credit-assignment result, v14 split implementation, or general cross-architecture support is implied by this release.

## TestPyPI and PyPI

1. Recheck package-name availability and verify version, license, repository URL, and included files.
2. Configure a trusted publisher for the actual repository/release environment following [the Python Packaging guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).
3. Obtain human approval, publish the reviewed artifacts to TestPyPI, and verify installation in a fresh environment.
4. Obtain approval for the PyPI release, publish those reviewed artifacts, and record hashes/provenance plus the source tag.

No package-upload workflow is configured by default. A built wheel or public GitHub repository is not a PyPI publication.

## Further research

Prioritize reproduced delayed-feedback comparisons, matched RLS/LMS resource budgets, multiple sites/regimes, and recovery/failure cases. State splitting requires a new preregistration: prediction-time ownership, pending-ticket migration, split witnesses, and matched no-split/mixture controls. Keep CREDO and v14 work separate from the v0.1 release contract.
