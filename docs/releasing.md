# Release handoff

## v0.1 checks

Before distribution, require green lint/format, strict package types, tests including adversarial regressions, compile/gradcheck/serialization smoke, and independent review. Build wheel and source distribution, run Twine metadata checks, and install the wheel outside the source checkout. Confirm the imported package comes from the installed wheel, then run `revisionlab-check`.

Keep result scope explicit: historical receipts are not a newly reproduced full weather experiment. No trained model, global credit-assignment result, v14 split implementation, or general cross-architecture support is implied by this release.

## TestPyPI and PyPI

The manual `publish.yml` workflow publishes the exact reviewed GitHub v0.1.0 assets.
It never rebuilds distributions or changes the version, and accepts dispatches only
from `cjw0076/revisionlab` on `main`. Pushes and releases do not trigger publishing.

Register the following GitHub pending publisher separately in
[TestPyPI](https://test.pypi.org/manage/account/publishing/) and
[PyPI](https://pypi.org/manage/account/publishing/). The two services use independent
accounts and publisher registrations. For an existing project, use its Publishing
settings instead of the pending-project form.

| Field | TestPyPI | PyPI |
| --- | --- | --- |
| PyPI project name | `revisionlab` | `revisionlab` |
| Owner | `cjw0076` | `cjw0076` |
| Repository | `revisionlab` | `revisionlab` |
| Workflow filename | `publish.yml` | `publish.yml` |
| GitHub environment | `testpypi` | `pypi` |

The filename excludes `.github/workflows/`. Match the environment exactly.
Successful initial upload converts the pending publisher into the project's normal
publisher. See the official [pending-project instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [Trusted Publisher usage](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
No long-lived API token is required. Only the isolated upload jobs receive
`id-token: write`; download, validation and installed-package execution jobs do not.
Official actions are pinned to verified commit SHAs.

Run in this order:

```bash
gh workflow run publish.yml --ref main -f target=validate
gh workflow run publish.yml --ref main -f target=testpypi
gh workflow run publish.yml --ref main -f target=pypi
```

Wait for each run to succeed before proceeding. `validate` checks the tag commit,
archive identities and SHA256, strict Twine metadata, verifier regressions, then
fresh wheel installation outside the checkout, CLI and checkpoint resume for all
four native memories. `testpypi` uploads those assets and checks the resulting index
JSON, downloaded bytes and a fresh hash-constrained installation. `pypi` repeats the
TestPyPI index and installation checks as a prerequisite before publishing the same
assets, then independently verifies PyPI. Index visibility retries are bounded to
five minutes; hash, identity, yank status and origin mismatches fail immediately.

Dependencies come from the official CPU PyTorch index and PyPI. Only `revisionlab`
is installed from TestPyPI with `--no-deps --require-hashes`; no mixed extra index is
used. If a version was already uploaded, the workflow fails rather than silently
skipping files. Inspect the existing index hashes and publication receipt before
choosing recovery; published version files cannot be replaced.

Reviewed source tag: `v0.1.0`, commit
`58567cc162f2d290be10c8e26ca251bbf6d3ccbc`.

| Distribution | SHA256 |
| --- | --- |
| `revisionlab-0.1.0-py3-none-any.whl` | `fd9500032333fd2cdb1a990e39f9c1d4c5e8b7556036eb79f167277dc73704e8` |
| `revisionlab-0.1.0.tar.gz` | `e8529a4c07ae9cb3672fe96168bb7ab71bdd205ddb428289285b1e54690d121f` |

Record successful workflow links and public index URLs in the release receipt.
A configured workflow or publisher alone is not proof of a successful upload.

## Further research

Prioritize reproduced delayed-feedback comparisons, matched RLS/LMS resource budgets, multiple sites/regimes, and recovery/failure cases. State splitting requires a new preregistration: prediction-time ownership, pending-ticket migration, split witnesses, and matched no-split/mixture controls. Keep CREDO and v14 work separate from the v0.1 release contract.
