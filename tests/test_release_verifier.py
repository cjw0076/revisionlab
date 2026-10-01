"""Reject changed bytes, metadata and package-index responses before publication."""

import copy
import hashlib
import io
import json
import tarfile
import urllib.error
import zipfile

import pytest

from scripts import verify_release as release


def index_payload():
    return {
        "info": {"name": "revisionlab", "version": "0.1.0"},
        "urls": [
            {
                "filename": name,
                "digests": {"sha256": digest},
                "yanked": False,
                "url": f"https://files.pythonhosted.org/packages/{name}",
            }
            for name, digest in release.EXPECTED.items()
        ],
    }


@pytest.fixture
def synthetic_assets(tmp_path, monkeypatch):
    metadata = b"Name: revisionlab\nVersion: 0.1.0\n"
    wheel = tmp_path / "revisionlab-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("revisionlab-0.1.0.dist-info/METADATA", metadata)
    sdist = tmp_path / "revisionlab-0.1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        member = tarfile.TarInfo("revisionlab-0.1.0/PKG-INFO")
        member.size = len(metadata)
        archive.addfile(member, io.BytesIO(metadata))
    monkeypatch.setattr(
        release,
        "EXPECTED",
        {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (wheel, sdist)},
    )
    return tmp_path


def test_reviewed_artifacts_accept_exact_bytes_and_metadata(synthetic_assets):
    release.verify_assets(synthetic_assets)


def test_changed_artifact_bytes_fail(synthetic_assets):
    wheel = next(synthetic_assets.glob("*.whl"))
    wheel.write_bytes(wheel.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        release.verify_assets(synthetic_assets)


@pytest.mark.parametrize("extra", ["injected.whl", "SHA256SUMS.txt"])
def test_unreviewed_files_fail(synthetic_assets, extra):
    (synthetic_assets / extra).write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="exactly"):
        release.verify_assets(synthetic_assets)


def test_missing_distribution_fails(synthetic_assets):
    next(synthetic_assets.glob("*.tar.gz")).unlink()
    with pytest.raises(ValueError, match="exactly"):
        release.verify_assets(synthetic_assets)


@pytest.mark.parametrize(
    "raw",
    [
        b"Name: other\nVersion: 0.1.0\n",
        b"Name: revisionlab\nVersion: 0.1.1\n",
        b"Name: revisionlab\nVersion: 0.1.0\nVersion: 0.1.1\n",
        b"Name: revisionlab\nVersion: 0.1.0\nName: other\n",
    ],
)
def test_metadata_mismatch_or_duplicate_identity_fails(raw):
    with pytest.raises(ValueError, match="name/version"):
        release.verify_metadata(raw)


def test_correct_index_payload_passes():
    assert len(release.verify_index_payload(index_payload())) == 2


def test_testpypi_uses_separate_download_host():
    payload = index_payload()
    for item in payload["urls"]:
        item["url"] = item["url"].replace("files.pythonhosted.org", "test-files.pythonhosted.org")
    assert len(release.verify_index_payload(payload, "testpypi")) == 2
    with pytest.raises(ValueError, match="origin"):
        release.verify_index_payload(payload, "pypi")


def test_testpypi_rejects_production_download_host():
    with pytest.raises(ValueError, match="origin"):
        release.verify_index_payload(index_payload(), "testpypi")


@pytest.mark.parametrize("field,value", [("name", "other"), ("version", "0.1.1")])
def test_index_identity_mismatch_fails(field, value):
    payload = index_payload()
    payload["info"][field] = value
    with pytest.raises(ValueError, match="mismatch"):
        release.verify_index_payload(payload)


def test_index_digest_mismatch_fails():
    payload = index_payload()
    payload["urls"][0]["digests"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        release.verify_index_payload(payload)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "extra"])
def test_index_file_set_mismatch_fails(mode):
    payload = index_payload()
    if mode == "missing":
        payload["urls"].pop()
    elif mode == "duplicate":
        payload["urls"][1] = copy.deepcopy(payload["urls"][0])
    else:
        payload["urls"].append(copy.deepcopy(payload["urls"][0]))
    with pytest.raises(ValueError, match="distribution set mismatch"):
        release.verify_index_payload(payload)


@pytest.mark.parametrize("yanked", [True, None])
def test_yanked_or_unknown_status_fails(yanked):
    payload = index_payload()
    payload["urls"][0]["yanked"] = yanked
    with pytest.raises(ValueError, match="yanked"):
        release.verify_index_payload(payload)


@pytest.mark.parametrize(
    "url", ["http://files.pythonhosted.org/a", "https://example.org/a", "file:///a"]
)
def test_untrusted_download_origin_fails(url):
    payload = index_payload()
    payload["urls"][0]["url"] = url
    with pytest.raises(ValueError, match="origin"):
        release.verify_index_payload(payload)


def test_no_wait_does_not_hide_absent_index(monkeypatch):
    def absent(*args, **kwargs):
        raise urllib.error.HTTPError("https://test.pypi.org", 404, "not found", {}, None)

    monkeypatch.setattr(release.urllib.request, "urlopen", absent)
    with pytest.raises(urllib.error.HTTPError):
        release.verify_index("testpypi")


def test_permission_error_is_not_retried(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("https://test.pypi.org", 403, "forbidden", {}, None)

    monkeypatch.setattr(release.urllib.request, "urlopen", forbidden)
    with pytest.raises(urllib.error.HTTPError):
        release.verify_index("testpypi", wait_seconds=300)
    assert len(calls) == 1


def test_downloaded_bytes_must_match_index_digest(monkeypatch):
    payload = index_payload()

    def response(url, **kwargs):
        if url.endswith("/json"):
            return io.BytesIO(json.dumps(payload).encode())
        return io.BytesIO(b"unreviewed downloaded bytes")

    monkeypatch.setattr(release.urllib.request, "urlopen", response)
    with pytest.raises(ValueError, match="Downloaded index artifact SHA256 mismatch"):
        release.verify_index("pypi")
