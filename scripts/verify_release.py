"""Fail-closed checks for the reviewed v0.1.0 distributions; never rebuild them."""

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import venv
import zipfile
from email.parser import BytesParser
from pathlib import Path

VERSION = "0.1.0"
SOURCE_COMMIT = "58567cc162f2d290be10c8e26ca251bbf6d3ccbc"
EXPECTED = {
    "revisionlab-0.1.0-py3-none-any.whl": (
        "fd9500032333fd2cdb1a990e39f9c1d4c5e8b7556036eb79f167277dc73704e8"
    ),
    "revisionlab-0.1.0.tar.gz": "e8529a4c07ae9cb3672fe96168bb7ab71bdd205ddb428289285b1e54690d121f",
}
INDEXES = {"testpypi": "https://test.pypi.org", "pypi": "https://pypi.org"}
DOWNLOAD_HOSTS = {"testpypi": "test-files.pythonhosted.org", "pypi": "files.pythonhosted.org"}


def verify_metadata(raw):
    metadata = BytesParser().parsebytes(raw)
    if metadata.get_all("Name") != ["revisionlab"] or metadata.get_all("Version") != [VERSION]:
        raise ValueError("Distribution name/version differs from the reviewed release")


def verify_assets(directory):
    directory = Path(directory)
    if {path.name for path in directory.iterdir()} != set(EXPECTED):
        raise ValueError("Expected exactly the reviewed wheel and source distribution")
    for filename, expected_hash in EXPECTED.items():
        path = directory / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError("Distribution must be a regular file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Artifact SHA256 mismatch: {filename}")
        if filename.endswith(".whl"):
            with zipfile.ZipFile(path) as archive:
                verify_metadata(archive.read(f"revisionlab-{VERSION}.dist-info/METADATA"))
        else:
            with tarfile.open(path, "r:gz") as archive:
                member = archive.extractfile(f"revisionlab-{VERSION}/PKG-INFO")
                if member is None:
                    raise ValueError("Missing source distribution metadata")
                verify_metadata(member.read())
    print(json.dumps({"version": VERSION, "source": SOURCE_COMMIT, "sha256": EXPECTED}))


def verify_index_payload(payload, index="pypi"):
    if payload.get("info", {}).get("name") != "revisionlab":
        raise ValueError("Index project name mismatch")
    if payload.get("info", {}).get("version") != VERSION:
        raise ValueError("Index version mismatch")
    files = payload.get("urls", [])
    if len(files) != len(EXPECTED) or {item.get("filename") for item in files} != set(EXPECTED):
        raise ValueError("Index distribution set mismatch")
    for item in files:
        if item.get("digests", {}).get("sha256") != EXPECTED[item["filename"]]:
            raise ValueError("Index artifact SHA256 mismatch")
        if item.get("yanked") is not False:
            raise ValueError("Index distribution is yanked or has unknown yank status")
        url = urllib.parse.urlparse(item.get("url", ""))
        if url.scheme != "https" or url.hostname != DOWNLOAD_HOSTS[index]:
            raise ValueError("Unexpected distribution download origin")
    return files


def verify_index(index, wait_seconds=0):
    endpoint = f"{INDEXES[index]}/pypi/revisionlab/{VERSION}/json"
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            with urllib.request.urlopen(endpoint, timeout=30) as response:
                payload = json.load(response)
            files = verify_index_payload(payload, index)
            # Check uploaded bytes as well as the index's reported digests.
            for item in files:
                with urllib.request.urlopen(item["url"], timeout=30) as response:
                    digest = hashlib.sha256(response.read()).hexdigest()
                if digest != EXPECTED[item["filename"]]:
                    raise ValueError("Downloaded index artifact SHA256 mismatch")
            print(json.dumps({"verified_index": index, "version": VERSION, "sha256": EXPECTED}))
            return
        except urllib.error.HTTPError as error:
            if error.code not in (404, 502, 503, 504) or time.monotonic() >= deadline:
                raise
        except urllib.error.URLError:
            if time.monotonic() >= deadline:
                raise
        time.sleep(min(10, max(0, deadline - time.monotonic())))


def installed_smoke():
    import importlib.metadata

    import numpy as np
    import torch

    import revisionlab
    from revisionlab.baselines import NoWrite, RLS_Corrector, ShuffledWrite
    from revisionlab.replay import DelayedReplay
    from revisionlab.serialization import load_checkpoint, save_checkpoint

    package_path = Path(revisionlab.__file__).resolve()
    if "site-packages" not in package_path.parts or "src" in package_path.parts:
        raise ValueError(
            f"Smoke imported source checkout instead of installed wheel: {package_path}"
        )
    if importlib.metadata.version("revisionlab") != VERSION or revisionlab.__version__ != VERSION:
        raise ValueError("Installed version mismatch")
    key = np.array([1.0, 0.0, 0.0, 0.0])
    memories = [
        revisionlab.ResidualMemory(4, rate=0.05),
        RLS_Corrector(4),
        NoWrite(4),
        ShuffledWrite(4, rate=0.05, seed=2),
    ]
    with tempfile.TemporaryDirectory() as temporary:
        for number, memory in enumerate(memories):
            runner = DelayedReplay(memory, horizon=2)
            ticket = runner.issue(0, 10.0, key)
            path = Path(temporary) / f"checkpoint-{number}.pt"
            save_checkpoint(path, runner)
            restored = load_checkpoint(path)
            assert restored.release(ticket.ticket_id, 12.0, 2) == runner.release(
                ticket.ticket_id, 12.0, 2
            )
            for name, tensor in runner.memory.state_dict().items():
                torch.testing.assert_close(tensor, restored.memory.state_dict()[name])
            try:
                restored.release(ticket.ticket_id, 12.0, 2)
            except ValueError:
                pass
            else:
                raise ValueError("Checkpoint lost exactly-once evidence ownership")
    print(f"Installed wheel, four native checkpoint restores, deduplication passed: {package_path}")


def fresh_install(index=None, directory=None):
    if index is None:
        verify_assets(directory)
    script = str(Path(__file__).resolve())
    with tempfile.TemporaryDirectory(prefix="revisionlab-release-") as temporary:
        venv.EnvBuilder(with_pip=True).create(temporary)
        binary = Path(temporary) / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )

        def run(*args):
            subprocess.run([str(binary), *args], cwd=temporary, check=True)

        run("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
        run("-m", "pip", "install", "numpy", "--index-url", "https://pypi.org/simple")
        wheel = next(name for name in EXPECTED if name.endswith(".whl"))
        if index is None:
            run("-m", "pip", "install", "--no-deps", str(Path(directory).resolve() / wheel))
        else:
            verify_index(index, wait_seconds=300)
            requirements = Path(temporary) / "reviewed-release.txt"
            requirements.write_text(
                f"revisionlab=={VERSION} --hash=sha256:{EXPECTED[wheel]}\n", encoding="utf-8"
            )
            run(
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-cache-dir",
                "--only-binary=:all:",
                "--require-hashes",
                "--index-url",
                f"{INDEXES[index]}/simple",
                "-r",
                str(requirements),
            )
        run(script, "smoke")
        cli = binary.parent / (
            "revisionlab-check.exe" if sys.platform == "win32" else "revisionlab-check"
        )
        subprocess.run([str(cli), "--steps", "32", "--json"], cwd=temporary, check=True)
        run("-m", "pip", "check")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assets = commands.add_parser("assets")
    assets.add_argument("directory", type=Path)
    index = commands.add_parser("index")
    index.add_argument("index", choices=INDEXES)
    index.add_argument("--wait-seconds", type=int, default=0)
    install = commands.add_parser("install")
    origin = install.add_mutually_exclusive_group(required=True)
    origin.add_argument("--index", choices=INDEXES)
    origin.add_argument("--directory", type=Path)
    commands.add_parser("smoke")
    args = parser.parse_args()
    if args.command == "assets":
        verify_assets(args.directory)
    elif args.command == "index":
        if not 0 <= args.wait_seconds <= 300:
            parser.error("Propagation wait must be between 0 and 300 seconds")
        verify_index(args.index, args.wait_seconds)
    elif args.command == "install":
        fresh_install(args.index, args.directory)
    else:
        installed_smoke()


if __name__ == "__main__":
    main()
