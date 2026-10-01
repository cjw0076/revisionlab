"""Core import isolation, even when Transformers is installed in the test environment."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("missing", ["transformers", "transitive_dependency"])
def test_optional_import_isolation_and_missing_dependency_errors(missing):
    script = """
import importlib.abc
import sys

class BlockTransformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "transformers" or fullname.startswith("transformers."):
            raise ModuleNotFoundError("Blocked dependency", name=sys.argv[1])

sys.meta_path.insert(0, BlockTransformers())
import revisionlab
import revisionlab.integrations
from revisionlab.replay import DelayedReplay
assert not any(name.startswith("transformers") for name in sys.modules)
try:
    import revisionlab.integrations.transformers
except ModuleNotFoundError as error:
    assert error.name == sys.argv[1]
    if error.name == "transformers":
        assert ".[transformers]" in str(error)
    else:
        assert str(error) == "Blocked dependency"
else:
    raise AssertionError("Optional adapter import should fail")
"""
    subprocess.run(
        [sys.executable, "-c", script, missing], check=True, capture_output=True, text=True
    )
