# Copyright (c) 2025 Bangyen Pham

import multiprocessing

import pytest

# Monkeypatch set_start_method to avoid RuntimeError when called multiple times
# This is specifically to fix mutmut on macOS with Python 3.12+
_orig_set_start_method = multiprocessing.set_start_method


def _patched_set_start_method(method, force=False):
    try:
        _orig_set_start_method(method, force=force)
    except RuntimeError:
        # If it's already set, we just ignore it
        pass


multiprocessing.set_start_method = _patched_set_start_method


# Mutmut names mutants without the package prefix (e.g.
# "data.x_get_dataset__mutmut_1") while the trampoline builds its qualname
# from __module__, which has it ("zsharp.data...."). Re-add the prefix so the
# selected mutant actually activates. "fail" is mutmut's forced-fail probe,
# not a mutant, so it passes through untouched.
import os  # noqa: E402

_mut = os.environ.get("MUTANT_UNDER_TEST", "")
if "__mutmut_" in _mut and not _mut.startswith("zsharp."):
    os.environ["MUTANT_UNDER_TEST"] = "zsharp." + _mut


@pytest.fixture(autouse=True)
def _isolate_results_dir(tmp_path, monkeypatch):
    """Redirect training result writes to a temp dir.

    train() calls _save_results unconditionally, so without this every test
    that trains would overwrite the real results/ directory with throwaway
    output.
    """
    monkeypatch.setattr(
        "zsharp.trainer.RESULTS_DIR", str(tmp_path / "results")
    )
