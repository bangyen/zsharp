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

# Monkeypatch mutmut.record_trampoline_hit to strip 'src.' prefix
# mutmut 3.x has an assertion that module names should not start with 'src.'
try:
    import mutmut.__main__

    _orig_record_trampoline_hit = mutmut.__main__.record_trampoline_hit

    def _patched_record_trampoline_hit(name, *args, **kwargs):
        # Pass through extra args (e.g. mutmut 3.7's `caller`) unchanged so
        # this shim survives upstream signature changes.
        name = name.removeprefix("src.")
        return _orig_record_trampoline_hit(name, *args, **kwargs)

    mutmut.__main__.record_trampoline_hit = _patched_record_trampoline_hit
except (ImportError, AttributeError):
    pass


@pytest.fixture(autouse=True)
def _isolate_results_dir(tmp_path, monkeypatch):
    """Redirect training result writes to a temp dir.

    train() calls _save_results unconditionally, so without this every test
    that trains would overwrite the real results/ directory with throwaway
    output.
    """
    monkeypatch.setattr("src.trainer.RESULTS_DIR", str(tmp_path / "results"))
