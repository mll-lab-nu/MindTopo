"""Public shell wrappers: present, executable, syntactically valid, no conda."""

from __future__ import annotations

import os
import subprocess

import pytest

from conftest import ROOT

SHELL_SCRIPTS = [
    "run_reasoning_task.sh",
    "run_planning_task.sh",
    "run_interleaved_task.sh",
    "run_interleaved_full.sh",
    "setup_frontends.sh",
    "topobench-eval",
]


@pytest.mark.parametrize("script", SHELL_SCRIPTS)
def test_script_is_executable(script):
    path = ROOT / "bin" / script
    assert path.is_file(), script
    assert os.access(path, os.X_OK), f"{script} is not executable"


@pytest.mark.parametrize("script", SHELL_SCRIPTS)
def test_script_passes_bash_syntax_check(script):
    result = subprocess.run(
        ["bash", "-n", str(ROOT / "bin" / script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", ["run_reasoning_task.sh", "run_planning_task.sh",
                                    "run_interleaved_task.sh"])
def test_wrappers_prefer_the_uv_venv_not_conda(script):
    text = (ROOT / "bin" / script).read_text()
    assert ".venv/bin/python" in text, f"{script} should resolve the uv .venv"
    assert "conda activate" not in text, f"{script} must not force conda"
