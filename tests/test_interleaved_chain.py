"""Interleaved chain: only intentionally wired tasks, configs, module compile."""

from __future__ import annotations

import py_compile

import yaml

from conftest import ROOT, INTERLEAVED_TASKS

INTERLEAVED_DIR = ROOT / "interleaved_eval_tasks"


def test_interleaved_env_configs_are_exactly_the_wired_tasks():
    env_dir = INTERLEAVED_DIR / "configs" / "env"
    present = {p.stem for p in env_dir.glob("*.yaml")}
    assert present == set(INTERLEAVED_TASKS), present


def test_interleaved_oracle_model_config_is_keyless():
    cfg = yaml.safe_load((INTERLEAVED_DIR / "configs" / "model" / "oracle.yaml").read_text())
    assert cfg["kind"] == "local_policy"
    assert not cfg.get("api_key")
    assert not cfg.get("base_url")


def test_interleaved_modules_compile():
    for name in (
        "agent_runner.py",
        "interleaved_client.py",
        "image_gen_client.py",
        "reasoning_loader.py",
        "reasoning_runner.py",
    ):
        py_compile.compile(str(INTERLEAVED_DIR / name), doraise=True)


def test_no_retired_smoke12_residue():
    # Retired one-shot dataset/smoke tooling must not reappear.
    for pattern in ("run_interleaved_smoke12.sh", "build_interleaved_dataset.py"):
        assert not (ROOT / "bin" / pattern).exists(), pattern
