"""Interactive planning chain: configs, adapters, no-key policies."""

from __future__ import annotations

import yaml

from conftest import ROOT, PLANNING_TASKS


def test_planning_env_configs_exist_and_parse():
    cfg_dir = ROOT / "planning_eval_tasks" / "configs" / "env"
    for task in PLANNING_TASKS:
        path = cfg_dir / f"{task}.yaml"
        assert path.is_file(), f"missing env config for {task}"
        cfg = yaml.safe_load(path.read_text())
        assert cfg.get("name") == task, f"{task}: name mismatch ({cfg.get('name')})"


def test_local_no_key_model_configs():
    model_dir = ROOT / "planning_eval_tasks" / "configs" / "model"
    for name in ("oracle", "random"):
        cfg = yaml.safe_load((model_dir / f"{name}.yaml").read_text())
        assert cfg["kind"] == "local_policy", name
        # No-key baselines must not carry credentials.
        assert not cfg.get("api_key"), name
        assert not cfg.get("base_url"), name


def test_build_adapter_resolves_every_planning_env():
    import env_adapters

    for task in PLANNING_TASKS:
        adapter = env_adapters.build_adapter(task)
        assert adapter is not None, task


def test_local_policy_client_needs_no_key():
    from omegaconf import OmegaConf
    import model_adapter

    cfg = OmegaConf.create(
        {"id": "oracle", "kind": "local_policy", "provider": "local",
         "api_key": "", "base_url": ""}
    )
    # local_policy dispatch returns None (no remote client, no key pool).
    client = model_adapter.build_model_client(model_cfg=cfg, system_prompt="")
    assert client is None
