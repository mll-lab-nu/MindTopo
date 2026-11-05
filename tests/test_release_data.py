"""Release regressions: actual sample identity and runner routing, without APIs."""
import importlib.util
import json
from pathlib import Path
from collections import Counter

import pytest
from omegaconf import OmegaConf

from conftest import ROOT
from topobench_eval.datasets import (
    TASKS, configure_run, data_root, question_path, read_rows,
    selected_split, validate_dataset, validate_images,
    require_same_dataset,
)


def test_extended_is_the_original_dataset_and_flags_core_overlap():
    all_rows = {}
    for split, expected in (("core", 3900), ("extended", 11035)):
        all_rows[split] = [(task, row) for task in TASKS for row in validate_dataset(task, split, images=False)]
        assert len(all_rows[split]) == expected
        assert len({(task, row["id"]) for task, row in all_rows[split]}) == expected
    hashes = lambda split: {(task, row["meta_info"]["release_input_sha256"]) for task, row in all_rows[split]}
    manifest = read_rows(ROOT / "data/manifests/extended_manifest.jsonl")
    assert len(hashes("extended")) == 11030
    assert hashes("core") & hashes("extended") == {(row["task"], row["input_sha256"]) for row in manifest if row["in_core"]}
    assert sum(row["in_core"] for row in manifest) == 2250
    core = {(task, row["id"]): row for task, row in all_rows["core"]}
    reused = [row for task, row in all_rows["extended"] if (task, row["id"]) in core
              and core[(task, row["id"])]["meta_info"]["release_input_sha256"] != row["meta_info"]["release_input_sha256"]]
    assert len(reused) == 150
    assert all(row["id"].startswith("origami") for row in reused)


def test_core_has_150_instances_per_difficulty_without_rewriting_generator_difficulty():
    for task in TASKS:
        rows = read_rows(question_path(task, "core"))
        assert Counter(row["meta_info"]["benchmark_level"] for row in rows) == {"Easy": 150, "Hard": 150}
    rows = read_rows(question_path("order_origami", "core"))
    assert {row["meta_info"]["difficulty"] for row in rows} == {"medium", "expert"}


def test_default_and_explicit_split_routes_and_provenance(monkeypatch):
    monkeypatch.delenv("TOPOBENCH_DATA_SPLIT", raising=False)
    assert selected_split() == "core"
    monkeypatch.setenv("TOPOBENCH_DATA_SPLIT", "extended")
    assert data_root("order_origami").parts[-2:] == ("extended", "order_origami")
    cfg = OmegaConf.create({"data": {"split": "extended"}, "env": {"name": "continuity_pipe"}})
    info = configure_run(cfg)
    assert cfg.data.split == info["dataset_split"] == "extended"
    assert cfg.env.source_question_jsonl == "data/extended/continuity_pipe/question.jsonl"
    assert len(info["dataset_sha256"]) == 64
    with pytest.raises(ValueError):
        selected_split("supplementary")


def test_probing_keeps_105_ids_and_requires_explicit_manifest_change():
    cfg = OmegaConf.create({"data": {"split": None}, "env": {"name": "continuity_pipe", "episode_ids_path": "data/slices/balanced_105/continuity_pipe.json"}})
    info = configure_run(cfg, probing=True)
    assert info["dataset_split"] is None and info["dataset_protocol"] == "legacy_probing"
    ids = json.loads((ROOT / cfg.env.episode_ids_path).read_text())
    assert len(ids) == 105
    assert set(ids) <= {row["id"] for row in read_rows(ROOT / cfg.env.source_question_jsonl)}
    cfg.data.split = "core"
    with pytest.raises(ValueError, match="balanced_105"):
        configure_run(cfg, probing=True)


def test_missing_release_image_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="must not be silently filtered"):
        validate_images([{"id": "example", "images": ["missing.png"]}], tmp_path)


def test_raw_static_config_cannot_silently_mix_partitions(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "lmms_eval_tasks"))
    from _common import make_hooks
    core_row = read_rows(question_path("order_origami", "core"))[0]
    with pytest.raises(ValueError, match="different benchmark splits"):
        make_hooks(data_root("order_origami", "extended"))["process_docs"]([core_row])


def test_resume_cannot_mix_splits_or_changed_inputs():
    current = {"dataset_sha256": "abc", "dataset_split": "core", "dataset_protocol": "benchmark"}
    require_same_dataset(dict(current), current)
    for changed in ({**current, "dataset_split": "extended"}, {**current, "dataset_sha256": "other"}, {}):
        with pytest.raises(ValueError, match="Cannot resume"):
            require_same_dataset(changed, current)


def test_custom_source_override_remains_explicit_and_is_not_labelled_core(tmp_path):
    path = tmp_path / "custom.jsonl"
    path.write_text(json.dumps({"id": "custom-episode"}) + "\n")
    cfg = OmegaConf.create({"data": {"split": "core"}, "env": {
        "name": "continuity_pipe", "source_question_jsonl": str(path)}})
    info = configure_run(cfg)
    assert cfg.env.source_question_jsonl == str(path)
    assert info["dataset_split"] is None and info["dataset_protocol"] == "custom"
    assert info["dataset_release"] is None


def test_static_configs_and_paths_follow_the_same_split(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(ROOT / "lmms_eval_tasks"))
    from _release_tasks import include_path
    import yaml
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_multi_constructor("!function", lambda loader, suffix, node: str(suffix))
    for split in ("core", "extended"):
        monkeypatch.setenv("TOPOBENCH_DATA_SPLIT", split)
        path = include_path() / "order_origami"
        config = yaml.load((path / "task.yaml").read_text(), Loader=Loader)
        assert Path(config["dataset_kwargs"]["data_files"]["test"]) == question_path("order_origami", split)
        spec = importlib.util.spec_from_file_location("test_release_utils_" + split, path / "utils.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.ENV_DATA_ROOT == data_root("order_origami", split)


def test_explicit_core_rules_are_explained_without_changing_legacy_prompts():
    from types import SimpleNamespace
    from unified_types import PromptParts
    from release_rules import apply_release_rules
    legacy = PromptParts(model_blocks=[{"type": "text", "text": "Original prompt"}])
    assert apply_release_rules(legacy, SimpleNamespace(reset_kwargs={})) is legacy
    assert legacy.model_blocks == [{"type": "text", "text": "Original prompt"}]
    parts = PromptParts(model_blocks=[{"type": "text", "text": "A legal action is any non-empty cell in the current grid."}])
    spec = SimpleNamespace(reset_kwargs={"options": {"frontend_config": {"moveRule": "row_column_v1"}}})
    apply_release_rules(parts, spec)
    text = " ".join(block["text"] for block in parts.model_blocks)
    assert "must share a row or column" in text and "any non-empty cell" not in text


def test_planning_adapters_load_both_partitions_in_one_process():
    from hydra import compose, initialize_config_dir
    from env_adapters import build_adapter
    from topobench_eval.datasets import PLANNING_TASKS
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "planning_eval_tasks/configs")):
        for split in ("core", "extended"):
            for task in PLANNING_TASKS:
                cfg = compose(config_name="config", overrides=[f"env={task}", "model=oracle", f"data.split={split}"])
                configure_run(cfg)
                specs = build_adapter(task).enumerate_episodes(cfg)
                expected = read_rows(question_path(task, split))
                assert [spec.episode_id for spec in specs] == [row["id"] for row in expected]
                if split == "core":
                    assert Counter(spec.metadata["benchmark_level"] for spec in specs) == {"Easy": 150, "Hard": 150}


def test_explicit_probing_path_is_custom_for_the_main_runner():
    probe = "data/probing/continuity_pipe/question.jsonl"
    cfg = OmegaConf.create({"data": {"split": "core"}, "env": {
        "name": "continuity_pipe", "source_question_jsonl": probe}})
    info = configure_run(cfg)
    assert info["dataset_protocol"] == "custom" and cfg.env.source_question_jsonl == probe


def test_custom_reasoning_honours_a_separate_image_root(tmp_path):
    from PIL import Image
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (4, 4)).save(images / "a.png")
    path = tmp_path / "custom.jsonl"
    path.write_text(json.dumps({"id": "custom", "images": ["a.png"]}) + "\n")
    cfg = OmegaConf.create({"data": {"split": None}, "env": {
        "name": "continuity_2d_maze", "question_jsonl": str(path), "image_root": str(images)}})
    assert configure_run(cfg, probing=True)["dataset_protocol"] == "custom"
    assert cfg.env.image_root == str(images)


@pytest.mark.parametrize("config,state,expected", [
    ({"moveRule": "row_column_v1"}, {}, "moveRule"),
    ({"pipeRule": "split_channels_v2"}, {"connectedIndices": []}, "pipeRule"),
    ({"catPolicy": "expert"}, {"catPolicy": {"name": "easy"}}, "catPolicy"),
])
def test_stale_frontend_cannot_silently_drop_a_saved_rule(config, state, expected):
    from types import SimpleNamespace
    from release_rules import require_frontend_rules
    spec = SimpleNamespace(episode_id="example", reset_kwargs={"options": {"frontend_config": config}})
    with pytest.raises(RuntimeError, match=expected):
        require_frontend_rules(spec, {"state": state})
