"""Shared answer parsing / scoring / provider registry."""

from __future__ import annotations

import json

from conftest import ROOT, STATIC_TASKS


def test_package_public_symbols():
    import topobench_eval

    for name in (
        "AnswerContract",
        "AnswerMode",
        "ParseResult",
        "answers_equal",
        "parse_answer",
        "parse_answer_result",
        "resolve_answer_contract",
    ):
        assert hasattr(topobench_eval, name), name


def test_provider_registry_loads_documented_models():
    from topobench_eval.provider_registry import (
        load_model_registry,
        resolve_model_registry_entry,
    )

    registry = load_model_registry()
    assert registry, "provider registry is empty"
    # A representative subset of the models advertised in the top-level README.
    for model_id in ("internvl", "qwen", "nemotron", "gpt_5_5"):
        entry = resolve_model_registry_entry(model_id)
        assert entry is not None, model_id


def test_answer_contract_roundtrip_on_real_rows():
    from topobench_eval.answer_scoring import answers_equal
    from topobench_eval.contract_resolver import resolve_answer_contract

    for task in STATIC_TASKS:
        path = ROOT / "environments" / task / "output" / "question.jsonl"
        row = json.loads(path.open().readline())
        contract = resolve_answer_contract(row)
        # Ground truth must always score as equal to itself under its contract.
        assert answers_equal(row["gt_answer"], row["gt_answer"], contract), task
