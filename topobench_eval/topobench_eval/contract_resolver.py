"""Resolve :class:`AnswerContract` objects from TopoBench JSONL rows.

Dataset adapters should use this single resolver instead of re-implementing
prompt-string heuristics. Explicit metadata wins; structured task metadata is
second; the prompt is only a compatibility fallback for legacy datasets.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from topobench_eval.answer_contract import (
    AnswerContract,
    AnswerMode,
    choice_contract,
    contract_from_spec,
)


BEAD_COLORS = ("RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "PURPLE", "WHITE", "BROWN")
DOOR_COLORS = ("red", "blue", "green", "yellow", "purple", "cyan", "orange", "pink")
T04_RING_COLORS = ("red", "blue", "green", "brown", "white", "purple", "none")

_NAME_LIST_QTYPES = frozenset({"reachability_set", "bar_removal", "connected_point_list", "door_open"})
_SCHEMA_WORDS = frozenset({"answer", "ans", "value", "json_answer_value", "choice", "letter", "name"})


def question_type(doc: Mapping[str, Any]) -> str:
    meta = doc.get("meta_info") or {}
    category = doc.get("category") or []
    if isinstance(meta, Mapping) and meta.get("question_type"):
        return str(meta["question_type"]).strip()
    if doc.get("question_type"):
        return str(doc["question_type"]).strip()
    if isinstance(category, list) and len(category) >= 3:
        return str(category[2]).strip()
    return str(doc.get("type") or "").strip()


def name_list_keys(qtype: str, meta: Mapping[str, Any]) -> list[str]:
    if qtype == "reachability_set":
        target = str(meta.get("target") or "").casefold()
        return [
            name
            for name in _dict_values(meta.get("points"), "name")
            if name.casefold() != target
        ]
    if qtype == "bar_removal":
        return _dict_values(meta.get("bars"), "color")
    payload = meta.get("question_payload") or {}
    if not isinstance(payload, Mapping):
        return []
    if qtype == "connected_point_list":
        return [str(value).strip() for value in payload.get("candidate_points", []) if str(value).strip()]
    if qtype == "door_open":
        return _dict_values(payload.get("door_colors"), "color_name")
    return []


def resolve_answer_contract(
    doc: Mapping[str, Any],
    *,
    qtype_override: str | None = None,
) -> AnswerContract:
    meta = doc.get("meta_info") or {}
    if not isinstance(meta, Mapping):
        meta = {}
    qtype = str(qtype_override or question_type(doc)).strip()

    explicit_spec = str(meta.get("legal_values_spec") or doc.get("legal_values_spec") or "").strip()
    if explicit_spec:
        contract = contract_from_spec(explicit_spec, _ground_truth(doc))
        return _with_task_constraints(contract, qtype, str(doc.get("question") or ""))

    answer_type = str(
        doc.get("answer_type") or meta.get("answer_type") or doc.get("_answer_type") or ""
    ).strip().casefold()

    if answer_type in {"name_list", "list", "color_list"} or qtype in _NAME_LIST_QTYPES:
        keys = name_list_keys(qtype, meta)
        if not keys and answer_type == "color_list":
            keys = list(DOOR_COLORS)
        return AnswerContract(
            AnswerMode.NAME_LIST,
            legal_values=tuple(keys),
            allow_empty=qtype in {"reachability_set", "bar_removal", "connected_point_list"},
        )

    answer_format = str(doc.get("question") or "").split("[Answer Format]")[-1]
    fmt = answer_format.casefold()

    if _is_id_list_format(fmt):
        return AnswerContract(AnswerMode.ID_LIST, none_token="none")
    if "bead-color sequence" in fmt:
        return AnswerContract(AnswerMode.COLOR_SEQUENCE, legal_values=BEAD_COLORS)
    if _is_integer_format(fmt):
        return AnswerContract(AnswerMode.INTEGER, non_negative="non-negative" in fmt)

    if _is_t04_color_list_format(fmt):
        return AnswerContract(
            AnswerMode.NAME_LIST,
            legal_values=T04_RING_COLORS,
            allow_empty=False,
            none_token="none",
        )

    if "json array of uppercase letters" in fmt:
        labels = [str(value).strip() for value in meta.get("all_labels", []) if str(value).strip()]
        if not labels:
            labels = _bracketed_uppercase_choices(answer_format)
        return AnswerContract(AnswerMode.NAME_LIST, legal_values=tuple(labels), allow_empty=False)

    # Choices are intentionally below structured-list signatures: examples
    # inside a list schema are elements, not mutually exclusive answers.
    choices = _explicit_choices(doc, answer_format, meta)
    if choices:
        return choice_contract(choices)

    return contract_from_spec(None, _ground_truth(doc))


def _with_task_constraints(
    contract: AnswerContract,
    qtype: str,
    question: str,
) -> AnswerContract:
    if contract.mode is AnswerMode.INTEGER:
        return AnswerContract(
            AnswerMode.INTEGER,
            non_negative=contract.non_negative or "non-negative" in question.casefold(),
        )
    if contract.mode is not AnswerMode.NAME_LIST:
        return contract
    return AnswerContract(
        AnswerMode.NAME_LIST,
        legal_values=contract.legal_values,
        allow_empty=qtype in {"reachability_set", "bar_removal", "connected_point_list"},
        none_token=contract.none_token,
    )


def _ground_truth(doc: Mapping[str, Any]) -> Any:
    value = doc.get("gt_answer")
    return doc.get("answer") if value is None else value


def _dict_values(values: Any, key: str) -> list[str]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, Mapping)):
        return []
    return [
        str(item[key]).strip()
        for item in values
        if isinstance(item, Mapping) and item.get(key) is not None and str(item[key]).strip()
    ]


def _is_id_list_format(fmt: str) -> bool:
    return (
        "comma-separated list of integer sheep ids" in fmt
        or ("sheep ids" in fmt and "separated by commas" in fmt)
        or ("escaping sheep ids" in fmt and "separated by commas" in fmt)
    )


def _is_integer_format(fmt: str) -> bool:
    return any(
        marker in fmt
        for marker in (
            "non-negative integer",
            "non-negative json integer",
            "single json integer",
            "valid integers for this task",
            "<integer>",
        )
    )


def _is_t04_color_list_format(fmt: str) -> bool:
    return (
        "json list of one or more strings drawn from" in fmt
        and all(color in fmt for color in T04_RING_COLORS)
    )


def _explicit_choices(
    doc: Mapping[str, Any],
    answer_format: str,
    meta: Mapping[str, Any],
) -> list[str]:
    option_types = meta.get("option_types")
    if isinstance(option_types, list):
        letters = _dict_values(option_types, "letter")
        if letters:
            return letters

    lower = answer_format.casefold()
    if not any(marker in lower for marker in ("chosen from", "one of", "legal answer")):
        return []
    if "such as" in lower or "e.g." in lower or "for example" in lower:
        return []
    values: list[str] = []
    for match in re.finditer(r'"([A-Za-z][A-Za-z0-9_]*)"', answer_format):
        value = match.group(1)
        if value.casefold() not in _SCHEMA_WORDS:
            values.append(value)
    return _dedupe(values)


def _bracketed_uppercase_choices(answer_format: str) -> list[str]:
    for match in re.finditer(r"\[([^\[\]]+)\]", answer_format):
        values = [value.strip() for value in match.group(1).split(",")]
        if len(values) >= 2 and all(re.fullmatch(r"[A-Z]", value) for value in values):
            return values
    return []


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out
