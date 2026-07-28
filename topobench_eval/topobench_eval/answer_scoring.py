"""Normalization, exact-format validation, and semantic answer equality."""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Iterable

from topobench_eval.answer_contract import AnswerContract, AnswerMode
from topobench_eval.structured_output import strict_int


_NONE_RE = re.compile(
    r"\b(?:none|empty|no\s+(?:sheep|ones?|animals?)|no\s+sheep\s+can\s+escape)\b",
    re.IGNORECASE,
)


def normalize_answer_value(
    value: Any,
    contract: AnswerContract,
    *,
    tolerate_singleton_wrapper: bool = False,
) -> str | None:
    """Return the canonical string representation or ``None`` when illegal."""
    if tolerate_singleton_wrapper and isinstance(value, (list, tuple)) and len(value) == 1:
        if contract.mode in {AnswerMode.INTEGER, AnswerMode.CHOICE, AnswerMode.ID_LIST, AnswerMode.TEXT}:
            value = value[0]

    if contract.mode is AnswerMode.INTEGER:
        parsed = strict_int(value)
        if parsed is None or (contract.non_negative and parsed < 0):
            return None
        return str(parsed)
    if contract.mode is AnswerMode.CHOICE:
        return _normalize_choice(value, contract)
    if contract.mode is AnswerMode.ID_LIST:
        return _normalize_id_list(value)
    if contract.mode is AnswerMode.NAME_LIST:
        return _normalize_name_list(value, contract)
    if contract.mode is AnswerMode.COLOR_SEQUENCE:
        return _normalize_color_sequence(value, contract)
    text = str(value).strip() if isinstance(value, str) else ""
    return text or None


def answers_equal(predicted: Any, ground_truth: Any, contract: AnswerContract) -> bool:
    predicted_value = normalize_answer_value(
        predicted,
        contract,
        tolerate_singleton_wrapper=True,
    )
    ground_truth_value = normalize_answer_value(
        ground_truth,
        contract,
        tolerate_singleton_wrapper=True,
    )
    if predicted_value is None or ground_truth_value is None:
        return False
    return predicted_value.casefold() == ground_truth_value.casefold()


def exact_json_format(raw: str, contract: AnswerContract) -> bool:
    """Validate the requested top-level JSON shape and answer value type."""
    try:
        payload = json.loads(str(raw).strip())
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(payload, dict) or set(payload) != {"answer"}:
        return False
    value = payload["answer"]
    if contract.mode is AnswerMode.INTEGER:
        return (
            type(value) is int
            and (not contract.non_negative or value >= 0)
        )
    if contract.mode is AnswerMode.CHOICE:
        return isinstance(value, str) and _normalize_choice(value, contract) is not None
    if contract.mode is AnswerMode.ID_LIST:
        return isinstance(value, str) and bool(
            re.fullmatch(r"(?i)(?:none|\d+(?:\s*,\s*\d+)*)", value.strip())
        )
    if contract.mode is AnswerMode.COLOR_SEQUENCE:
        if not isinstance(value, str) or not re.fullmatch(
            r"[A-Za-z]+(?:\s*,\s*[A-Za-z]+)*",
            value.strip(),
        ):
            return False
        return _normalize_color_sequence(value, contract) is not None
    if contract.mode is AnswerMode.TEXT:
        return isinstance(value, str) and bool(value.strip())
    if contract.mode is AnswerMode.NAME_LIST:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return False
        folded = [item.strip().casefold() for item in value]
        if len(folded) != len(set(folded)):
            return False
        return _normalize_name_list(value, contract) is not None
    return False


def canonicalize_name_list(
    value: Any,
    *,
    legal_keys: Iterable[str] | None = None,
) -> str:
    """Compatibility helper returning sorted comma-joined uppercase names."""
    keys = tuple(str(key).strip() for key in (legal_keys or ()) if str(key).strip())
    contract = AnswerContract(AnswerMode.NAME_LIST, legal_values=keys, allow_empty=True)
    normalized = normalize_answer_value(value, contract)
    if normalized is None:
        return ""
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError:
        return ""
    return ",".join(sorted(str(item).upper() for item in parsed))


def canonicalize_id_list(value: Any) -> str:
    normalized = _normalize_id_list(value)
    return normalized or ""


def _normalize_choice(value: Any, contract: AnswerContract) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    by_key = {_normalize_key(item): item.upper() for item in contract.legal_values}
    exact = by_key.get(_normalize_key(text))
    return exact


def _normalize_id_list(value: Any) -> str | None:
    if isinstance(value, (list, tuple, set)):
        if not value:
            return "none"
        ids: list[int] = []
        for item in value:
            parsed = strict_int(item)
            if parsed is None or parsed < 0:
                return None
            ids.append(parsed)
        return ",".join(str(item) for item in sorted(set(ids)))
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if re.fullmatch(r"(?i)(?:none|empty|\[\s*\])", text) or _NONE_RE.fullmatch(text):
        return "none"
    if not re.fullmatch(r"\d+(?:\s*(?:,|and)\s*\d+)*", text, re.IGNORECASE):
        return None
    ids = [int(item) for item in re.findall(r"\d+", text)]
    return ",".join(str(item) for item in sorted(set(ids)))


def _normalize_name_list(value: Any, contract: AnswerContract) -> str | None:
    items = _list_items(value)
    if items is None:
        return None
    if not items:
        return "[]" if contract.allow_empty else None
    if not all(isinstance(item, str) and item.strip() for item in items):
        return None

    legal = {item.casefold(): item for item in contract.legal_values}
    canonical: list[str] = []
    seen: set[str] = set()
    for raw in items:
        key = raw.strip().casefold()
        if legal and key not in legal:
            return None
        value_name = legal.get(key, raw.strip())
        if key not in seen:
            seen.add(key)
            canonical.append(value_name)

    none_key = contract.none_token.casefold() if contract.none_token else None
    if none_key and none_key in seen:
        if len(seen) != 1:
            return None
        canonical = [legal.get(none_key, contract.none_token or "none")]

    order = {value.casefold(): index for index, value in enumerate(contract.legal_values)}
    canonical.sort(key=lambda item: (order.get(item.casefold(), len(order)), item.casefold()))
    return json.dumps(canonical, ensure_ascii=False)


def _normalize_color_sequence(value: Any, contract: AnswerContract) -> str | None:
    if isinstance(value, (list, tuple)):
        raw_items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        raw_items = [item for item in re.split(r"\s*(?:,|->|→|\band\b|\bthen\b)\s*", text, flags=re.I) if item]
    else:
        return None
    if not raw_items or not all(isinstance(item, str) for item in raw_items):
        return None
    legal = {item.casefold(): item.upper() for item in contract.legal_values}
    out: list[str] = []
    for raw in raw_items:
        key = raw.strip().casefold()
        if not re.fullmatch(r"[A-Za-z]+", raw.strip()) or (legal and key not in legal):
            return None
        out.append(legal.get(key, raw.strip().upper()))
    return ", ".join(out)


def _list_items(value: Any) -> list[Any] | None:
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [part.strip().strip("\"'") for part in inner.split(",")]
    if "," in text:
        return [part.strip().strip("\"'") for part in text.split(",") if part.strip()]
    return None


def _normalize_key(value: str) -> str:
    return re.sub(r"[\s_-]+", "_", str(value).strip().casefold())
