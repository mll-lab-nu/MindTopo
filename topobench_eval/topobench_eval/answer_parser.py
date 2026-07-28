"""Contract-driven parsing of TopoBench reasoning answers.

This module owns parsing orchestration only.  Answer schemas live in
``answer_contract`` and semantic normalization/equality live in
``answer_scoring``.  Keeping those responsibilities separate lets LMMS and
the interleaved runner share exactly the same behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from topobench_eval.answer_contract import AnswerContract, AnswerMode, contract_from_spec
from topobench_eval.answer_scoring import (
    canonicalize_id_list,
    canonicalize_name_list,
    exact_json_format,
    normalize_answer_value,
)
from topobench_eval.structured_output import answer_objects


UNCLEAR = "unclear"


@dataclass(frozen=True)
class ParseResult:
    """One parsed response, with syntax and semantics reported separately."""

    answer: str = UNCLEAR
    method: str = UNCLEAR
    semantic_valid: bool = False
    format_valid: bool = False
    error: str | None = None

    @property
    def invalid_response(self) -> bool:
        return not self.semantic_valid


_ANSWER_PHRASES = (
    re.compile(
        r"\b(?:final\s+answer|correct\s+answer|answer|final\s+decision|best\s+guess)"
        r"\s*(?:is|are|would\s+be|:|=)\s*(.{1,300}?)(?=[.!?](?:\s|$)|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i(?:'ll|\s+will|\s+would)?\s*(?:choose|select|pick|answer|go\s+with|settle\s+on))"
        r"\s*(?:option|choice)?\s*[:=]?\s*(.{1,300}?)(?=[.!?](?:\s|$)|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:escaping\s+sheep|sheep\s+(?:ids?|that\s+can\s+escape))"
        r"\s*(?:are|is|would\s+be|:)\s*(.{1,300}?)(?=[.!?](?:\s|$)|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:the\s+)?(?:order|sequence|color\s+sequence|colors)"
        r"\s*(?:is|are|would\s+be|:)\s*(.{1,300}?)(?=[.!?](?:\s|$)|$)",
        re.IGNORECASE,
    ),
)
_CONCLUSION_RE = re.compile(
    r"\b(?:therefore|thus|hence|so)\s*[,;:]?\s*"
    r"(.{1,160}?)(?=[.!?](?:\s|$)|$)",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def parse_answer_result(raw: Any, contract: AnswerContract) -> ParseResult:
    """Parse ``raw`` according to ``contract``.

    A syntactically recognizable ``{"answer": ...}`` whose value violates the
    contract is terminally invalid.  It is never rescued by scanning a legal
    subset from the surrounding prose; this is what prevents unknown list
    members from becoming false positives.
    """
    text = "" if raw is None else str(raw).strip()
    if not text:
        return ParseResult(error="empty_response")

    format_valid = exact_json_format(text, contract)
    objects = answer_objects(text)
    if objects:
        value = objects[-1]["answer"]
        normalized = normalize_answer_value(
            value,
            contract,
            tolerate_singleton_wrapper=True,
        )
        if normalized is None:
            return ParseResult(
                method="json",
                format_valid=format_valid,
                error="illegal_json_answer",
            )
        return ParseResult(
            answer=normalized,
            method="json",
            semantic_valid=True,
            format_valid=format_valid,
        )

    candidates = _phrase_candidates(text)
    if candidates:
        _, candidate = candidates[-1]
        normalized = _normalize_phrase(candidate, contract)
        if normalized is None:
            return ParseResult(
                method="phrase",
                error="illegal_explicit_answer",
            )
        return ParseResult(
            answer=normalized,
            method="phrase",
            semantic_valid=True,
            format_valid=False,
        )

    normalized = _normalize_bare(text, contract)
    if normalized is not None:
        return ParseResult(
            answer=normalized,
            method="scan",
            semantic_valid=True,
            format_valid=False,
        )

    conclusions = list(_CONCLUSION_RE.finditer(_flatten(text)[-5000:]))
    if conclusions:
        match = conclusions[-1]
        normalized = _normalize_phrase(_trim_segment(match.group(1)), contract)
        if normalized is None:
            return ParseResult(
                method="tail",
                error="illegal_explicit_answer",
            )
        return ParseResult(
            answer=normalized,
            method="tail",
            semantic_valid=True,
            format_valid=False,
        )

    return ParseResult(error="no_legal_answer")


def parse_answer(
    raw: Any,
    *,
    legal_spec: str | None = None,
    gt_hint: Any = None,
) -> tuple[str, str]:
    """Compatibility wrapper for the historical ``(answer, method)`` API."""
    result = parse_answer_result(raw, contract_from_spec(legal_spec, gt_hint))
    return result.answer, result.method


def _phrase_candidates(text: str) -> list[tuple[int, str]]:
    flattened = _flatten(text)[-5000:]
    candidates: list[tuple[int, str]] = []
    for pattern in _ANSWER_PHRASES:
        candidates.extend(
            (match.start(), _trim_segment(match.group(1)))
            for match in pattern.finditer(flattened)
        )
    return sorted(candidates)


def _flatten(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _trim_segment(segment: str) -> str:
    text = segment.strip().strip("\"'` ")
    text = re.split(
        r"(?i)(?:[.!?]\s+|\s+)(?:but|however|alternatively|wait|another\s+way|"
        r"looking\s+at|on\s+second\s+thought)\b",
        text,
        maxsplit=1,
    )[0]
    return text.strip().strip(" .;:!?\"'`")


def _normalize_phrase(segment: str, contract: AnswerContract) -> str | None:
    if not segment:
        return None

    direct = normalize_answer_value(segment, contract, tolerate_singleton_wrapper=True)
    if direct is not None:
        return direct

    if contract.mode is AnswerMode.INTEGER:
        match = re.match(r"^(-?\d+)\b", segment)
        if match:
            return normalize_answer_value(match.group(1), contract)
        word = re.match(r"^([A-Za-z]+)\b", segment)
        if word and word.group(1).casefold() in _NUMBER_WORDS:
            return normalize_answer_value(_NUMBER_WORDS[word.group(1).casefold()], contract)
        return None

    if contract.mode is AnswerMode.CHOICE:
        return _choice_at_start(segment, contract)

    # A list/sequence may be followed by a short justification.  Only consume
    # the leading list grammar, never arbitrary names/numbers later in prose.
    if contract.mode in {AnswerMode.ID_LIST, AnswerMode.NAME_LIST, AnswerMode.COLOR_SEQUENCE}:
        first_clause = re.split(r"[.;!?]", segment, maxsplit=1)[0].strip()
        return normalize_answer_value(first_clause, contract)

    if contract.mode is AnswerMode.TEXT:
        return segment
    return None


def _normalize_bare(text: str, contract: AnswerContract) -> str | None:
    stripped = text.strip()
    if len(stripped) > 500 or "\n" in stripped:
        return None
    stripped = stripped.strip().strip("`\"'").strip()
    stripped = re.sub(r"[.!?]+$", "", stripped).strip()
    return normalize_answer_value(stripped, contract, tolerate_singleton_wrapper=True)


def _choice_at_start(text: str, contract: AnswerContract) -> str | None:
    candidate = re.sub(r"(?i)^(?:option|choice)\s*", "", text.strip())
    for legal in sorted(contract.legal_values, key=len, reverse=True):
        match = re.match(re.escape(legal), candidate, re.IGNORECASE)
        if not match:
            continue
        tail = candidate[match.end():]
        if not tail or re.match(r"^(?:\s|[).,:;!?-])", tail):
            return normalize_answer_value(legal, contract)
    return None


__all__ = [
    "ParseResult",
    "parse_answer",
    "parse_answer_result",
    "canonicalize_id_list",
    "canonicalize_name_list",
]
