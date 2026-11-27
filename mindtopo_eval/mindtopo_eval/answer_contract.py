"""Declarative answer contracts shared by all reasoning evaluators."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Iterable


class AnswerMode(str, Enum):
    INTEGER = "integer"
    CHOICE = "choice"
    ID_LIST = "id_list"
    NAME_LIST = "name_list"
    COLOR_SEQUENCE = "color_sequence"
    TEXT = "text"


@dataclass(frozen=True)
class AnswerContract:
    """The schema and semantic domain of one benchmark answer.

    ``legal_values`` is case-insensitive. ``NAME_LIST`` contracts represent
    unordered sets; ``COLOR_SEQUENCE`` remains ordered. ``none_token`` is an
    explicit list member such as ``"none"`` and is never inferred from an
    empty list.
    """

    mode: AnswerMode
    legal_values: tuple[str, ...] = ()
    allow_empty: bool = False
    none_token: str | None = None
    non_negative: bool = False

    def __post_init__(self) -> None:
        values: list[str] = []
        seen: set[str] = set()
        for raw in self.legal_values:
            value = str(raw).strip()
            key = value.casefold()
            if value and key not in seen:
                seen.add(key)
                values.append(value)
        object.__setattr__(self, "legal_values", tuple(values))

    @property
    def legal_spec(self) -> str | None:
        if self.mode is AnswerMode.INTEGER:
            return "INTEGER"
        if self.mode is AnswerMode.ID_LIST:
            return "ID_LIST"
        if self.mode is AnswerMode.NAME_LIST:
            return "NAME_LIST:" + ",".join(self.legal_values)
        if self.mode is AnswerMode.COLOR_SEQUENCE:
            return "COLOR_SEQUENCE"
        if self.mode is AnswerMode.CHOICE:
            return "|".join(self.legal_values)
        return None

    @property
    def legacy_mode(self) -> str:
        if self.mode is AnswerMode.CHOICE and self.legal_values and all(
            re.fullmatch(r"[A-Za-z]", value) for value in self.legal_values
        ):
            return "letter"
        if self.mode is AnswerMode.TEXT:
            return "string"
        return self.mode.value

    def prompt_instruction(self) -> str:
        if self.mode is AnswerMode.INTEGER:
            return 'Return JSON only: {"answer": <integer>}.'
        if self.mode is AnswerMode.CHOICE:
            choices = ", ".join(f'"{value}"' for value in self.legal_values)
            return f'Return JSON only: {{"answer": "<choice>"}}. Legal choices: {choices}.'
        if self.mode is AnswerMode.NAME_LIST:
            empty = " An empty list is allowed." if self.allow_empty else " Do not return an empty list."
            legal = ", ".join(f'"{value}"' for value in self.legal_values)
            domain = f" Legal names: {legal}." if legal else ""
            return 'Return JSON only: {"answer": ["<name>"]}.' + domain + empty
        if self.mode is AnswerMode.ID_LIST:
            return 'Return JSON only: {"answer": "<comma-separated IDs or NONE>"}.'
        if self.mode is AnswerMode.COLOR_SEQUENCE:
            return 'Return JSON only: {"answer": "<comma-separated color sequence>"}.'
        return 'Return JSON only: {"answer": "<value>"}.'


_COLOR_WORDS = frozenset(
    {
        "red", "blue", "green", "yellow", "orange", "purple", "white",
        "black", "pink", "cyan", "brown",
    }
)


def contract_from_spec(legal_spec: str | None, gt_hint: Any = None) -> AnswerContract:
    """Build a contract from the legacy parser specification."""
    spec = str(legal_spec or "").strip()
    upper = spec.upper()
    if upper == "INTEGER":
        return AnswerContract(AnswerMode.INTEGER)
    if upper == "ID_LIST":
        return AnswerContract(AnswerMode.ID_LIST, none_token="none")
    if upper.startswith("NAME_LIST:"):
        values = tuple(value.strip() for value in spec.split(":", 1)[1].split(",") if value.strip())
        none = next((value for value in values if value.casefold() == "none"), None)
        return AnswerContract(
            AnswerMode.NAME_LIST,
            legal_values=values,
            allow_empty=none is None,
            none_token=none,
        )
    if upper in {"COLOR_SEQUENCE", "BEAD_COLOR_SEQUENCE"}:
        return AnswerContract(AnswerMode.COLOR_SEQUENCE, legal_values=tuple(sorted(_COLOR_WORDS)))
    if spec:
        values = tuple(value.strip() for value in spec.split("|") if value.strip())
        return AnswerContract(AnswerMode.CHOICE, legal_values=values)

    if gt_hint is None:
        return AnswerContract(AnswerMode.TEXT)
    gt = str(gt_hint).strip()
    if re.fullmatch(r"-?\d+", gt):
        return AnswerContract(AnswerMode.INTEGER)
    if gt.casefold() == "none" or re.fullmatch(r"\d+(?:\s*,\s*\d+)+", gt):
        return AnswerContract(AnswerMode.ID_LIST, none_token="none")
    tokens = [token.strip().casefold() for token in gt.split(",") if token.strip()]
    if tokens and all(token in _COLOR_WORDS for token in tokens):
        return AnswerContract(AnswerMode.COLOR_SEQUENCE, legal_values=tuple(sorted(_COLOR_WORDS)))
    if re.fullmatch(r"[A-Za-z]", gt):
        return AnswerContract(
            AnswerMode.CHOICE,
            legal_values=tuple(chr(code) for code in range(ord("A"), ord("Z") + 1)),
        )
    return AnswerContract(AnswerMode.TEXT)


def choice_contract(values: Iterable[Any]) -> AnswerContract:
    return AnswerContract(AnswerMode.CHOICE, tuple(str(value).strip() for value in values if str(value).strip()))
