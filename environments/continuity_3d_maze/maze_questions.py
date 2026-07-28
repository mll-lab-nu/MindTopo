from __future__ import annotations

import json
import random
import re
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


QUESTION_TYPE_CONNECTED_SUBSET_CHOICE = "connected_subset_choice"
QUESTION_TYPE_CONNECTED_POINT_LIST = "connected_point_list"
QUESTION_TYPE_DOOR_OPEN = "door_open"
QUESTION_TYPES: Tuple[str, ...] = (
    QUESTION_TYPE_CONNECTED_SUBSET_CHOICE,
    QUESTION_TYPE_CONNECTED_POINT_LIST,
    QUESTION_TYPE_DOOR_OPEN,
)
QUESTION_TYPE_CLI_SUBSET_CHOICE = "subset_choice"
QUESTION_TYPE_CLI_POINT_LIST = "point_list"
QUESTION_TYPE_CLI_DOOR_OPEN = "door_open"
QUESTION_TYPE_CLI_NAMES: Tuple[str, ...] = (
    QUESTION_TYPE_CLI_SUBSET_CHOICE,
    QUESTION_TYPE_CLI_POINT_LIST,
    QUESTION_TYPE_CLI_DOOR_OPEN,
)
DEFAULT_QUESTION_TYPES_CLI = QUESTION_TYPE_CLI_POINT_LIST
QUESTION_TYPE_CLI_TO_INTERNAL = {
    QUESTION_TYPE_CLI_SUBSET_CHOICE: QUESTION_TYPE_CONNECTED_SUBSET_CHOICE,
    QUESTION_TYPE_CLI_POINT_LIST: QUESTION_TYPE_CONNECTED_POINT_LIST,
    QUESTION_TYPE_CLI_DOOR_OPEN: QUESTION_TYPE_DOOR_OPEN,
}

ANSWER_TYPE_MULTIPLE_CHOICE = "multiple_choice"
ANSWER_TYPE_NAME_LIST = "name_list"
ANSWER_TYPE_COLOR_LIST = "color_list"
ANSWER_TYPE_YES_NO = "yes_no"

OPTION_LABELS: Tuple[str, str, str, str] = ("1", "2", "3", "4")
MIN_POINT_COUNT = 3

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_BRACED_OBJECT_RE = re.compile(r"\{[^{}]*\}")


def parse_question_types(spec: str | Sequence[str] | None) -> Tuple[str, ...]:
    if spec is None:
        spec = DEFAULT_QUESTION_TYPES_CLI
    allow_internal_names = not isinstance(spec, str)
    if isinstance(spec, str):
        raw_items = [item.strip() for item in spec.replace("，", ",").split(",")]
        normalized_cli_items = tuple(item.casefold() for item in raw_items if item)
        if not normalized_cli_items:
            raise ValueError("At least one continuity_3d_maze question type must be selected.")
        unknown_items = [
            item for item in normalized_cli_items if item not in QUESTION_TYPE_CLI_TO_INTERNAL
        ]
        if unknown_items:
            legal = ", ".join(QUESTION_TYPE_CLI_NAMES)
            raise ValueError(
                f"Unsupported question type setting {spec!r}. Use a comma-separated subset of: {legal}."
            )
        raw_items = list(normalized_cli_items)
    else:
        raw_items = [str(item).strip() for item in spec]

    selected: List[str] = []
    for raw_item in raw_items:
        if not raw_item:
            continue
        lowered = raw_item.casefold()
        question_type = (
            raw_item
            if allow_internal_names and raw_item in QUESTION_TYPES
            else QUESTION_TYPE_CLI_TO_INTERNAL.get(lowered)
        )
        if question_type is None:
            legal = ", ".join(QUESTION_TYPE_CLI_NAMES)
            raise ValueError(
                f"Unsupported question type setting {raw_item!r}. Use a comma-separated subset of: {legal}."
            )
        if question_type not in selected:
            selected.append(question_type)

    if not selected:
        raise ValueError("At least one continuity_3d_maze question type must be selected.")
    return tuple(selected)


def question_count_for_point_count(
    point_count: int,
    question_types: str | Sequence[str] | None = None,
) -> int:
    selected_question_types = parse_question_types(question_types)
    count = 0
    if QUESTION_TYPE_CONNECTED_SUBSET_CHOICE in selected_question_types:
        count += 1
    if QUESTION_TYPE_CONNECTED_POINT_LIST in selected_question_types:
        count += 1
    if QUESTION_TYPE_DOOR_OPEN in selected_question_types:
        count += 1
    return count


def point_names_from_maze_state(maze_state: Dict[str, Any]) -> List[str]:
    return [str(point["name"]) for point in maze_state.get("points", [])]


def connected_pair_lookup(maze_state: Dict[str, Any]) -> set[frozenset[str]]:
    lookup: set[frozenset[str]] = set()
    for pair in maze_state.get("pairs", []):
        if pair.get("connected"):
            a_name = str(pair["a_name"])
            b_name = str(pair["b_name"])
            lookup.add(frozenset((a_name, b_name)))
    return lookup


def are_points_mutually_connected(
    point_names: Sequence[str],
    connected_pairs: Iterable[frozenset[str]],
) -> bool:
    normalized = [str(name) for name in point_names]
    if len(normalized) < 2:
        return False

    connected_lookup = set(connected_pairs)
    for a_name, b_name in combinations(normalized, 2):
        if frozenset((a_name, b_name)) not in connected_lookup:
            return False
    return True


def connected_points_for_source(
    maze_state: Dict[str, Any],
    source_name: str,
) -> List[str]:
    connected_lookup = connected_pair_lookup(maze_state)
    source = str(source_name)
    names = point_names_from_maze_state(maze_state)
    return sorted(
        other_name
        for other_name in names
        if other_name != source and frozenset((source, other_name)) in connected_lookup
    )


def _all_point_subsets(point_names: Sequence[str]) -> List[Tuple[str, ...]]:
    names = [str(name) for name in point_names]
    subsets: List[Tuple[str, ...]] = []
    for subset_size in range(2, len(names) + 1):
        subsets.extend(tuple(group) for group in combinations(names, subset_size))
    return subsets


def _subset_choice_pools(
    maze_state: Dict[str, Any],
) -> Tuple[List[Tuple[str, ...]], List[Tuple[str, ...]]]:
    names = point_names_from_maze_state(maze_state)
    connected_lookup = connected_pair_lookup(maze_state)
    valid_subsets: List[Tuple[str, ...]] = []
    invalid_subsets: List[Tuple[str, ...]] = []
    for subset in _all_point_subsets(names):
        if are_points_mutually_connected(subset, connected_lookup):
            valid_subsets.append(subset)
        else:
            invalid_subsets.append(subset)
    return valid_subsets, invalid_subsets


def supports_connected_subset_choice(maze_state: Dict[str, Any]) -> bool:
    valid_subsets, invalid_subsets = _subset_choice_pools(maze_state)
    return len(valid_subsets) >= 1 and len(invalid_subsets) >= len(OPTION_LABELS) - 1


def _format_point_list(points: Sequence[str]) -> str:
    return "[" + ", ".join(str(point) for point in points) + "]"


def _subset_choice_shuffle_seed(maze_state: Dict[str, Any]) -> Any:
    if "subset_choice_shuffle_seed" in maze_state:
        return maze_state["subset_choice_shuffle_seed"]
    stable_state = {
        "points": point_names_from_maze_state(maze_state),
        "pairs": sorted(
            (
                str(pair.get("a_name")),
                str(pair.get("b_name")),
                bool(pair.get("connected")),
            )
            for pair in maze_state.get("pairs", [])
        ),
    }
    return json.dumps(stable_state, sort_keys=True, separators=(",", ":"))


def _choice_question_text(options: Sequence[Dict[str, Any]]) -> str:
    option_space = ", ".join(str(option["label"]) for option in options)
    lines = [
        "Which option lists labeled points that are all mutually connected under the shown door states?",
        "Each option contains two or more labeled points.",
        (
            "A listed option is correct only if every point in that option can reach every other point "
            "in the same option through a collision-free navigable path."
        ),
        f"Exactly one option is correct. Choose one option from {option_space}.",
        "",
    ]
    for option in options:
        lines.append(
            f"Option {option['label']}: {_format_point_list(option['points'])}"
        )
    return "\n".join(lines)


def build_connected_subset_choice_question(
    maze_state: Dict[str, Any],
) -> Dict[str, Any]:
    valid_subsets, invalid_subsets = _subset_choice_pools(maze_state)
    if len(valid_subsets) < 1 or len(invalid_subsets) < len(OPTION_LABELS) - 1:
        raise ValueError(
            f"Unable to build a {len(OPTION_LABELS)}-option connected-subset question from this maze state."
        )

    valid_subsets = sorted(valid_subsets, key=lambda subset: (-len(subset), subset))
    correct_subset = valid_subsets[0]
    ranked_invalids = sorted(
        invalid_subsets,
        key=lambda subset: (
            abs(len(subset) - len(correct_subset)),
            -len(set(subset) & set(correct_subset)),
            -len(subset),
            subset,
        ),
    )
    distractors = ranked_invalids[: len(OPTION_LABELS) - 1]
    option_entries = [(correct_subset, True), *((subset, False) for subset in distractors)]
    random.Random(_subset_choice_shuffle_seed(maze_state)).shuffle(option_entries)
    options = []
    correct_label: Optional[str] = None
    for label, (subset, is_correct) in zip(OPTION_LABELS, option_entries):
        options.append({"label": label, "points": list(subset)})
        if is_correct:
            correct_label = label
    if correct_label is None:
        raise RuntimeError("Unable to locate the shuffled correct subset option.")
    return {
        "question_type": QUESTION_TYPE_CONNECTED_SUBSET_CHOICE,
        "answer_type": ANSWER_TYPE_MULTIPLE_CHOICE,
        "question": _choice_question_text(options),
        "ground_truth": correct_label,
        "question_payload": {
            "options": options,
            "correct_option": correct_label,
            "connected_subset": list(correct_subset),
            "subset_choice_shuffle_seed": maze_state.get("subset_choice_shuffle_seed"),
        },
        "highlighted_points": [],
    }


def build_connected_point_list_question(
    maze_state: Dict[str, Any],
) -> Dict[str, Any]:
    point_names = point_names_from_maze_state(maze_state)
    if not point_names:
        raise ValueError("At least one labeled point is required for a point-list question.")

    source_name = str(maze_state.get("point_list_source_name") or point_names[0])
    if source_name not in point_names:
        raise ValueError(f"Unknown point-list target point {source_name!r}.")
    other_points = [name for name in point_names if name != source_name]
    other_points_text = ", ".join(other_points) if other_points else "(none)"
    connected_points = connected_points_for_source(maze_state, source_name)
    return {
        "question_type": QUESTION_TYPE_CONNECTED_POINT_LIST,
        "answer_type": ANSWER_TYPE_NAME_LIST,
        "question": (
            f"List all other labeled points that are connected to point {source_name} "
            f"under the shown door states. The other labeled points are: {other_points_text}."
        ),
        "ground_truth": connected_points,
        "question_payload": {
            "source_name": source_name,
            "candidate_points": other_points,
            "connected_points": connected_points,
        },
        "highlighted_points": [source_name],
    }


def build_door_open_question(
    maze_state: Dict[str, Any],
) -> Dict[str, Any]:
    payload = dict(maze_state.get("door_open_question") or {})
    if not payload:
        raise ValueError("This maze state does not contain a valid door_open question.")

    source_name = str(payload["source_name"])
    target_name = str(payload["target_name"])
    door_colors = list(payload.get("door_colors") or [])
    answer_colors = sorted(str(color) for color in payload.get("answer_colors") or [])
    if not answer_colors:
        raise ValueError("door_open questions require at least one answer door color.")

    color_names = ", ".join(str(entry["color_name"]) for entry in door_colors)
    return {
        "question_type": QUESTION_TYPE_DOOR_OPEN,
        "answer_type": ANSWER_TYPE_COLOR_LIST,
        "question": (
            f"Points {source_name} and {target_name} are not connected under the shown current door states. "
            "Open the unique minimum set of currently closed colored doors needed to make those two points connected. "
            f"The colored doors are: {color_names}. "
            "Answer only with the color names of the doors that must be opened."
        ),
        "ground_truth": answer_colors,
        "question_payload": payload,
        "highlighted_points": [source_name, target_name],
        "highlighted_doors": door_colors,
    }


def build_scene_question_specs(
    maze_state: Dict[str, Any],
    question_types: str | Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    if len(point_names_from_maze_state(maze_state)) < MIN_POINT_COUNT:
        raise ValueError(
            f"At least {MIN_POINT_COUNT} labeled points are required for the current 3D maze question set."
        )
    selected_question_types = parse_question_types(question_types)
    question_specs: List[Dict[str, Any]] = []
    for question_type in selected_question_types:
        if question_type == QUESTION_TYPE_CONNECTED_SUBSET_CHOICE:
            question_specs.append(build_connected_subset_choice_question(maze_state))
        elif question_type == QUESTION_TYPE_CONNECTED_POINT_LIST:
            question_specs.append(build_connected_point_list_question(maze_state))
        elif question_type == QUESTION_TYPE_DOOR_OPEN:
            question_specs.append(build_door_open_question(maze_state))
        else:
            raise ValueError(f"Unsupported question type for continuity_3d_maze: {question_type}")
    return question_specs


def build_user_prompt_blocks(
    question: str,
    answer_type: str,
    *,
    option_labels: Optional[Sequence[str]] = None,
) -> Tuple[str, str]:
    shared_prefix = (
        "[Task]\n"
        "You are solving a topological task called 3D Maze under the continuity category.\n"
    )
    shared_definition = (
        "Two labeled points are connected only if an agent can travel between them without crossing walls, "
        "closed doors, furniture, or other solid obstacles. Open doors and open floor/corridor spaces are "
        "passable. Closed doors are not passable.\n"
        "The visual evidence for this question is provided below."
    )
    rules = (
        "[Rules]\n"
        "1. Use only the images and text provided in this prompt.\n"
        "2. If answer options are provided, choose only from the provided options.\n"
        "3. Do not output explanation beyond the required final answer.\n\n"
        "[Question]\n"
        f"{question}\n\n"
        "[Answer Format]\n"
    )

    if answer_type == ANSWER_TYPE_MULTIPLE_CHOICE:
        labels = list(option_labels or OPTION_LABELS)
        option_space = ", ".join(f'"{label}"' for label in labels)
        before_images = (
            shared_prefix
            + "In this task, you must determine which candidate option lists labeled points that are all mutually connected under the shown door states.\n"
            + shared_definition
        )
        after_images = (
            rules
            + 'Output exactly one JSON object: {"answer":"{ans}"} and nothing else.\n'
            + f'Replace {{ans}} with the single legal answer for this task, chosen from {option_space}.\n'
        )
        return before_images, after_images

    if answer_type == ANSWER_TYPE_NAME_LIST:
        before_images = (
            shared_prefix
            + "In this task, you must list every other labeled point that is connected to the target point under the shown door states.\n"
            + shared_definition
        )
        after_images = (
            rules
            + 'Output exactly one JSON object: {"answer":{ans}} and nothing else.\n'
            + "Replace {ans} with the actual answer: a JSON array of point names, excluding the target point itself, listed in alphabetical order. Use [] if no other labeled point is connected to the target.\n"
        )
        return before_images, after_images

    if answer_type == ANSWER_TYPE_COLOR_LIST:
        before_images = (
            shared_prefix
            + "In this task, you must identify which colored doors need to be opened to connect two target points under the shown current door states.\n"
            + shared_definition
        )
        after_images = (
            rules
            + 'Output exactly one JSON object: {"answer":{ans}} and nothing else.\n'
            + "Replace {ans} with the actual answer: a JSON array of door color names. Use the unique minimum set of currently closed colored doors that must be opened, listed in alphabetical order.\n"
        )
        return before_images, after_images

    if answer_type == ANSWER_TYPE_YES_NO:
        before_images = (
            shared_prefix
            + "In this task, you must determine whether two labeled points in a furnished 3D maze/house are connected by a physically valid navigable path.\n"
            + shared_definition
        )
        after_images = (
            rules
            + 'Output exactly one JSON object: {"answer":"{ans}"} and nothing else.\n'
            + 'Replace {ans} with the single legal answer for this task, chosen from "yes" or "no".\n'
        )
        return before_images, after_images

    raise ValueError(f"Unsupported answer_type for continuity_3d_maze: {answer_type}")


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _to_string_list(value: Any) -> Optional[List[str]]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            inner = json.loads(text)
        except json.JSONDecodeError:
            inner = None
        if isinstance(inner, list):
            return [str(item).strip() for item in inner if str(item).strip()]
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if not text.strip():
            return []
        return [
            token
            for token in (part.strip().strip('"').strip("'") for part in text.split(","))
            if token
        ]
    return None


def _normalize_option_label(
    value: Any,
    *,
    legal_options: Sequence[str] = OPTION_LABELS,
) -> Optional[str]:
    legal = {str(option).strip(): str(option).strip() for option in legal_options}
    if isinstance(value, int):
        return legal.get(str(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lowered = text.casefold()
        for prefix in ("option ", "choice ", "answer "):
            if lowered.startswith(prefix):
                text = text[len(prefix) :].strip()
                break
        return legal.get(text)
    return None


def parse_multiple_choice_answer(
    raw_text: str,
    *,
    legal_options: Sequence[str] = OPTION_LABELS,
) -> Optional[str]:
    if not raw_text:
        return None
    text = _strip_markdown_fence(raw_text.strip())

    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "answer" in payload:
            return _normalize_option_label(payload["answer"], legal_options=legal_options)
    except json.JSONDecodeError:
        pass

    for brace_text in reversed(_BRACED_OBJECT_RE.findall(text)):
        try:
            payload = json.loads(brace_text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "answer" in payload:
            parsed = _normalize_option_label(payload["answer"], legal_options=legal_options)
            if parsed is not None:
                return parsed

    legal_option_pattern = "|".join(
        re.escape(str(option)) for option in sorted(legal_options, key=lambda item: -len(str(item)))
    )
    explicit_option_match = re.search(
        rf"\b(?:option|choice)\s*({legal_option_pattern})\b",
        text,
        flags=re.IGNORECASE,
    )
    if explicit_option_match:
        return _normalize_option_label(explicit_option_match.group(1), legal_options=legal_options)

    stripped = text.strip().strip('"').strip("'")
    parsed = _normalize_option_label(stripped, legal_options=legal_options)
    if parsed is not None:
        return parsed

    fallback_matches = list(
        re.finditer(rf"(?<!\w)({legal_option_pattern})(?!\w)", text, flags=re.IGNORECASE)
    )
    for match in reversed(fallback_matches):
        parsed = _normalize_option_label(match.group(1), legal_options=legal_options)
        if parsed is not None:
            return parsed
    return None


def parse_name_list_answer(raw_text: str) -> Optional[List[str]]:
    if not raw_text:
        return None
    text = _strip_markdown_fence(raw_text.strip())

    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "answer" in payload:
            return _to_string_list(payload["answer"])
    except json.JSONDecodeError:
        pass

    for brace_text in reversed(_BRACED_OBJECT_RE.findall(text)):
        try:
            payload = json.loads(brace_text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "answer" in payload:
            parsed = _to_string_list(payload["answer"])
            if parsed is not None:
                return parsed

    boxed_hits = _BOXED_RE.findall(text)
    if boxed_hits:
        collected: List[str] = []
        for hit in boxed_hits:
            parsed = _to_string_list(hit)
            if parsed is not None:
                collected.extend(parsed)
        if collected:
            return collected

    for bracket_text in reversed(re.findall(r"\[[^\[\]]*\]", text, flags=re.DOTALL)):
        parsed = _to_string_list(bracket_text)
        if parsed is not None:
            return parsed
    return None


def parse_yes_no_answer(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None
    lowered = raw_text.casefold()
    try:
        payload = json.loads(lowered)
        if isinstance(payload, dict) and "answer" in payload and isinstance(payload["answer"], str):
            normalized = payload["answer"].strip().casefold()
            if normalized in {"yes", "no"}:
                return normalized
    except json.JSONDecodeError:
        pass

    for brace_text in re.findall(r"\{[^{}]*\}", lowered, flags=re.DOTALL):
        brace_answer_match = re.search(r"\b(yes|no)\b", brace_text)
        if brace_answer_match:
            return brace_answer_match.group(1)

    answer_matches = list(re.finditer(r"\b(yes|no)\b", lowered))
    if answer_matches:
        return answer_matches[-1].group(1)
    return None


def parse_answer(raw_text: str, answer_type: str) -> Optional[Any]:
    if answer_type == ANSWER_TYPE_MULTIPLE_CHOICE:
        return parse_multiple_choice_answer(raw_text)
    if answer_type in {ANSWER_TYPE_NAME_LIST, ANSWER_TYPE_COLOR_LIST}:
        return parse_name_list_answer(raw_text)
    if answer_type == ANSWER_TYPE_YES_NO:
        return parse_yes_no_answer(raw_text)
    raise ValueError(f"Unsupported answer_type for continuity_3d_maze: {answer_type}")


def answers_equal(
    predicted_answer: Any,
    ground_truth: Any,
    answer_type: str,
) -> bool:
    if answer_type == ANSWER_TYPE_MULTIPLE_CHOICE:
        predicted = _normalize_option_label(predicted_answer)
        truth = _normalize_option_label(ground_truth)
        return predicted is not None and predicted == truth
    if answer_type in {ANSWER_TYPE_NAME_LIST, ANSWER_TYPE_COLOR_LIST}:
        predicted = _to_string_list(predicted_answer)
        truth = _to_string_list(ground_truth)
        if predicted is None or truth is None:
            return False
        return frozenset(item.casefold() for item in predicted) == frozenset(
            item.casefold() for item in truth
        )
    if answer_type == ANSWER_TYPE_YES_NO:
        return parse_yes_no_answer(str(predicted_answer or "")) == parse_yes_no_answer(
            str(ground_truth or "")
        )
    raise ValueError(f"Unsupported answer_type for continuity_3d_maze: {answer_type}")
