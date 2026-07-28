"""Prompt templates for continuity_2d_maze.

iter E1 (dev_trace_5, 2026-04-24): templates rewritten to the unified
reasoning-question format requested by the collaborator. Both
REACHABILITY_SET_TEMPLATE and BAR_REMOVAL_TEMPLATE share a single
[Task] / [Rules] / [Question] / [Answer Format] skeleton; per-task
descriptions and visual definitions are baked in. The answer-format
text uses schema placeholders rather than concrete labels/colors, so it
does not leak an answer-shaped candidate list. The placeholders left behind for downstream consumers are
unchanged: `{question}` and `{{images}}` (escaped doubled-brace; becomes
a literal `{images}` after `.format()`).

* `REACHABILITY_SET_TEMPLATE` — Q1 (reachability_set). Model lists the other
  labeled points connected to the target; answer is a JSON array of names.
* `BAR_REMOVAL_TEMPLATE` — Q2. Model lists the colors of bars whose
  single-bar removal connects A and B; answer is a JSON array of colors.

MUST STAY IN SYNC with `frontend/src/maze/prompts.ts` — run
`python backend/check_prompts_sync.py` after any template edit.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


TASK_NAME = "continuity_2d_maze"

IMAGE_PLACEHOLDER = "{images}"

# Shared visual-definition blurb. Both templates inline this verbatim so
# the model receives identical wall semantics regardless of question type.
_WALL_DEFINITION = (
    "Walls in the maze come in three visual styles, all of which block "
    "movement equally: (a) full edge walls — solid lines along an entire "
    "cell edge; (b) partial edge walls — solid line segments shorter than "
    "a full edge; (c) diagonal walls — solid lines cutting across a cell "
    "along its diagonal. Any solid line, regardless of length or "
    "orientation, is a wall and cannot be crossed. Each labeled circle "
    "marks one cell."
)

REACHABILITY_SET_TEMPLATE = (
    "[Task]\n"
    "You are looking at a top-down view of a 2D maze. Several cells "
    "are marked with labeled circles (A, B, C, ...). Your job is to "
    "decide, for the target point, which of the OTHER labeled points "
    "you can reach from it through the open maze corridors. A point Y "
    "is reachable from X exactly when you can travel from X to Y "
    "without crossing any wall. You can only step from one cell to a "
    "neighboring cell across an open edge — you cannot cut diagonally "
    "through a wall corner.\n"
    f"Wall types: {_WALL_DEFINITION}\n"
    "The visual evidence for this question is provided below.\n"
    "[Image 1]\nAttached image.\n"
    "\n"
    "[Rules]\n"
    "1. Use only the images and text provided in this prompt.\n"
    "2. If answer options are provided, choose only from the provided options.\n"
    "3. Do not output explanation beyond the required final answer.\n"
    "\n"
    "[Question]\n"
    "{question}\n"
    "\n"
    "[Answer Format]\n"
    "Output exactly one JSON object using this schema: {{\"answer\": {{ans}}}} and nothing else.\n"
    "Replace {{ans}} with the actual answer: a JSON array of point "
    "names, picked from the other labeled points shown in the image and "
    "NOT including the target itself, listed in alphabetical order. Use "
    "[] if no other point is reachable.\n"
)

BAR_REMOVAL_TEMPLATE = (
    "[Task]\n"
    "You are looking at a top-down view of a 2D maze. Two cells are "
    "marked with labeled circles A and B; in the current maze they are "
    "blocked from each other. Some of the maze's internal walls have been "
    "recolored as colored bars (purple / red / green / blue / yellow / "
    "orange). Suppose you are allowed to remove EXACTLY ONE bar — which "
    "colors of bar, when removed alone, would let A and B reach each "
    "other? Evaluate each bar independently: imagine removing only that "
    "one bar, leave every other bar in place, and check whether a path "
    "from A to B opens up. List EVERY color that works.\n"
    f"Wall types: {_WALL_DEFINITION} A bar is a colored (non-black) "
    "wall; black walls are NOT bars.\n"
    "Scoring: your answer is correct ONLY if you list every color whose "
    "single-bar removal reconnects A and B — no missing colors and no "
    "extras. A partial list, an extra color, or an empty answer when at "
    "least one qualifying bar exists, all count as wrong.\n"
    "The visual evidence for this question is provided below.\n"
    "[Image 1]\nAttached image.\n"
    "\n"
    "[Rules]\n"
    "1. Use only the images and text provided in this prompt.\n"
    "2. If answer options are provided, choose only from the provided options.\n"
    "3. Do not output explanation beyond the required final answer.\n"
    "\n"
    "[Question]\n"
    "{question}\n"
    "\n"
    "[Answer Format]\n"
    "Output exactly one JSON object using this schema: {{\"answer\": {{ans}}}} and nothing else.\n"
    "Replace {{ans}} with the actual answer: a JSON array of "
    "color names from the bars shown in the image, in alphabetical order. "
    "Use [] only if NO single-bar removal connects them.\n"
)


def _format_image_block(images: Sequence[str]) -> str:
    if not images:
        return "[No images provided]"
    lines = []
    for image_path in images:
        lines.append("[Image]")
        lines.append(image_path)
    return "\n".join(lines)


def build_reachability_set_question(target: str, point_names: Sequence[str]) -> str:
    """Render the Q1 short question text. `point_names` is the full list of
    labeled points in the scene (includes the target itself); used to spell
    out the legal answer space inside the [Question] section."""
    others = sorted(n for n in point_names if n != target)
    others_str = ", ".join(others) if others else "(none)"
    return (
        f"Which of the other labeled points can you reach from point "
        f"{target}? The other labeled points in this maze are: {others_str}."
    )


def build_bar_removal_question(
    target_pair: Sequence[str],
    bar_colors: Sequence[str],
) -> str:
    """Render the human-readable question string for one bar_removal sample."""
    if len(target_pair) != 2:
        raise ValueError(
            f"target_pair must have exactly 2 names, got {list(target_pair)}"
        )
    a, b = target_pair
    colors_str = ", ".join(sorted(set(bar_colors)))
    return (
        f"Identify EVERY color of bar that, if removed alone, reconnects "
        f"point {a} and point {b} (which are currently blocked from each "
        f"other). List all qualifying colors — missing any one of them "
        f"counts as wrong. The bars in this maze are colored: {colors_str}."
    )


def formulate_reachability_set_prompt(question: str) -> str:
    """Substitute `question` into the Q1 template and return the full
    formulated prompt with `{images}` LEFT LITERAL. The downstream inference
    framework is responsible for replacing `{images}` with image refs at
    send time."""
    return REACHABILITY_SET_TEMPLATE.format(question=question)


def formulate_bar_removal_prompt(question: str) -> str:
    """Substitute `question` into the Q2 template and return the full
    formulated prompt with `{images}` LEFT LITERAL."""
    return BAR_REMOVAL_TEMPLATE.format(question=question)


def build_reachability_set_prompt(
    question: str,
    images: Optional[Sequence[str]] = None,
) -> str:
    """Render the Q1 prompt as a single string with image refs already
    inlined at the {images} site. Used by run_benchmark.py at inference
    time when the framework wants a fully-resolved text prompt."""
    body = REACHABILITY_SET_TEMPLATE.format(question=question)
    return body.replace(IMAGE_PLACEHOLDER, _format_image_block(list(images or [])))


def split_reachability_set_prompt_at_images(question: str) -> Tuple[str, str]:
    """Return (before_images, after_images) halves of the rendered Q1 prompt."""
    body = REACHABILITY_SET_TEMPLATE.format(question=question)
    if IMAGE_PLACEHOLDER not in body:
        raise ValueError("REACHABILITY_SET_TEMPLATE is missing the {images} placeholder.")
    before, after = body.split(IMAGE_PLACEHOLDER, 1)
    return before.rstrip("\n"), after.lstrip("\n")


def build_bar_removal_prompt(
    question: str,
    images: Optional[Sequence[str]] = None,
) -> str:
    body = BAR_REMOVAL_TEMPLATE.format(question=question)
    return body.replace(IMAGE_PLACEHOLDER, _format_image_block(list(images or [])))


def split_bar_removal_prompt_at_images(question: str) -> Tuple[str, str]:
    body = BAR_REMOVAL_TEMPLATE.format(question=question)
    if IMAGE_PLACEHOLDER not in body:
        raise ValueError("BAR_REMOVAL_TEMPLATE is missing the {images} placeholder.")
    before, after = body.split(IMAGE_PLACEHOLDER, 1)
    return before.rstrip("\n"), after.lstrip("\n")
