"""Canonical question phrasings for the knots_static benchmark.

Each task carries its full prompt spec (description, definition, answer
options, answer space, parse key, phrasings) in a single entry of
`TASK_SPECS`. `build_prompt` renders the meta_prompt/reasoning.txt layout:
[Task] / [Rules] / [Question] / [Answer Format].
"""

from __future__ import annotations


TASK_SPECS: dict[str, dict] = {
    "T01_structure_classification": {
        "task_name": "knot.structure_classification",
        "task_category": "knots",
        "detailed_description": (
            "look at all the ropes in the image and decide which of the six categories below the whole scene fits into"
        ),
        "definition": (
            "\n"
            "A) Simple closed ring\n"
            "  - exactly ONE rope, its two ends are joined into a closed ring, AND it has NO knot tied in it.\n"
            "  - If you smoothed it out it would just be a circle with no self-overlap or crossings.\n"
            "\n"
            "B) Knot\n"
            "  - exactly ONE rope tied into a knot. It may be a closed knotted ring, OR an open-ended knotted rope whose two ends are tied to sticks.\n"
            "  - Sticks may be present for open-ended knots, but are not required for this category. If sticks are shown, they are FIXED and IMMOVABLE, so the rope endpoints cannot be moved.\n"
            "  - The rope cannot be simplified into category A or C without cutting, or without moving fixed endpoints when sticks are present.\n"
            "\n"
            "C) Open-ended rope with no knot\n"
            "  - exactly ONE open-ended rope, its two ends are tied to fixed sticks. The sticks are FIXED and IMMOVABLE in space, so the rope endpoints cannot be moved.\n"
            "  - With the endpoint constraints, the rope can still be pulled into a simple open arc without cutting.\n"
            "\n"
            "D) Link\n"
            "  - TWO OR MORE closed-ring ropes, AND every single rope is linked to at least one other rope.\n"
            "  - No rope is completely free, every rope passes through some other rope and cannot be pulled away on its own.\n"
            "\n"
            "E) Unlinked multiple ropes\n"
            "  - TWO OR MORE separate ropes; each individual rope may be open-ended or closed-ended, and may itself be tied into a knot.\n"
            "  - NONE of the ropes is linked to any OTHER rope. You could pick up each whole rope and carry it away from all the others without cutting anything.\n"
            "\n"
            "F) Others\n"
            "  - the picture combines rope structures of two or more DIFFERENT categories from A-E in the same scene.\n"
            "  - Key test: if you cannot describe the WHOLE picture with a single A/B/C/D/E label and would need more category labels to cover everything, the answer is F."
        ),
        "answer_options": (
            "A) Simple closed ring\n"
            "B) Knot\n"
            "C) Open-ended rope with no knot\n"
            "D) Link\n"
            "E) Unlinked multiple ropes\n"
            "F) Others"
        ),
        "legal_answer_space_description": (
            'one of the JSON strings "A", "B", "C", "D", "E", or "F"'
        ),
        "parse_key": "OPTION",
        "legal_values": ["A", "B", "C", "D", "E", "F"],
        "phrasings": [
            "Using the six categories above, decide which one the entire scene in this image fits into.",
        ],
    },
    "T02_component_count": {
        "task_name": "knot.component_count", #rope number
        "task_category": "knots",
        "detailed_description": (
            "count how many separate ropes are in the picture"
        ),
        "definition": (
            "A 'rope' is a single continuous strand. \n"
            "Important: \n"
            "- If two ropes are tied together, tangled up, or just lying on top of each other in the picture, they still count as TWO. "
            "We are counting physical strands, NOT how many groups would fall apart if you shook them.\n"
            "- A single rope that crosses over itself is still ONE, not several ones.\n"
            "- Sticks are FIXED and IMMOVABLE anchors in space; they are NOT ropes and must NOT be counted.\n"
        ),
        "answer_options": "",
        "legal_answer_space_description": (
            "a single non-negative JSON integer giving the number of ropes, "
            "for example 3"
        ),
        "parse_key": "INTEGER",
        "legal_values": None,
        "phrasings": [
            "How many separate ropes are there in this image?",
        ],
    },
    "T03_link_topology": {
        "task_name": "knot.link_topology",
        "task_category": "knots",
        "detailed_description": (
            "look at how the rings in the image are caught onto each other and pick which pattern matches the whole picture"
        ),
        "definition": (
            "\n"
            "All ropes here are closed rings, like rings on a key chain. Two rings are 'linked' when they go through each other so they cannot be pulled apart without cutting one open.\n"
            "Rings that just touch, sit next to each other, or look tangled but would slide apart with enough wiggling are NOT linked.\n"
            "\n"
            "A) Not linked\n"
            "  - no two rings go through each other.\n"
            "  - Every ring could be lifted away from all the others without cutting.\n"
            "\n"
            "B) Chain\n"
            "  - TWO OR MORE rings are hooked together one after another, like a metal chain.\n"
            "  - A two-ring paired link counts as a chain with two rings.\n"
            "  - Each ring is caught only on its direct neighbour(s) in its chain group, not on rings farther down the chain.\n"
            "  - If the picture contains multiple separate chain groups and no free rings or all-interlocked groups, the answer is still B.\n"
            "\n"
            "C) All interlocked\n"
            "  - exactly THREE rings are woven together as a group.\n"
            "  - Any TWO rings by themselves would NOT be caught on each other, but the three together cannot be pulled apart unless one is cut.\n"
            "\n"
            "D) Mixed\n"
            "  - no single A/B/C label describes the WHOLE picture.\n"
            "  - This also includes scenes where some rings are caught together while other rings are entirely free, such as a chain plus loose rings or an all-interlocked trio plus a free ring.\n"
            "  - This also includes scenes that combine chain-style links with all-interlocked groups.\n"
            "  - Key test: if you cannot describe the WHOLE picture with a single A/B/C label and would need more pattern labels to cover everything, the answer is D."
        ),
        "answer_options": (
            "A) Not linked\n"
            "B) Chain\n"
            "C) All interlocked\n"
            "D) Mixed"
        ),
        "legal_answer_space_description": (
            'one of the JSON strings "A", "B", "C", or "D"'
        ),
        "parse_key": "OPTION",
        "legal_values": ["A", "B", "C", "D"],
        "phrasings": [
            "Using the four patterns above, decide which one the entire scene in this image fits into.",
        ],
    },
    "T04_link_property": {
        "task_name": "knot.link_property",
        "task_category": "knots",
        "detailed_description": (
            "imagine cutting and removing the ring of a specified color, then list the colors of every remaining ring that is now FREE (not linked to any other remaining ring)"
        ),
        "definition": (
            "\n"
            "Setup:\n"
            "  - Each ring in this picture has a distinct solid color, drawn from the palette: red, blue, green, brown, white, purple.\n"
            "  - You will be told the color of one specific ring. Imagine you cut that ring open and pull it out of the picture.\n"
            "  - Look at all the rings that REMAIN.\n"
            "\n"
            "Definitions:\n"
            "  - 'Linked' = two rings pass through each other and cannot be pulled apart no matter how much you wiggle them."
            " Rings that just touch, sit next to each other, or look tangled but would slide apart with enough wiggling are NOT linked.\n"
            "  - A remaining ring is FREE if it is not linked to ANY other remaining ring; you could pick it up and carry it away from all the other remaining rings without cutting anything.\n"
            "  - A remaining ring is NOT free if it is still linked to at least one other remaining ring.\n"
            "\n"
            "What to output:\n"
            "  - List the colors of every remaining ring that is now FREE.\n"
            "  - Use only the color names from the palette: red, blue, green, brown, white, purple.\n"
            "  - The colors may be listed in any order.\n"
            "  - Do NOT include the color of the ring that was removed.\n"
            "  - If NO remaining ring is free (every remaining ring is still linked to at least one other), output exactly [\"none\"]. Do NOT output an empty list.\n"
        ),
        "answer_options": "",
        "legal_answer_space_description": (
            'a JSON list of one or more strings drawn from '
            '{"red", "blue", "green", "brown", "white", "purple", "none"}; '
            'list every remaining ring that is now free; colors may appear in any order; '
            'for example ["red", "blue"]; if NO remaining ring is free output exactly ["none"] '
            '(never an empty list)'
        ),
        "parse_key": "COLOR_LIST",
        "legal_values": [
            "red",
            "blue",
            "green",
            "brown",
            "white",
            "purple",
            "none",
        ],
        "phrasings": [
            "Imagine you cut and remove the {removed_color} ring from this configuration. "
            "After removing it, which of the remaining rings are now FREE (not linked to any other remaining ring)? "
            "List the colors of all rings that are now free.",
        ],
    },
    "T05_linked_count": {
        "task_name": "knot.linked_count",
        "task_category": "knots",
        "detailed_description": (
            "go through every rope in the image and count how many of them are linked to at least one OTHER rope"
        ),
        "definition": (
            "A rope is one continuous strand or ring. Two ropes that touch, overlap, or are tangled in the image are still two separate ropes.\n"
            "Now decide which ropes are linked. A rope counts as linked if it passes through some OTHER rope so that the two cannot be separated without cutting one of them.\n"
            "Important rules:\n"
            "- A knot tied within one rope does NOT count as linked to another rope.\n"
            "- Two ropes that just touch, overlap in the picture, or are tangled but would slide apart are NOT linked.\n"
            "- Count each rope at most once, even if it is linked to several other ropes.\n"
            "- If no rope is linked to another rope, the answer is 0."
        ),
        "answer_options": "",
        "legal_answer_space_description": (
            "a single non-negative JSON integer giving how many ropes are linked to at least one OTHER rope, for example 2"
        ),
        "parse_key": "INTEGER",
        "legal_values": None,
        "phrasings": [
            "How many ropes in this image are linked to at least one OTHER rope?",
        ],
    },
}

PHRASINGS: dict[str, list[str]] = {
    task_id: spec["phrasings"] for task_id, spec in TASK_SPECS.items()
}


def _image_block(count: int = 1) -> str:
    return "\n".join(
        f"[Image {i}]\nAttached image." for i in range(1, count + 1)
    )


def build_prompt(
    task_id: str,
    phrasing_index: int = 0,
    **format_vars: str,
) -> str:
    """Render the meta-format prompt for *task_id* and phrasing.

    Any keyword arguments are substituted into ``{placeholder}`` fields in the
    phrasing string. Phrasings without placeholders ignore ``format_vars``.
    """
    spec = TASK_SPECS[task_id]
    question = spec["phrasings"][phrasing_index]
    if format_vars:
        question = question.format(**format_vars)
    if spec["answer_options"]:
        question = f"{question}\n\nAnswer options:\n{spec['answer_options']}"
    return "\n".join([
        "[Task]",
        f"You are solving the {spec['task_category']} task.",
        f"In this task, you must {spec['detailed_description']}.",
        "If this task depends on a specific visual definition, use this "
        f"definition exactly: {spec['definition']}",
        "The visual evidence for this question is provided below.",
        _image_block(1),
        "",
        "[Rules]",
        "1. Use only the images and text provided in this prompt.",
        "2. If answer options are provided, choose only from the provided options.",
        "3. Do not output explanation beyond the required final answer.",
        "",
        "[Question]",
        question,
        "",
        "[Answer Format]",
        'Output exactly one JSON object: {"answer": {value}} and nothing else.',
        "Replace {value} with the single legal answer for this task, "
        f"chosen from {spec['legal_answer_space_description']}.",
    ])


def build_default_prompt(task_id: str) -> str:
    return build_prompt(task_id, 0)


def num_phrasings(task_id: str) -> int:
    spec = TASK_SPECS.get(task_id)
    return len(spec["phrasings"]) if spec else 0
