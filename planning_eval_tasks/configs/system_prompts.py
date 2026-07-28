def _planning_prompt(*, task: str, rules: list[str], answer_format: str) -> str:
    rule_lines = "\n".join(f"{index}. {rule}" for index, rule in enumerate(rules, start=1))
    return (
        f"[Task]\n{task}\n\n"
        f"[Rules]\n{rule_lines}\n\n"
        f"[Answer Format]\n{answer_format}"
    )


SYSTEM_PROMPTS = {
    "none": "",
    "knots_untangle_planning": _planning_prompt(
        task=(
            "You are solving Knots Untangle. Move one rope endpoint at a time on the indexed "
            "pegboard until no two rope paths cross. Coordinates use zero-based (row, col)."
        ),
        rules=[
            "Choose exactly one move per turn. If legal actions are listed, choose one of them; otherwise infer a legal move from the current state.",
            "Move one endpoint from its occupied source hole to an empty target hole.",
            "Illegal actions leave the state unchanged but still consume a step.",
            "The task succeeds when zero crossings remain.",
        ],
        answer_format=(
            'Output exactly one JSON object and nothing else: '
            '{"answer":{"src_row":R,"src_col":C,"tgt_row":R,"tgt_col":C}}.'
        ),
    ),
    "continuity_pipe_planning": _planning_prompt(
        task=(
            "You are solving Continuity Pipe. Rotate non-empty pipe cells until every pipe is "
            "connected to the green source. Grid coordinates are zero-based (x, y)."
        ),
        rules=[
            "Choose exactly one non-empty pipe cell per turn.",
            "The selected pipe rotates clockwise by 90 degrees.",
            "If legal actions are listed, choose one of them; otherwise infer a legal non-empty cell from the current state.",
            "The task succeeds when every pipe is connected to the source.",
        ],
        answer_format='Output exactly one JSON object and nothing else: {"answer":{"x":X,"y":Y}}.',
    ),
    "separation_one_stroke_planning": _planning_prompt(
        task=(
            "You are solving One-Stroke Color Grouping. Extend one continuous stroke from the "
            "bottom-left green start peg to the top-right red goal peg so same-colored cells "
            "share a region and different colors are separated. The white line is the current path."
        ),
        rules=[
            "Choose exactly one move from U, D, L, or R to extend the path by one edge.",
            "Do not reuse an edge or create a closed loop. Backtracking over the most recent edge is allowed and undoes that move.",
            "If legal actions are listed, choose one of them; otherwise infer a legal direction from the current state.",
            "Illegal actions leave the state unchanged but still consume a step.",
            "The task succeeds when the path reaches the goal and the final regions satisfy the color-separation rule.",
        ],
        answer_format='Output exactly one JSON object and nothing else: {"answer":"D"}, replacing D with U, D, L, or R.',
    ),
    "order_swap_2d_puzzle_planning": _planning_prompt(
        task=(
            "You are solving a 2D grid swapping puzzle. Swap one colored block with the blank "
            "cell per turn until the current grid exactly matches the goal grid. Rows increase "
            "top-to-bottom and columns left-to-right; both are zero-based."
        ),
        rules=[
            "Choose exactly one non-empty current-grid cell per turn.",
            "The selected block swaps with the blank cell.",
            "If legal actions are listed, choose one of them; otherwise infer a legal non-empty cell from the current grid.",
            "Illegal actions leave the state unchanged but still consume a step.",
        ],
        answer_format='Output exactly one JSON object and nothing else: {"answer":{"row":R,"col":C}}.',
    ),
    "enclosure_chat_noir_planning": _planning_prompt(
        task=(
            "You are solving Chat Noir / Encircle the Cat. Block one indexed hex cell per turn "
            "to trap the cat before it reaches the boundary. Cell indices are printed on the board."
        ),
        rules=[
            "Choose exactly one open non-cat cell to block.",
            "After the block, the cat may stay still or move one hex step according to its policy.",
            "If legal actions are listed, choose one of them; otherwise infer a legal open non-cat cell from the current state.",
            "Illegal actions leave the state unchanged but still consume a step.",
            "The task succeeds when the cat has no path to a boundary and fails if it reaches one.",
        ],
        answer_format='Output exactly one JSON object and nothing else: {"answer":CELL_INDEX}.',
    ),
}


def get_system_prompt(key: str) -> str:
    if key not in SYSTEM_PROMPTS:
        known = ", ".join(sorted(SYSTEM_PROMPTS))
        raise KeyError(f"Unknown system prompt key {key!r}. Known keys: {known}")
    return SYSTEM_PROMPTS[key]
