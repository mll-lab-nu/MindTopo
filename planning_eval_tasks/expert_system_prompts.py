"""Expert-only planning prompts; ordinary difficulty prompts remain unchanged."""

SWAP_2D_EXPERT_SYSTEM_PROMPT = """[Task]
You are solving a 2D grid swapping puzzle. Swap one colored block with the blank cell per turn until the current grid exactly matches the goal grid. Rows increase top-to-bottom and columns left-to-right; both are zero-based.

[Rules]
1. Choose exactly one non-empty current-grid cell in the same row or column as the blank.
2. Adjacency is not required. Swap only the selected block and the blank; all other blocks remain in place.
3. If legal actions are listed, choose one of them; otherwise infer a non-empty cell in the blank's row or column from the current grid.
4. Illegal actions leave the state unchanged but still consume a step.

[Answer Format]
Output exactly one JSON object and nothing else: {"answer":{"row":R,"col":C}}."""


_PIPE_CHANNEL_RULES = {
    "overpass_v1": (
        "Every + shaped piece is an overpass, not a junction: north connects only to south, "
        "and east connects only to west. The raised pipe passes over the other pipe without joining it. "
        "Rotating an overpass changes its appearance but not these connections."
    ),
    "split_channels_v2": (
        "Four-port pieces contain two separate channels, never a four-way junction. "
        "A raised overpass connects north to south and east to west without joining at the crossing. "
        "A double-bend piece shows two separate 90-degree curves with a visible gap: follow each curve "
        "to determine whether it connects north to east and south to west, or north to west and south to east. "
        "One clockwise quarter-turn switches a double bend between these two pairings; a half-turn "
        "restores its pairing. Rotating an overpass does not change its connections."
    ),
}


def get_expert_system_prompt(
    key: str, *, difficulty: str, pipe_rule: str | None = None
) -> str | None:
    """Return an Expert override, or None to keep the original system prompt."""
    if difficulty != "expert":
        return None
    if key == "order_swap_2d_puzzle_planning":
        return SWAP_2D_EXPERT_SYSTEM_PROMPT
    if key != "continuity_pipe_planning" or pipe_rule not in _PIPE_CHANNEL_RULES:
        # Classic Pipe Expert data has no split channels; retain its original rules.
        return None
    return (
        "[Task]\n"
        "You are solving Continuity Pipe. Rotate non-empty pipe cells until every pipe channel "
        "is connected to the green source. Grid coordinates are zero-based (x, y).\n\n"
        "[Rules]\n"
        "1. Choose exactly one non-empty pipe cell per turn.\n"
        "2. Each action rotates one whole cell 90 degrees clockwise, including both channels "
        "of a four-port piece.\n"
        f"3. {_PIPE_CHANNEL_RULES[pipe_rule]}\n"
        "4. Each channel turns green independently when connected to the source; a four-port "
        "piece may have one green channel and one blue channel.\n"
        "5. If legal actions are listed, choose one of them; otherwise infer a legal non-empty "
        "cell from the current state.\n"
        "6. Both channels of every four-port piece and every ordinary pipe must connect to "
        "the source to win.\n\n"
        "[Answer Format]\n"
        'Output exactly one JSON object and nothing else: {"answer":{"x":X,"y":Y}}.'
    )
