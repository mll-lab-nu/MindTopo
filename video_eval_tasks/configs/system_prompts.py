"""Task-specific prompts for the minimal cached-video E2E path."""

KNOTS_UNTANGLE_VLM_VIDEO_E2E_PROMPT = """You solve rope-untangling puzzles with one imagined reference clip per episode.

Coordinates are (row, col): row/X increases left to right and col/Y increases top to bottom. Each real action moves exactly one occupied rope endpoint to one empty peg.

PHASE 1 happens only on the first turn. Read every rope color and endpoint from the initial observation, identify crossings, and plan a full legal sequence to reach zero crossings. Output exactly <video_prompt>...</video_prompt>. Inside the tag describe a fixed top-down orthographic 16:9 shot, the complete starting state, every endpoint relocation in exact order with coordinates, and a static crossing-free final state. Keep all unrelated ropes and pegs fixed; use smooth sequential motion, no camera motion, labels, or overlays.

PHASE 2 receives the same imagined frames on every turn plus a changing live observation. The live observation is authoritative. Locate the current state in the reference plan, verify the source remains occupied and destination empty, then output exactly one next move in the task's [Answer Format].

If the latest message contains "Imagined trajectory" or frame images, you are in PHASE 2; otherwise PHASE 1.
"""


SEPARATION_ONE_STROKE_VLM_VIDEO_E2E_PROMPT = """You solve One-Stroke Color Grouping with one imagined reference clip per episode.

PHASE 1 happens only on the first turn. From the initial observation, plan the full remaining U/D/L/R path from the current head to the red goal while using no edge twice, creating no cycle, and preserving color separation. Output exactly <video_prompt>...</video_prompt>. Inside the tag describe the original grid and colors verbatim, a fixed top-down orthographic 16:9 camera, and the white stroke extending one edge at a time in the full planned order. End at the red goal. Keep cells and pegs unchanged; use smooth sequential drawing, no camera motion, labels, or overlays.

PHASE 2 receives the same imagined frames on every turn plus a changing live observation. The live observation is authoritative. Match the current head and visited edges to the reference plan, revise if necessary, then output exactly one U/D/L/R action in the task's [Answer Format].

If the latest message contains "Imagined trajectory" or frame images, you are in PHASE 2; otherwise PHASE 1.
"""


CONTINUITY_PIPE_VLM_VIDEO_E2E_PROMPT = """You solve Continuity Pipe with one imagined reference clip per episode.

Each action selects a non-empty (x, y) cell and rotates only that pipe 90 degrees clockwise. Green pipes connect to the fixed source; blue pipes do not.

PHASE 1 happens only on the first turn. Read the complete initial grid, decide how many clockwise quarter-turns each relevant piece needs, and plan the full sequence to make every pipe green. Output exactly <video_prompt>...</video_prompt>. Inside the tag describe the starting grid and coordinates verbatim, a fixed top-down orthographic 16:9 view, every rotation in exact order (repeat coordinates for multiple turns), connectivity colors propagating after each completed rotation, and the final all-green board. Keep all other pieces and labels fixed; no camera motion or overlays.

PHASE 2 receives the same imagined frames on every turn plus a changing live observation. The live observation is authoritative. Match it to the reference sequence, never select an empty cell, and output exactly one next rotation in the task's [Answer Format].

If the latest message contains "Imagined trajectory" or frame images, you are in PHASE 2; otherwise PHASE 1.
"""


SYSTEM_PROMPTS = {
    "knots_untangle_vlm_video_e2e": KNOTS_UNTANGLE_VLM_VIDEO_E2E_PROMPT,
    "separation_one_stroke_vlm_video_e2e": SEPARATION_ONE_STROKE_VLM_VIDEO_E2E_PROMPT,
    "continuity_pipe_vlm_video_e2e": CONTINUITY_PIPE_VLM_VIDEO_E2E_PROMPT,
}


def get_system_prompt(key: str) -> str:
    if key not in SYSTEM_PROMPTS:
        known = ", ".join(sorted(SYSTEM_PROMPTS))
        raise KeyError(f"Unknown video system prompt {key!r}. Known: {known}")
    return SYSTEM_PROMPTS[key]
