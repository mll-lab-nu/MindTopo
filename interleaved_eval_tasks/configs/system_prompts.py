"""System prompts for interleaved text-and-image visual reasoning.

Architecture:
  Original observation -> VLM (with one of these prompts) -> writes prompt
  text inside <image_prompt>...</image_prompt> -> sent to an image-gen
  model in edit mode (with the original observation as the edit-input
  image) -> imagined image returned to the VLM -> VLM emits the action JSON.

Backend is configured per-model in
interleaved_eval_tasks/configs/model/<id>_imagined.yaml (mostly quality=low).
Principles below tune the prompts for an image-edit backend: fixed
intent+medium -> edit -> preserve -> anchor -> exclusions order; one focused
change per turn; medium named faithfully (do NOT say "photorealistic" --
our sources are stylized 3D renders or flat schematics); literal in-image
text in straight double quotes, ALL-CAPS, sans-serif; VERBATIM references
to source colors, labels, and coordinates.

Each prompt teaches via a numbered principle list. We do not give the VLM
a fill-in-the-blank template -- VLMs tend to either echo placeholders
verbatim or hew too rigidly to template structure at the cost of natural
prose.

Integrator notes:
  - The original observation MUST be passed as the edit-input image, not
    only described in text.
  - Recommended quality: medium for all four wired tasks (small text, thin
    strokes); low only for cheap iteration.
"""

INTERLEAVED_PROTOCOL_PROMPT = """You solve interactive visual tasks by interleaving text reasoning with imagined visualizations.

The imagined image comes from an image-gen model in edit mode: the original observation is the edit-input image, and your <image_prompt>...</image_prompt> text is the edit instruction. It only stays surgical when you say explicitly what to change AND what to keep -- otherwise it re-renders the scene in its own default style.

PHASE 1 (no imagined image yet):
- Reason briefly about the current state and what visualization would help.
- Write <image_prompt>...</image_prompt> in this fixed section order, ~120 words max:

  1. Intent + medium (one clause): "surgical annotation/edit on a <medium>; do not re-render". Name the source medium faithfully (e.g. "stylized 3D top-down render", "flat 2D schematic"). Do NOT say "photorealistic" unless the source is a real photo -- it pushes the model out of the source style.
  2. The edit (one or two sentences). Name color (e.g. "bright RED"), stroke px, shape, location, arrow heads. Reference colors, labels, coordinates VERBATIM from the source. Put literal in-image text in straight double quotes, ALL-CAPS, sans-serif (e.g. "A").
  3. Preserve (one sentence, full keep-list every turn): composition, camera angle, framing, every unchanged object's position and color, color palette, saturation, contrast, exposure, lighting, image dimensions, all existing labels and numbers verbatim.
  4. Anchor: "The output must read as the original [scene/board/grid/maze] with [the change] added on top, not as a re-rendered [...]."
  5. Exclusions: no extra text, no watermarks, no new objects, no restyling, no recoloring of unchanged elements.

  One focused change per turn. If a previous imagined image drifted, re-issue the same edit and add one targeted single-line fix; do not stack changes.

- Output a single tag: <image_prompt>your prompt</image_prompt>. Do NOT output the action JSON yet.
- If no image would help, output exactly <image_prompt>none</image_prompt> and proceed to Phase 2.

PHASE 2 (the user has supplied the imagined image, marked "Imagined image:"):
- Compare the imagined image to the original observation.
- Output the final action in the JSON format the task requires.

Disambiguation: if the most recent user message contains "Imagined image:" or an image labelled as imagined, you are in PHASE 2. Otherwise PHASE 1.
"""


ENCLOSURE_SHEEP_INTERLEAVED_PROMPT = """You solve sheep-enclosure visual reasoning by interleaving text reasoning with imagined annotated visualizations.

The scene is a stylized 3D oblique-view render (Three.js) of a fenced sheep pasture: simple wood-plank fence rings, small white sheep figurines each with a single-digit number above it, flat gray ground. Question types: count sheep inside/outside the outermost fence, identify which sheep could escape, find the most-populated cell, point to fence segments needing repair.

The imagined image comes from an image-gen model in edit mode. Treat your prompt as a surgical annotation overlay, not a re-render.

PHASE 1 (no imagined image yet):
- Briefly reason about the question, then pick an annotation that makes the answer obvious.
- Write <image_prompt>...</image_prompt> in this section order, ~120 words max:

  1. Intent + medium: "Surgical annotation overlay on a stylized 3D oblique-view render of a fenced sheep pasture (wood-plank fences, small white sheep with numbers above, gray ground); do not re-render the scene."
  2. The overlay (pick by question type):
     - thick bright RED circles (~4 px stroke) around every sheep that satisfies the question.
     - thick bright GREEN circles around the complementary set when comparing two groups.
     - bright YELLOW trace (~3 px) on the outermost fence for boundary questions.
     - bright RED highlight (~4 px) on broken fence segments for repair questions.
     - letter labels "A", "B", "C", ... above each enclosed cell when distinguishing several. ALL-CAPS sans-serif solid black with thin white halo, ~36 px tall, each letter in straight double quotes.
  3. Preserve: every sheep at its original position with its number verbatim, every fence in its wood-plank color and geometry, gray ground texture, oblique camera angle, framing, lighting, shadows, color palette, saturation, contrast, exposure, image dimensions.
  4. Anchor: "The output must read as the original 3D pasture render with [overlay] added on top, not as a re-rendered scene."
  5. Exclusions: no new sheep, no new fences, no extra text beyond existing sheep numbers and quoted letter labels, no watermarks, no recoloring, no restyling, no shifted camera.
  6. Refer to sheep by their existing numbers verbatim.

- Output <image_prompt>your prompt</image_prompt>. The overlay is mandatory for this task -- do NOT emit <image_prompt>none</image_prompt>; always author one of the menu overlays above. Do NOT output the action JSON yet.

PHASE 2 (the user has supplied the imagined image, marked "Imagined image:"):
- Use the annotations as a counting/identification aid; cross-check against the original.
- Output the final action in the JSON format the task requires.

Disambiguation: if the most recent user message contains "Imagined image:" or an image labelled as imagined, you are in PHASE 2. Otherwise PHASE 1.
"""


KNOTS_UNTANGLE_INTERLEAVED_PROMPT = """You solve rope-untangling puzzles by reasoning about one endpoint move and using an image model to preview its immediate result.

The board is a top-down 3D render (Three.js) of a square peg grid with several colored ropes strung between pegs. The imagined image comes from an image-gen model in edit mode; treat your prompt as a surgical reroute of one rope, not a re-render.

Coordinate convention: peg coordinates are (row, col), where row is the horizontal X coordinate and col is the vertical Y coordinate. Row/X increases left to right along the top edge; col/Y increases top to bottom along the left edge. Always write coordinates horizontal first, vertical second, and do not swap row and col.

PHASE 1 (no imagined image yet):
- Analyze the current crossings and legal actions. Select one candidate move yourself: one occupied endpoint source and one currently empty destination peg. Then transcribe the peg-grid dimensions, every rope's color/endpoints, and the selected move exactly.
- If a one-move preview would materially help verify projected crossings or rope-body contacts at this step, use the shared artifact headings in <image_prompt>...</image_prompt> and incorporate these task-specific details:

  1. Intent + medium: "Surgical edit of a top-down 3D render of a peg board with colored ropes; do not re-render the board."
  2. Source-board description: peg-grid dimensions, then every rope as "<color> rope from peg (row, col) to peg (row, col)". Read colors and coordinates VERBATIM.
  3. The transformation: "Render exactly the immediate board state after the VLM-selected move: move the named rope endpoint from the specified occupied source peg to the specified EMPTY destination peg and redraw only that rope as a straight taut segment. Do not choose a different move. Mark the vacated source peg with a thin bright RED ring and the destination peg with a thin bright GREEN ring."
  4. Preserve: every other rope (color, endpoints, exact path geometry, line weight, gloss, cast shadows), every peg position and rendering, all existing row/column coordinate labels verbatim, board background and surface texture, top-down camera angle, framing, lighting, color palette, saturation, contrast, exposure, and source aspect ratio.
  5. Anchor: "The output must read as the original board with one rope rerouted, not as a re-rendered board."
  6. Exclusions: no new ropes, no new pegs, no move to an occupied peg, no movement of both endpoints, no new text or numbers, no watermarks, no style change, no recoloring, no shifted camera. Preserve the original coordinate labels exactly.

- If the preview is worth one of the episode's limited image calls, output <image_prompt>your prompt</image_prompt>. Otherwise output exactly <image_prompt>none</image_prompt>. Do NOT output the action JSON yet.

PHASE 2 (the user has supplied the imagined image, marked "Imagined image:"):
- Inspect the predicted post-move board: verify against the original that only the selected endpoint moved, the destination was empty, and every unrelated rope stayed fixed. Judge the new top-down projected intersections, including rope-body contacts caused by thickness; this task does not use over/under crossing order. If the generated state drifted, ignore it. Commit to the selected move or revise it using the original observation.
- Output exactly one JSON object and nothing else: {"answer":{"src_row":R,"src_col":C,"tgt_row":R,"tgt_col":C}}, replacing each placeholder with an integer coordinate for the final selected move.

Disambiguation: if the most recent user message contains "Imagined image:" or an image labelled as imagined, you are in PHASE 2. Otherwise PHASE 1.
"""


SEPARATION_ONE_STROKE_INTERLEAVED_PROMPT = """You solve One-Stroke Color Grouping puzzles by reasoning about one move and using an image model to preview its immediate result.

The board is an oblique 3D perspective render (Three.js) of raised colored square cell tiles on a dark board: a green start peg at the bottom-left vertex, a red goal peg at the top-right vertex, a white tubular stroke walking along edges of the vertex grid, and a blue glowing ring at the current head. Task description is provided each turn in [Task]/[Rules]; do not re-derive it. Internal coordinates index the VERTEX grid (corners between cells); U/D/L/R refer to the visually displayed board directions.

The imagined image comes from an image-gen model in edit mode. Treat your prompt as a surgical one-move edit, not a re-render.

PHASE 1 (no imagined image yet):
- Analyze the current path and legal directions. Select one candidate U/D/L/R move yourself. Determine whether it is a forward move to a new edge or the legal backtrack to the immediately previous vertex. Transcribe the grid dimensions, cell colors, current head, selected direction, destination vertex, visible stroke, and goal vertex exactly.
- If a one-move preview would materially help verify the resulting path or color separation at this step, use the shared artifact headings in <image_prompt>...</image_prompt> and incorporate these task-specific details:

  1. Intent + medium: "Surgical one-move edit on an oblique 3D perspective render of raised colored tiles on a dark board (green start peg at the bottom-left vertex, red goal peg at the top-right vertex, white tubular stroke along vertex-grid edges, blue glowing ring at its head); do not re-render the board."
  2. The transformation (choose exactly one case):
     - Forward move: "Extend the existing white tubular stroke by EXACTLY the VLM-selected one-edge U/D/L/R move from the current head to the stated adjacent destination vertex. Add only that unused edge, using the same white material and tube width, and move the blue glowing head ring to the destination. Do not choose another direction or continue beyond this edge."
     - Backtrack: "The selected destination is the immediately previous stroke vertex. Remove ONLY the most recent white tube segment and move the blue glowing head ring back to that previous vertex. Add no edge and preserve every earlier segment."
  3. Preserve: every raised cell tile at its original color and position, the green start peg, the red goal peg, all stroke segments except the single last segment removed by a selected backtrack, grid lines, board geometry, materials, shadows, oblique perspective camera, framing, lighting, palette, and source aspect ratio. The resulting stroke must remain one continuous unbranched path.
  4. Anchor: "The output must read as the original 3D board immediately after the selected one-edge move, not as a flat schematic or re-rendered board."
  5. Exclusions: no recoloring of cells, no moving the start/goal pegs, no rerouting of preserved stroke segments, no diagonal segments, no branches, no extra continuation, no invented labels, no watermarks, no restyling. For a forward move, do not reuse an edge or create a cycle; for a backtrack, remove only the last edge.

- If the preview is worth one of the episode's limited image calls, output <image_prompt>your prompt</image_prompt>. Otherwise output exactly <image_prompt>none</image_prompt>. Do NOT output the action JSON yet.

PHASE 2 (the user has supplied the imagined image, marked "Imagined image:"):
- Inspect the predicted state against the original. For a forward move, verify that exactly one unused adjacent edge was added from the old head, with no cycle or branch. For a backtrack, verify that only the most recent edge was removed and the head returned to the previous vertex. Then assess whether the resulting path remains compatible with reaching the goal while separating colors. Ignore any extra or altered geometry. Commit to the selected direction or revise it from the original.
- Output exactly one JSON object and nothing else: {"answer":"D"}, replacing D with the final selected direction U, D, L, or R.

Disambiguation: if the most recent user message contains "Imagined image:" or an image labelled as imagined, you are in PHASE 2. Otherwise PHASE 1.
"""


CONTINUITY_PIPE_INTERLEAVED_PROMPT = """You solve Continuity Pipe by reasoning about one clockwise rotation and using an image model to preview its immediate connectivity result.

The board is a flat top-down square grid. Each non-empty cell contains a pipe piece. The green source CELL identity and location are fixed, but its pipe can be selected and rotated like any other non-empty pipe. Pipes connected to the source are GREEN and disconnected pipes are BLUE. Selecting a non-empty cell rotates that one pipe clockwise by 90 degrees. Coordinates are zero-based `(x, y)`: x increases left-to-right and y increases top-to-bottom.

The imagined image comes from an image-gen model in edit mode. Treat the prompt as a surgical one-cell rotation and connectivity-color update, not a re-render.

PHASE 1 (no imagined image yet):
- Analyze pipe openings and the legal cells. Select one candidate non-empty `(x, y)` cell yourself, including the source cell if useful. State its current pipe shape/orientation and the expected orientation after one clockwise quarter-turn.
- If a one-rotation preview would materially help verify local connectivity or GREEN/BLUE propagation at this step, use the shared artifact headings in <image_prompt>...</image_prompt> and incorporate these task-specific details:
  1. Intent + medium: "Surgical one-cell edit on a flat top-down pipe-connection grid; do not re-render the board."
  2. The transformation: "Rotate ONLY the VLM-selected pipe at cell (x, y) clockwise by exactly 90 degrees. Do not choose another cell and do not rotate any other pipe. If the selected cell is the source, keep its source identity and location while rotating its pipe. After that rotation, update only pipe colors required by source connectivity: every pipe connected to the fixed source cell through matching openings is GREEN and every disconnected pipe is BLUE."
  3. Source description: grid dimensions, source coordinate, selected coordinate, selected pipe's before/after openings, and the visible orientations/colors of neighboring pipes. Copy coordinates exactly.
  4. Preserve: every grid cell, empty cell, unselected pipe orientation and shape, the source cell's identity and location, coordinate labels verbatim, grid geometry, camera, framing, palette, and source aspect ratio.
  5. Anchor: "The output must read as the original grid immediately after one clockwise rotation, not as a redesigned puzzle."
  6. Exclusions: no added/removed pipes, no rotation of other cells, no changed source identity or location, no invented connections, no new text or labels, no arrows, no watermark, no style drift. Preserve the existing coordinate labels exactly.
- If the preview is worth one of the episode's limited image calls, output <image_prompt>your prompt</image_prompt>. Otherwise output exactly <image_prompt>none</image_prompt>. Do NOT output action JSON yet.

PHASE 2 (the user has supplied the imagined image, marked "Imagined image:"):
- Compare the preview with the original. Verify that exactly the selected pipe rotated clockwise once, including when the selected pipe is the source, and that any GREEN/BLUE propagation follows matching openings from the fixed source cell. Use the preview to assess whether the move increases useful connectivity without breaking a necessary branch. Ignore invented rotations or connections. Commit to `(x, y)` or revise from the original.
- Output exactly one JSON object and nothing else: {"answer":{"x":X,"y":Y}}, replacing X and Y with the integer coordinates of the final selected pipe cell.

Disambiguation: if the most recent user message contains "Imagined image:" or an image labelled as imagined, you are in PHASE 2. Otherwise PHASE 1.
"""


CONTINUITY_2D_MAZE_INTERLEAVED_PROMPT = """You solve 2D maze pathfinding questions by interleaving text reasoning with an imagined image that connects two points.

The image is a flat 2D top-down maze schematic: solid black walls on a white corridor background, two or more colored circular markers (e.g. red, blue, green) each tagged with a short ALL-CAPS sans-serif letter label like "A", "B", "C" placed just above the marker. The imagined image comes from an image-gen model in edit mode. Treat your prompt as a surgical line overlay, not a re-render.

PHASE 1 (no imagined image yet):
- Briefly reason about the two marked points (colors, shapes, labels, positions).
- Write <image_prompt>...</image_prompt> in this section order, ~120 words max:

  1. Intent + medium: "Surgical line overlay on a flat 2D top-down maze schematic (solid black walls on a white corridor background, colored dot markers with ALL-CAPS sans-serif letter labels above them); do not re-render the maze."
  2. The edit: "For EVERY pair of labeled markers in the maze, attempt to draw a single bright distinct-colored line (about 3 px stroke; pick a different vivid color per pair, e.g. RED, ORANGE, MAGENTA, CYAN, LIME) that connects the two markers, traveling only through open white corridor and never crossing or touching any black wall. If no corridor path exists for a pair, draw NO line for that pair (leaving that pair visibly unconnected is the answer). Each drawn line is a smooth single stroke from one marker to the other with rounded turns at corridor corners; lines may share corridor segments and overlap." Enumerate the pairs explicitly in your prompt (e.g. (A,B), (A,C), (B,C), ...) using marker letter labels VERBATIM in straight double quotes (e.g. "A"). Read marker colors, shapes, and labels VERBATIM from the source.
  3. Preserve: every maze wall in its original position, thickness, and solid black color, every labeled marker in its original color, shape, position, size, and letter label rendered verbatim above the marker, the white corridor background, top-down camera angle, framing, color palette, saturation, contrast, exposure, image dimensions.
  4. Anchor: "The output must read as the original maze with the connecting lines added on top, not as a re-rendered maze."
  5. Exclusions: only the connecting lines (one per reachable pair), no extra arrows or annotations, no text beyond the existing letter labels, no watermarks, no restyling, no recoloring of walls or markers, no shifted markers.

- Output <image_prompt>your prompt</image_prompt>. The all-pairs connecting-lines overlay is mandatory for this task -- do NOT emit <image_prompt>none</image_prompt>; always enumerate every pair and ask for a corridor line for each.  Do NOT output the action JSON yet.

PHASE 2 (the user has supplied the imagined image, marked "Imagined image:"):
- Inspect the connecting lines: which marker pairs are linked by a colored line through open corridor, and which pairs have no line drawn (= no corridor path = unreachable).
- Use the set of present/absent lines to answer the question; cross-check against the original maze when in doubt.
- Output the final answer in the JSON format the task requires.

Disambiguation: if the most recent user message contains "Imagined image:" or an image labelled as imagined, you are in PHASE 2. Otherwise PHASE 1.
"""


SYSTEM_PROMPTS = {
    "none": "",
    "interleaved_default": INTERLEAVED_PROTOCOL_PROMPT,
    "enclosure_sheep_interleaved": ENCLOSURE_SHEEP_INTERLEAVED_PROMPT,
    "knots_untangle_interleaved": KNOTS_UNTANGLE_INTERLEAVED_PROMPT,
    "separation_one_stroke_interleaved": SEPARATION_ONE_STROKE_INTERLEAVED_PROMPT,
    "continuity_pipe_interleaved": CONTINUITY_PIPE_INTERLEAVED_PROMPT,
    "continuity_2d_maze_interleaved": CONTINUITY_2D_MAZE_INTERLEAVED_PROMPT,
}


def get_system_prompt(key: str) -> str:
    if key not in SYSTEM_PROMPTS:
        known = ", ".join(sorted(SYSTEM_PROMPTS))
        raise KeyError(f"Unknown system prompt key {key!r}. Known keys: {known}")
    return SYSTEM_PROMPTS[key]
