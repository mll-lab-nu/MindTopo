# Continuity 2D Maze

`continuity_2d_maze` is a TopoBench L1 reasoning task. It renders a top-down 2D maze inside a square background, samples two labeled navigation points, and asks whether the two points are connected through the maze passages.

## Task

The renderer draws a grid-based 2D maze (walls as solid lines/blocks) on a square canvas, places labeled points in the maze, and exports a single top-down PNG. Two question types are supported:

- **`reachability_set`** (Q1) — points A, B, C, D are sampled. The model is asked which OTHER labeled points are connected to target A through the corridors; the answer is a lex-sorted bracket-list like `[B, D]` (empty list `[]` if A is isolated from all others).
- **`bar_removal`** (Q2) — fix exactly two points A and B that are guaranteed **disconnected** in the base maze. Colored "bars" (purple / red / green / blue / yellow / orange) are painted over N of the maze's blocking walls (H/V Mode-A walls **or** full dA diagonals). The model must list all bar colors whose **single-bar removal** reconnects A and B; the answer is a color set like `[purple, red]`.

Connectivity is defined identically in both variants: there must exist a path in the corridor BFS graph that does not cross any wall. Both question types share `answer_type = "name_list"` and evaluate with set-equality via `parse_name_list`.

## Folder Layout

- `metadata.json`: machine-readable task metadata (protocol §8).
- `frontend/`: Three.js + TypeScript + Vite scene renderer. Exposes `window.topoBench.{generate, screenshot}`.
- `backend/`: Python sample generator and benchmark runner (Playwright-driven). Populated in later iterations.

## Tech Stack

- Rendering: Three.js (orthographic camera, top-down view) on a square canvas.
- Language: TypeScript.
- Build: Vite (dev server on port `3100`).
- Backend driver: Python + Playwright (added in later iterations).

## Output Schema

Each generated sample follows the universal TopoBench reasoning format. Q1 example (`reachability_set`):

```json
{
  "id": "continuity_2d_maze_difficulty_easy_seed_12345_reachability_set_topdown_0000",
  "question": "Which of the other labeled points are connected to point A in this 2D maze? Respond with a bracket-enclosed comma-separated list of point names, for example '[A, D]'. If none, respond with '[]'.",
  "answer": "[B, D]",
  "question_type": "reachability_set",
  "answer_type": "name_list",
  "metadata": {"target": "A", "points": [{"name": "A", "x": 0, "y": 0}, "..."]},
  "images": ["images/easy/grid_05/.../topdown.png"]
}
```

Q2 example (`bar_removal`):

```json
{
  "id": "continuity_2d_maze_difficulty_easy_seed_12345_bar_removal_topdown_0000",
  "question": "Which single-bar removals would reconnect point A and point B? The bars available are: blue, green, purple, red, yellow. ...",
  "answer": "[purple]",
  "question_type": "bar_removal",
  "answer_type": "name_list",
  "metadata": {
    "target_pair": ["A", "B"],
    "correct_removals": ["purple"],
    "bars": [{"id": "bar_0", "color": "purple", "edge": "3,2,e"}, "..."]
  },
  "images": ["images/easy/grid_06/.../topdown.png"]
}
```

`images` is always a single-element list (one top-down view per sample).

## Generate Samples

One-time setup (local machine with Node and a browser):

```bash
cd continuity_2d_maze/frontend
npm install

cd ../backend
pip install -r requirements.txt
playwright install chromium
```

Run the generator (Q1 / reachability_set; default):

```bash
cd continuity_2d_maze
python backend/generate_samples.py \
  --question-type reachability_set \
  --difficulty-tier easy,medium,hard \
  --count 3 \
  --output-json output/generate_samples/samples.json \
  --image-root output/generate_samples/images \
  --output-question-jsonl output/generate_samples/question.jsonl
```

Run the generator (Q2 / bar_removal):

```bash
python backend/generate_samples.py \
  --question-type bar_removal \
  --difficulty-tier easy,medium,hard \
  --count 3 \
  --bar-count 5 \
  --output-json output/generate_samples/q2_samples.json \
  --image-root output/generate_samples/q2_images \
  --output-question-jsonl output/generate_samples/q2_question.jsonl
```

Defaults and flags:

- `--count 3` — scenes per target tier (rejection-sampled until each tier has this many).
- `--seed 12345` — base seed; per-scene seed = `seed + global_attempt`.
- `--question-type` — `reachability_set` (default Q1; one list-answer per scene, target fixed at `A`) or `bar_removal` (Q2; color-set answer).
- `--difficulty-tier easy,medium,hard` — metric-driven tier target; comma-separated subset. Each tier's jitter box and threshold differ per `--question-type` (Q1 uses `DIFFICULTY_TIER_THRESHOLDS = {6, 12}`, Q2 uses `Q2_DIFFICULTY_TIER_THRESHOLDS = {5, 9}`).
- `--bar-count 5` — Q2 only; number of colored bars per scene (1–6, max = palette size).
- `--max-attempts-per-scene 200` — safety cap for rejection sampling.
- `--dump-metric-distribution N` — calibration helper: dump histogram of `scene_score` across N jittered configs (respects `--question-type` to choose Q1 vs Q2 jitter box). Use this to recalibrate tier thresholds.
- `--headless` — defaults to true; use `--no-headless` to watch the browser.
- `--port 0` — auto-pick a free port; the Vite dev server is started and stopped by the script.

Total samples: `count × |tiers|` (one question per scene for both `reachability_set` and `bar_removal`).

## Interactive UI

`npm run dev` in `frontend/` starts a Vite server with a live UI. The sidebar has two tabs:

- **Q1 Reachability** — Parameters (`seed / grid_size / point_count`), Advanced (`wall_density / diagonal_ratio / partial_ratio / partial_bc_split`, collapsed), Generate / Next Seed, Difficulty Tier buttons (`Easy / Medium / Hard` rejection-sample to match the target tier using Q1 thresholds), Presets, Export PNG, Stats (tier/score + per-pair yes/no aggregates), Pairs list for debug. Preview panel shows the `reachability_set` question for target `A` plus the expected list answer like `[B, D]`.
- **Q2 Bar Removal** — Parameters (`seed / grid_size / bar_count`), Advanced (`wall_density / diagonal_ratio / partial_ratio / partial_bc_split`, collapsed; default `wall_density=0.5`), Generate / Next Seed, Difficulty Tier buttons (use Q2 thresholds + Q2 jitter box), Export PNG, Stats (tier, score, target_pair, correct_removals list, bar_count), single-pair list.

Each tab has its own form state. The URL hash (`#q1` / `#q2`) mirrors the active tab so reloads and deep links preserve context. A right-side preview panel shows the rendered question text, expected answer, sample metadata (JSON), and the full prompt template (collapsed) — matching what backend will emit for the currently displayed scene.

The UI and CLI share the same generation pipeline (`window.topoBench.generate(config)`); backend Playwright calls trigger Q2 by setting `config.question_type = 'bar_removal'`.

## Benchmark

`backend/run_benchmark.py` reads `question.jsonl` (produced by `generate_samples.py`) and writes `model_answer.jsonl` + `score.json`. Iter 6 ships stub strategies only — no real model client yet.

```bash
python backend/run_benchmark.py --strategy oracle
```

Available `--strategy` values:

- `oracle` — copies `gt_answer`; accuracy should be 1.000 on every question type.
- `bar_random` — Q2 baseline: pick one random bar color from the scene's bars and return `[<color>]`. On Q1 `reachability_set` rows (no `bars` in meta) falls back to `[]`. Expected Q2 accuracy ≈ `E[|correct_removals|] / bar_count`.
- `bar_empty` — universal baseline: always return `[]`. Matches gt only when the gt set happens to be empty (Q2 guarantees non-empty; Q1 occasionally has target A isolated).

Pass `--dump-prompt` to include the rendered prompt in each `model_answer.jsonl` row. Answer comparison is set-equality via `parse_name_list` for both active question types.

Example output (Q2 tier run, 3 scenes per tier):

```
strategy=oracle      total=9  correct=9  accuracy=1.000
strategy=bar_random  total=9  correct=4  accuracy=0.444
strategy=bar_empty   total=9  correct=0  accuracy=0.000
```

Prompt templates live in `backend/prompts.py` (`REACHABILITY_SET_TEMPLATE` for Q1, `BAR_REMOVAL_TEMPLATE` for Q2) and are mirrored in `frontend/src/maze/prompts.ts` for the UI preview panel. Run `python backend/check_prompts_sync.py` to verify the two sources are byte-equal (normalized for Python's `{{…}}` escape convention). Real-model clients should use `split_reachability_set_prompt_at_images(question)` / `split_bar_removal_prompt_at_images(question)` to get `(before_images, after_images)` halves and insert multimodal image content between them — see `continuity_3d_maze/backend/internvl3_benchmark.py` for the reference shape.

## Notes

- This task is reasoning-only, so there is no `gym/` wrapper.
- The 2D renderer uses Three.js orthographic projection instead of a real 3D engine.
