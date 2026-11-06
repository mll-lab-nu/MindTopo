# Order Swap 2D Puzzle

`order_swap_2d_puzzle` is a TopoBench interactive planning task in the `order` category.

The model sees two images at each turn:

- the current grid
- the goal grid

It must output the next legal move as JSON:

```json
{"answer": {"row": 1, "col": 2}}
```

Rows and columns are zero-based. A legal move selects one non-empty cell; that block swaps with the blank cell. The task is solved when the current grid exactly matches the goal grid.

The benchmark uses the official multi-turn chat format. The first user turn contains the task, rules, answer format, and current task. Later user turns contain only the updated current task. Each turn sends the current-state image and the goal-state image as separate image inputs.

## Difficulty

Grid shape is the setup level:

- easy: seed-determined `2x2`, `2x3`, `3x2`, or `3x3`
- medium: seed-determined `3x4` or `4x3`
- hard: `4x4`

The environment supports grids up to `4x4`; the oracle uses an exact shortest action sequence for the blank-swap rule.
The frontend, generator, and evaluation helper all use the same mapping: choose a difficulty, then use the seed to resolve the concrete grid shape and sample the initial/goal arrangements.

## Debug Frontend

```bash
# From the repository root:
cd environments/order_swap_2d_puzzle
uv run python backend/serve_frontend.py --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

The debug page exposes difficulty, rows, columns, seed, and a row/column action control. Selecting `easy` resolves to `2x2`, `2x3`, `3x2`, or `3x3` from the seed; selecting `medium` resolves to `3x4` or `4x3` from the seed. To inspect an exact generated row, load its reset config in the browser console with `window.topoBench.loadQuestionRow(row)`.

## Generate Questions

`generate_samples.py` writes an interactive `question.jsonl`. It does not run the model and does not produce rollout images.
Each row stores the canonical reproduction payload in `meta_info.initial_state.reset_config`; `initialGrid` and `goalGrid` are included only as derived readable views.

```bash
# From the repository root:
cd environments/order_swap_2d_puzzle
uv run python backend/generate_samples.py \
  --repeats 100 \
  --output-json output/question.jsonl
```

Current generate defaults:

- `--difficulties easy,medium,hard`
- `--repeats 3`
- `--seed-start 1`
- `--budget-multiplier 1.2`
- `--output-json output/question.jsonl`

Use `--grids 2x2,2x3,3x2,3x3,3x4,4x3,4x4` only when you want explicit grid-shape generation instead of difficulty-based generation.

## Run Oracle Or Model

Run oracle evaluation and write `question.jsonl` plus `model_answer.jsonl`:

```bash
# From the repository root:
cd environments/order_swap_2d_puzzle
uv run python backend/internvl3_benchmark.py \
  --oracle \
  --difficulties easy,medium,hard \
  --repeats 5 \
  --output-json output/internvl3_order_swap_2d_puzzle.json \
  --output-csv output/internvl3_order_swap_2d_puzzle.csv
```

Run from an existing generated `question.jsonl`:

```bash
uv run python backend/internvl3_benchmark.py \
  --oracle \
  --question-jsonl output/question.jsonl \
  --output-json output/internvl3_order_swap_2d_puzzle.json \
  --output-csv output/internvl3_order_swap_2d_puzzle.csv
```

Outputs:

- `output/question.jsonl`
- `output/model_answer.jsonl`
- per-step current/goal images plus `step_0000_prompt_debug.txt` / `step_0000_pure_prompt.txt` style prompt files under `output/{id}/`

## Planning Harness

Run from a generated `question.jsonl`:

```bash
# Run from the repository root.
uv run python planning_eval_tasks/agent_runner.py env=order_swap_2d_puzzle model=oracle
```

## Smoke Tests

Environment smoke test:

```bash
# From the repository root:
cd environments/order_swap_2d_puzzle
uv run python gym/test_env.py --headless
```

Protocol smoke test:

```bash
uv run python gym/test_protocol.py --headless
```
