# Knots Untangle

`knots_untangle` is a TopoBench interactive planning task in the `knots` category.

The current benchmark follows the shared interactive rollout format used by the standardized planning environments:

- one `question.jsonl` line per episode initialization
- one `model_answer.jsonl` line per episode rollout
- per-step screenshots written under `output/generate_samples/images/...`
- headless Playwright execution by default
- local `random`, `greedy`, and `oracle` baselines via `--policy`

## Task

The model sees one current-state screenshot of a pegboard with several tangled ropes and must output the next legal endpoint move as JSON:

```json
{"src_row":0,"src_col":1,"tgt_row":2,"tgt_col":0}
```

The action specifies one occupied source hole and one empty target hole.

The prompt uses:

- `[Task]`
- `[Rules]`
- `[Answer Format]`
- `[Current Task]`

On the first turn, the prompt includes the full task and rule description. On later turns, the prompt only includes the updated current task block.

Difficulty controls board size and rope count:

- `easy`: `5x5` grid, `4` ropes
- `medium`: `6x6` grid, `5` ropes
- `hard`: `6x6` grid, `6` ropes

## Layout

- `metadata.json`: task metadata
- `frontend/index.html`: main renderer entry
- `frontend/src/main.ts`: `window.topoBench` protocol layer
- `frontend/src/untangle.ts`: puzzle generation and rope/pegboard logic
- `frontend/src/render.ts`: Three.js renderer
- `backend/serve_frontend.py`: frontend preview server with correct MIME types for `.ts`
- `backend/internvl3_benchmark.py`: standardized interactive benchmark runner
- `backend/internvl3_config.py`: local/env/CLI config loader for API settings
- `backend/oracle_solver.py`: endpoint-level BFS oracle and greedy helpers
- `gym/env.py`: Playwright-backed Gym wrapper
- `gym/test_env.py`: environment smoke test
- `gym/test_protocol.py`: protocol smoke test
- `gym/example_usage.py`: random-play example

## Generate question.jsonl

```bash
# From the repository root:
uv run python environments/knots_untangle/backend/generate_samples.py \
  --output-json environments/knots_untangle/output/question.jsonl
```

## Run

Environment smoke test:

```bash
cd environments/knots_untangle
python gym/test_env.py --headless
```

Protocol smoke test:

```bash
cd environments/knots_untangle
python gym/test_protocol.py --headless
```

Start the frontend preview:

```bash
cd environments/knots_untangle
python backend/serve_frontend.py --port 8000
```

Then open:

```text
http://127.0.0.1:8000/
```

Or run the Vite dev server:

```bash
cd environments/knots_untangle/frontend
npm install
npm run dev
```

## Benchmark

Run a real model benchmark, generating fresh episodes (writes both `question.jsonl` and `model_answer.jsonl`):

```bash
cd environments/knots_untangle
INTERN_API_KEY=<your-key> \
python backend/internvl3_benchmark.py \
  --policy internvl \
  --difficulties easy,medium,hard \
  --repeats 3 \
  --output-json output/benchmark_output/internvl3_knots_untangle.json \
  --output-csv output/benchmark_output/internvl3_knots_untangle.csv
```

Replay an existing `question.jsonl` (only `model_answer.jsonl` is written; `question.jsonl` is left untouched):

```bash
cd environments/knots_untangle
INTERN_API_KEY=<your-key> \
python backend/internvl3_benchmark.py \
  --policy internvl \
  --question-jsonl output/question.jsonl \
  --output-json output/benchmark_output/internvl3_knots_untangle.json \
  --output-csv output/benchmark_output/internvl3_knots_untangle.csv
```

In replay mode `--difficulties` and `--repeats` are ignored; difficulty, seed, and step budget come from each row's `meta_info`.

Generate 180 question rows (60 per difficulty) without calling the API:

```bash
# From the repository root:
cd environments/knots_untangle
uv run python backend/generate_samples.py \
  --difficulties easy,medium,hard \
  --repeats 60 \
  --output-json output/question.jsonl
```

Current default arguments:

- `--policy internvl`
- `--difficulties easy,medium,hard`
- `--repeats 20`
- `--max-tokens 0`
- `--model-image-max-side 768`
- `--image-transport data_url`
- `--step-image-root output/generate_samples/images`
- `--image-dir output/generate_samples/internvl3_frames`
- `--output-json` not written unless explicitly provided
- `--output-csv` not written unless explicitly provided
- `--question-jsonl` empty (replay mode disabled by default)
- `output/question.jsonl`
- `output/benchmark_output/model_answer.jsonl`
- `--headless` enabled by default
- `--include-prompt-trajectory` enabled by default
- `--save-demo-gifs` enabled by default

## Outputs

Benchmark outputs are written to:

- `output/question.jsonl`
- `output/benchmark_output/model_answer.jsonl`
- optional summary JSON and CSV files if `--output-json` or `--output-csv` is provided
- per-episode screenshots and prompt snapshots under `output/benchmark_output/images/`

Per-episode files are organized as:

- `output/benchmark_output/images/<episode_id>/`

Where `<episode_id>` matches the `id` field in `output/question.jsonl`, e.g. `knots_untangle_difficulty_easy_seed_00`.

Each episode directory may contain:

- `step_0000.png`
- `step_0000_after.png`
- `step_0000_prompt.txt`
- `episode_summary.json`
- `demo.gif`

Each saved state image is labeled in-image as `State 0`, `State 1`, and so on.

Unlike some earlier interactive benchmarks, this runner currently writes one combined prompt snapshot file per step (`step_XXXX_prompt.txt`) instead of separate `prompt_debug.txt` and `pure_prompt.txt` files.

The interactive JSONL files follow the shared schema used across TopoBench tasks:

- `question.jsonl` stores episode initialization only
- `model_answer.jsonl` stores the rollout `trajectory`
- each trajectory step keeps `step_index`, `current_images`, `state`, `answer`, `invalid_response`, `api_error`, `illegal`, and `raw_response_text`
