# Enclosure Chat Noir

`enclosure_chat_noir` is an interactive TopoBench task. The model sees one board image per turn and must block one open non-cat hex cell before the cat reaches the boundary.

Canonical answer format:

```json
{"answer":12}
```

## Difficulty

Difficulty is defined only by the cat policy:

- easy: `cat_policy=easy`, a seeded mixed walker that is random 50% of the time and shortest-path greedy 50% of the time
- medium: `cat_policy=medium`, the previous shortest-path greedy runner
- hard: `cat_policy=hard`, a connectivity-aware greedy runner

`board_radius` and `initial_block_count` are setup parameters. They are seed-sampled by default, but they do not define difficulty.

Seeds are consumed sequentially across policy, then repeat. With default auto setup and `--cat-policies easy,medium,hard --repeats 2 --seed-start 1`, the generated rows use:

- easy: seeds `1,2`
- medium: seeds `3,4`
- hard: seeds `5,6`

For a fixed command, generation is deterministic.

With `--board-radius auto`, each seed samples `board_radius` from `{3,4}`. With `--initial-block-count auto`, each seed samples the block count from:

- radius 3: easy `8-10`, medium `10-12`, hard `12-14`
- radius 4: easy `10-13`, medium `13-16`, hard `16-19`

Initial blockers are sampled uniformly at random from non-cat cells. The generator then uses deterministic rejection sampling with a bounded multi-step search filter. A sampled setup is kept only when the cat has at least one legal move, has at least one path to the boundary, the search solver can find at least 5 distinct winning first block moves within its configured horizon, and at least 3 of those winning first block moves are adjacent to the cat's starting cell. The horizon is capped by the episode `max_actions_per_traj`. The same command and seed still reproduce the same accepted setup.

## Frontend Debug

```bash
cd /Users/andrewliu/Desktop/manling/topobench/environments/enclosure_chat_noir
conda activate ptm
python backend/serve_frontend.py --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

The debug page exposes `Board radius`, `Initial blocked`, `Min winning starts`, `Cat policy`, and `Seed`. `Board radius`, `Initial blocked`, and `Min winning starts` accept `auto` or a concrete integer. For `Min winning starts`, `auto` means easy=5, medium=5, hard=5.
When `catIndex` and `initialBlockedIndices` are not provided, the frontend uses the same seed-based initializer as `backend/generate_samples.py`, so the same radius, initial blocked count, policy, and seed produce the same initial state.

## Generate Question JSONL

Generate easy data:

```bash
python backend/generate_samples.py \
  --board-radius auto \
  --initial-block-count auto \
  --cat-policies easy \
  --repeats 100 \
  --seed-start 1 \
  --output-json output/question_easy.jsonl
```

Generate medium data:

```bash
python backend/generate_samples.py \
  --board-radius auto \
  --initial-block-count auto \
  --cat-policies medium \
  --repeats 100 \
  --seed-start 1 \
  --output-json output/question_medium.jsonl
```

Generate hard data:

```bash
python backend/generate_samples.py \
  --board-radius auto \
  --initial-block-count auto \
  --cat-policies hard \
  --repeats 100 \
  --seed-start 1 \
  --output-json output/question_hard.jsonl
```

Generate all three difficulties in one file:

```bash
python backend/generate_samples.py \
  --board-radius auto \
  --initial-block-count auto \
  --cat-policies easy,medium,hard \
  --repeats 100 \
  --seed-start 1 \
  --output-json output/question.jsonl
```

Default generate args:

- `--board-radius auto`
- `--initial-block-count auto`
- `--cat-policies easy,medium,hard`
- `--repeats 1`
- `--seed-start 1`
- `--min-winning-first-actions auto` (easy=5, medium=5, hard=5)
- `--min-adjacent-winning-first-actions 3`
- `--output-json output/question.jsonl`

## Benchmark

Load an existing question JSONL and run oracle mode:

```bash
python backend/internvl3_benchmark.py \
  --oracle \
  --question-jsonl output/question.jsonl \
  --output-json output/internvl3_enclosure_chat_noir.json \
  --output-csv output/internvl3_enclosure_chat_noir.csv \
  --image-root output
```

Run direct generation plus benchmark without a prebuilt question JSONL:

```bash
python backend/internvl3_benchmark.py \
  --oracle \
  --board-radius auto \
  --initial-block-count auto \
  --cat-policies easy,medium,hard \
  --repeats 1 \
  --seed-start 1
```

Default benchmark args:

- `--board-radius auto`
- `--initial-block-count auto`
- `--cat-policies easy,medium,hard`
- `--repeats 1`
- `--seed-start 1`
- `--min-winning-first-actions auto` (easy=5, medium=5, hard=5)
- `--min-adjacent-winning-first-actions 3`
- `--max-tokens 0`
- `--output-json output/internvl3_enclosure_chat_noir.json`
- `--output-csv output/internvl3_enclosure_chat_noir.csv`
- `--image-root output`
- `--headless` enabled by default

Benchmark output includes:

- full JSON results
- full episode CSV
- `_by_combo.json` and `_by_combo.csv`
- `question.jsonl`
- `model_answer.jsonl`
- per-step images and prompt files under `output/{question_id}/`

Each episode directory contains files such as `step_0000_current.png`, `step_0000_prompt_debug.txt`, and `step_0000_pure_prompt.txt`.

In `--oracle` mode, the benchmark first asks the same bounded multi-step search solver for an action and falls back to the older one-step heuristic only when the search cannot find a move.

## Smoke Tests

```bash
python gym/test_env.py --headless
python gym/test_protocol.py --headless
```
