# Enclosure Hole Detection

`enclosure_hole_detection` is a TopoBench reasoning task. Each data item contains one top-down board image. The model answers how many visible openings pass through the board to open space.

Canonical answer format:

```json
{"answer": 3}
```

## Difficulty

Difficulty is controlled by the allowed hole types and final visible through-hole answer count.

- easy: difficulty `1`
  - board size `10 x 10`
  - board shape: seed-selected from `rect`, `circle`, `polygon`
  - visible through-hole answer range `5-10`
  - hole types: `hole`, `pit`
- medium: difficulty `2`
  - board size `10 x 10`
  - board shape: seed-selected from `rect`, `circle`, `polygon`
  - visible through-hole answer range `7-13`
  - hole types: `hole`, `pit`, `mixed`
- hard: difficulty `3`
  - board size `10 x 10`
  - board shape: seed-selected from `rect`, `circle`, `polygon`
  - visible through-hole answer range `9-17`
  - hole types: `hole`, `pit`, `mixed`, `mixed2`

Seed behavior is deterministic. The scripts initialize one RNG with `--seed 12345` by default, then consume one generated scene seed per repeat. With `--difficulty 1,2,3 --repeats 2`, generation order is:

- difficulty 1: repeats 1-2
- difficulty 2: repeats 1-2
- difficulty 3: repeats 1-2

Each per-scene seed also fixes the board shape. For the same generated scene seed, the board shape and all random scene details are reproduced exactly.
The same per-scene seed also deterministically controls whether the lower board is present.

## Frontend Debug

```bash
# From the repository root:
cd environments/enclosure_hole_detection
uv run python backend/serve_frontend.py --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

The debug page exposes `Difficulty Preset`, `Global Seed`, board geometry, hole settings, camera, appearance, and lighting controls.

## Generate Question JSONL

Generate easy data:

```bash
uv run python backend/generate_samples.py \
  --difficulty 1 \
  --repeats 100 \
  --seed 12345 \
  --output-json output/question_easy.jsonl \
  --image-root output/images
```

Generate medium data:

```bash
uv run python backend/generate_samples.py \
  --difficulty 2 \
  --repeats 100 \
  --seed 12345 \
  --output-json output/question_medium.jsonl \
  --image-root output/images
```

Generate hard data:

```bash
uv run python backend/generate_samples.py \
  --difficulty 3 \
  --repeats 100 \
  --seed 12345 \
  --output-json output/question_hard.jsonl \
  --image-root output/images
```

Generate all default difficulties:

```bash
uv run python backend/generate_samples.py \
  --difficulty 1,2,3 \
  --repeats 1 \
  --seed 12345 \
  --output-json output/question.jsonl \
  --image-root output/images
```

Default generate args:

- `--difficulty 1,2,3`
- `--repeats 1`
- `--seed 12345`
- `--output-json output/question.jsonl`
- `--image-root output/images`
- `--headless` enabled by default

Each image folder is named with the corresponding JSONL data `id`:

```text
output/images/{data_id}/state_0.png
```

## Benchmark

Load an existing generated question JSONL and call the model API:

```bash
uv run python backend/internvl3_benchmark.py \
  --config ../api.json \
  --question-jsonl output/question.jsonl
```

Load an existing generated question JSONL with the local oracle:

```bash
uv run python backend/internvl3_benchmark.py \
  --oracle \
  --question-jsonl output/question.jsonl \
  --output-json output/internvl3_enclosure_hole_detection.json \
  --output-csv output/internvl3_enclosure_hole_detection.csv \
  --image-root output/images
```

Run direct generation plus model benchmark:

```bash
uv run python backend/internvl3_benchmark.py \
  --config ../api.json \
  --difficulty 1,2,3 \
  --repeats 1 \
  --seed 12345
```

Run direct generation plus oracle benchmark:

```bash
uv run python backend/internvl3_benchmark.py \
  --oracle \
  --difficulty 1,2,3 \
  --repeats 1 \
  --seed 12345
```

Default benchmark args:

- `--difficulty 1,2,3`
- `--repeats 1`
- `--seed 12345`
- `--max-tokens 0`
- `--output-json output/internvl3_enclosure_hole_detection.json`
- `--output-csv output/internvl3_enclosure_hole_detection.csv`
- `--image-root output/images`
- `--headless` enabled by default

Benchmark output includes:

- full JSON results
- full CSV results
- `_by_difficulty.json/csv`
- `question.jsonl`
- `model_answer.jsonl`
- per-sample images and prompt files under `output/images/{data_id}/`

## Smoke Test

```bash
uv run python backend/playwright_smoke.py --headless --difficulty 2 --seed 12345
```
