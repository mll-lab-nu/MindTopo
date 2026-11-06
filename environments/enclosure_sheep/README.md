# Enclosure Sheep

`enclosure_sheep` is a TopoBench reasoning task in the `enclosure` category.

The standardized benchmark reads pre-rendered images from `output/` by default and writes `question.jsonl` plus `model_answer.jsonl` in the shared reasoning schema under `output/`.

## Task

The benchmark evaluates four single-image questions:

- `Q1_count_inside`: how many sheep cannot escape to the outside — integer
- `Q2_escape_possibility`: which numbered sheep can escape through fence gaps — `"NONE"` or comma-separated ID list
- `Q3_max_cell_count`: which labeled partition cell contains the most sheep — region label
- `Q4_fence_repair`: how many fence gaps must be repaired to fully close the fence — integer

The prompt follows the shared reasoning template:

- `[Task]`
- image block
- `[Rules]`
- `[Question]`
- `[Answer Format]`

Depending on the task, the canonical response is a JSON integer or JSON string:

```json
{"answer": 2}
{"answer": "1, 3"}
```

## Layout

- `metadata.json`: task metadata
- `frontend/index.html`: interactive renderer entry
- `frontend/src/fence-generator.js`: fence polygon generation
- `frontend/src/sheep-placer.js`: sheep placement and labeling logic
- `frontend/src/region-analyzer.js`: geometric ground-truth helpers
- `frontend/src/fence-scene-renderer.js`: Three.js scene renderer
- `frontend/src/difficulty-controller.js`: difficulty presets
- `backend/serve_frontend.py`: preview server for the frontend
- `backend/vite_server.py`: Vite dev-server launcher used by backend automation
- `backend/generate_samples.py`: headless dataset generation through the frontend batch API
- `backend/internvl3_benchmark.py`: standardized reasoning benchmark runner
- `backend/internvl3_config.py`: local/env/CLI config loader
- `backend/vlm_benchmark_sheep.py`: shared task definitions, prompt builder, parsers, and scoring
- `backend/question_phrasings_sheep.py`: canonical question text for the four benchmark tasks
- `backend/regenerate_dataset.js`: dataset regeneration script
- `output/`: generated samples written by `backend/generate_samples.py`

## Run

Start the frontend preview:

```bash
cd environments/enclosure_sheep
python3 backend/serve_frontend.py --port 8000
```

Then open:

```text
http://127.0.0.1:8000/enclosure_sheep/frontend/index.html
```

Or run the Vite dev server:

```bash
cd environments/enclosure_sheep/frontend
npm install
npm run dev
```

## Benchmark

This snapshot includes `output/question.jsonl` but not its referenced images or
a fallback `dataset/` directory. Restore the matching image assets before
evaluating the fixed benchmark, or generate a separate local dataset using the
commands below.

Run a real model benchmark:

```bash
cd ..
python3 backend/internvl3_benchmark.py \
  --tasks all \
  --output-json output/internvl3_enclosure_sheep.json \
  --output-csv output/internvl3_enclosure_sheep.csv
```

Optional model selection sources, in precedence order:

- `--model ...`
- `.env` via `INTERNVL_MODEL` or `INTERNVL3_MODEL`
- `backend/internvl3_local_config.json` via `"model": "..."`

The benchmark runner is named `internvl3_benchmark.py`, but the request path is OpenAI-compatible. In practice you can point it at other compatible VLMs by changing `--base-url`, `--api-key`, and `--model`.

Run against a separately prepared local dataset (replace `path/to/dataset`):

```bash
cd environments/enclosure_sheep
python3 backend/internvl3_benchmark.py \
  --data-dir path/to/dataset \
  --tasks all \
  --output-json output/internvl3_enclosure_sheep.json \
  --output-csv output/internvl3_enclosure_sheep.csv
```

If you want to use a local config file, create `backend/internvl3_local_config.json` and omit `--config`, or pass a real custom path with `--config /abs/path/to/file.json`.

Examples:

```bash
python3 backend/internvl3_benchmark.py --tasks all --model internvl3.5-latest
python3 backend/internvl3_benchmark.py --tasks all --model gpt-4.1-mini --base-url https://api.openai.com/v1
```

Run an oracle smoke test without calling the API:

```bash
python3 backend/internvl3_benchmark.py \
  --oracle \
  --tasks all \
  --output-json output/oracle_check.json \
  --output-csv output/oracle_check.csv
```

Current default arguments:

- `--data-dir ./output`
- `--tasks all`
- `--difficulty` unset
- `--limit` unset
- `--phrasing` unset
- `--max-tokens 2048`
- `--output-json output/internvl3_enclosure_sheep.json`
- `--output-csv output/internvl3_enclosure_sheep.csv`
- `output/question.jsonl`
- `output/model_answer.jsonl`

## Outputs

The standardized benchmark writes:

- `output/internvl3_enclosure_sheep.json`
- `output/internvl3_enclosure_sheep.csv`
- `output/question.jsonl`
- `output/model_answer.jsonl`

This runner evaluates existing images in place. The JSONL `images` fields therefore point back to files in the selected `--data-dir` rather than copying images into the benchmark output directory.

The JSONL files follow the shared reasoning schema used across TopoBench tasks:

- `question.jsonl` stores `id`, `category`, `type`, `meta_info`, `question`, `images`, and `gt_answer` per sample
- `model_answer.jsonl` stores `id`, `category`, `type`, `meta_info`, `question`, `answer`, `correct`, `invalid_response`, `api_error`, and `raw_response_text` per sample

## Dataset Regeneration

Install frontend dependencies once:

```bash
cd environments/enclosure_sheep/frontend
npm install
```

Then generate samples through the frontend batch API:

```bash
cd ..
python3 backend/generate_samples.py \
  --count 67 \
  --difficulty easy,medium,hard \
  --scene-types fence,partitioned \
  --image-size 1024 \
  --workers 32
```

This is the standard 1000-question dataset (`--count 67` → 1005 rows ≈ 1000). Use `--count 66` → 990 rows if you want to stay strictly under 1000.

`--workers N` opens N parallel Playwright pages, each rendering configs where `i % N == workerIdx`. Each worker has its own three.js scene and WebGL context, so they render independently. `--workers 1` is the default (sequential).

Default outputs:

- `output/dataset_metadata.json`
- `output/summary.json`
- `output/question.jsonl`
- `output/images/<question_id>/...png`

Notes:

- `generate_samples.py` writes images, `dataset_metadata.json`, `summary.json`, and `question.jsonl` in one pass.
- After question rows are built, images are reorganized so that each row id has its own folder under `images/<question_id>/`. Several rows that share a scene get their own copy of the image.
- `--count` means the number of images generated for each `difficulty × scene_type` combination, not the final question count.
- With `--difficulty easy,medium,hard` and `--scene-types fence,partitioned`, total images = `6 × count`.
- After expansion, each `fence` image contributes 3 questions (`Q1`, `Q2`, `Q4`) and each `partitioned` image contributes 2 questions (`Q1`, `Q3`), so total `question.jsonl` rows = `15 × count`.

### Partition layouts

The `partitioned` scene type now mixes four layout families (chosen per-sample by difficulty):

- `grid` — rectangular `numCols × numRows` cells (legacy default).
- `hex_cross` — hexagon split by two long diagonals through the center → 4 cells (2 triangles + 2 quads).
- `nested_polygon` — outer N-gon with a concentric inner N-gon joined by corresponding-vertex spokes → `N+1` cells (e.g. hexagon-in-hexagon → 7 cells).
- `polygon_star` — N-gon with radial spokes from the center to every vertex → `N` triangular cells.

`numPolygonSides` controls the outer-polygon side count for `hex_cross` (always 6), `nested_polygon`, and `polygon_star`.

The benchmark reads newly generated samples by default. To use a separately
prepared local dataset, pass `--data-dir path/to/dataset`.

## Benchmark vs Evaluation

`backend/internvl3_benchmark.py` is the per-environment runner. It does one direct model run over this task's dataset, scores the answers, and writes:

- summary JSON / CSV
- `question.jsonl`
- `model_answer.jsonl`

That is useful for smoke tests, debugging prompt format, checking one model quickly, and materializing the standardized JSONL artifacts.

`lmms_eval_tasks/` is the cross-model evaluation layer. It consumes `question.jsonl` and plugs tasks into the LMMS Eval framework so multiple models can be compared under a shared task interface and shared metrics. That layer is what you want for model-vs-model comparison.

`enclosure_sheep` is now also registered under `lmms_eval_tasks/`, so once `output/question.jsonl` has been refreshed you can run it through the shared `lmms-eval` pipeline like the other static reasoning tasks.
