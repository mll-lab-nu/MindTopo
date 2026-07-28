# Knots Static

`knots_static` is a TopoBench reasoning task in the `knots` category.

The standardized benchmark reads pre-rendered images from `output/generate_samples/` by default and writes `question.jsonl` plus `model_answer.jsonl` in the shared reasoning schema under `output/`.

## Task

The benchmark evaluates five single-image tasks defined in `backend/vlm_benchmark.py`:

- `T01_structure_classification`: is the scene a simple closed ring, knot, link, unlinked multi-component scene, open-ended unknotted rope, or mixed — option letter
- `T02_component_count`: how many separate rope components are visible — integer
- `T03_link_topology`: what link topology is shown — option letter
- `T04_link_property`: if one colored ring is removed, which remaining rings become free — JSON color list
- `T05_linked_count`: how many components are linked to at least one other component — integer

The legacy benchmark helper still contains additional pair and anchor tasks, but the standardized `backend/internvl3_benchmark.py` exports only this single-image subset.

The prompt follows the shared reasoning template:

- `[Task]`
- image block
- `[Rules]`
- `[Question]`
- `[Answer Format]`

## Layout

- `metadata.json`: task metadata
- `frontend/index.html`: gallery and renderer entry
- `frontend/src/unified-gallery.js`: main frontend app
- `frontend/src/invariance-renderer.js`: Three.js rendering and camera setup
- `frontend/src/knot-type-registry.js`: knot and link type definitions
- `frontend/src/centerline-pipeline.js`: rope geometry preprocessing
- `backend/serve_frontend.py`: preview server for the frontend
- `backend/vite_server.py`: Vite dev-server launcher used by backend automation
- `backend/generate_samples.py`: headless dataset generation through the frontend batch API
- `backend/internvl3_benchmark.py`: standardized reasoning benchmark runner
- `backend/internvl3_config.py`: local/env/CLI config loader
- `backend/vlm_benchmark.py`: shared task definitions, prompt logic, parsers, and scoring
- `backend/question_phrasings.py`: canonical question phrasing definitions
- `backend/validate_dataset.py`: dataset integrity checks
- `dataset/`: checked-in baseline dataset available for fallback benchmarking
- `output/generate_samples/`: generated samples written by `backend/generate_samples.py`

## Run

Start the frontend preview:

```bash
cd environments/knots_static
python backend/serve_frontend.py --port 8000
```

Then open:

```text
http://127.0.0.1:8000/knots_static/frontend/index.html
```

Or run the Vite dev server:

```bash
cd environments/knots_static/frontend
npm install
npm run dev
```

## Benchmark

Run a real model benchmark:

```bash
cd environments/knots_static
python3 backend/internvl3_benchmark.py \
  --tasks all \
  --output-json output/internvl3_knots_static.json \
  --output-csv output/internvl3_knots_static.csv
```

Optional model selection sources, in precedence order:

- `--model ...`
- `.env` via `INTERNVL_MODEL` or `INTERNVL3_MODEL`
- `backend/internvl3_local_config.json` via `"model": "..."`

The benchmark runner is named `internvl3_benchmark.py`, but the request path is OpenAI-compatible. In practice you can point it at other compatible VLMs by changing `--base-url`, `--api-key`, and `--model`.

Run against the checked-in baseline dataset instead:

```bash
cd environments/knots_static
python3 backend/internvl3_benchmark.py \
  --data-dir dataset \
  --tasks all \
  --output-json output/internvl3_knots_static.json \
  --output-csv output/internvl3_knots_static.csv
```

If you want to use a local config file, create `backend/internvl3_local_config.json` and omit `--config`, or pass a real custom path with `--config /abs/path/to/file.json`.

Examples:

```bash
python3 backend/internvl3_benchmark.py --tasks all --model internvl3.5-latest
python3 backend/internvl3_benchmark.py --tasks all --model gpt-4.1-mini --base-url https://api.openai.com/v1
```

Run an oracle smoke test without calling the API:

```bash
cd environments/knots_static
python3 backend/internvl3_benchmark.py \
  --oracle \
  --tasks all \
  --output-json output/oracle_check.json \
  --output-csv output/oracle_check.csv
```

Current default arguments:

- `--data-dir ./output/generate_samples`
- `--tasks all`
- `--difficulty` unset
- `--limit` unset
- `--phrasing` unset
- `--max-tokens 200`
- `--output-json output/internvl3_knots_static.json`
- `--output-csv output/internvl3_knots_static.csv`
- `output/question.jsonl`
- `output/model_answer.jsonl`

## Outputs

The standardized benchmark writes:

- `output/internvl3_knots_static.json`
- `output/internvl3_knots_static.csv`
- `output/question.jsonl`
- `output/model_answer.jsonl`

This runner evaluates existing dataset images in place. The JSONL `images` fields therefore point back to dataset files using relative paths rather than copying images into the benchmark output directory.

The JSONL files follow the shared reasoning schema used across TopoBench tasks:

- `question.jsonl` stores `id`, `category`, `type`, `meta_info`, `question`, `images`, and `gt_answer` per sample
- `model_answer.jsonl` stores `id`, `category`, `type`, `meta_info`, `question`, `answer`, `correct`, `invalid_response`, `api_error`, and `raw_response_text` per sample

## Dataset Regeneration

Install frontend dependencies once:

```bash
cd environments/knots_static/frontend
npm install
```

Then generate samples through the frontend batch API. The frontend selector, registry, dataset generator, and backend ground-truth code cover the same 24 generated knot/link types:

- 16 single-rope knots/unknots: `unknot`, `twisted_ring`, `spiral_disk`, `kinky_unknot`, `ring_like_open_rope`, `loose_open_knot`, `ring_like_open_knot`, `loose_cinquefoil`, `occluded_knot`, `trefoil`, `figure8`, `torus_2_5`, `torus_2_7`, `torus_2_9`, `torus_3_4`, `torus_3_5`
- 8 multi-component scenes: `hopf_link`, `unlinked_rings`, `chain`, `borromean`, `double_hopf`, `link_cluster`, `hopf_plus_free`, `chain_plus_free`

Generate:

```bash
cd ..
python3 backend/generate_samples.py \
  --variants-per-type 30 \
  --angles-per-variant 3 \
  --num-pairs 0 \
  --image-size 1024
```

The standardized `internvl3_benchmark.py` only consumes single-image samples (T01–T05), so `--num-pairs 0` skips wasted pair-render compute. With `--variants-per-type 30` you end up with 589 single-image scenes before task expansion.

To cap question.jsonl at 1000 rows (interleaved per sample, balanced across task types):

```bash
cd ..
python3 backend/generate_samples.py \
  --variants-per-type 30 \
  --angles-per-variant 3 \
  --num-pairs 0 \
  --image-size 1024 \
  --max-questions 1000 \
  --workers 16
```

Each row gets its own image folder under `output/images/<question_id>/<basename>.png`.

Default outputs:

- `output/dataset_metadata.json`
- `output/summary.json`
- `output/question.jsonl`
- `output/images/<question_id>/...png`

The benchmark now reads these newly generated samples by default. To benchmark the checked-in dataset instead, pass `--data-dir dataset`.

## Notes

- This task is reasoning-only, so there is no `gym/` wrapper.
- Difficulty is normalized to `easy`, `medium`, or `hard` at export time using the metadata-driven `compute_difficulty()` helper in `backend/vlm_benchmark.py`.
- `knots_static` is already registered under `lmms_eval_tasks/`, so once `output/question.jsonl` has been refreshed you can run it through the shared `lmms-eval` pipeline.
