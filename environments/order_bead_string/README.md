# Order Bead String

`order_bead_string` is a TopoBench reasoning task in the `order` category.

The standardized benchmark reads pre-rendered images from `output/` by default and writes `question.jsonl` plus `model_answer.jsonl` in the shared reasoning schema under `output/`.

## Task

The benchmark evaluates two tasks:

- `T_BS01_describe_sequence`: trace the bead string from one endpoint to the other and report the bead colors in order — comma-separated color sequence
- `T_BS02_pair_relationship`: compare two bead-string images and classify the relationship between their color sequences — `IDENTICAL`, `REVERSED`, `CYCLIC_ROTATION`, or `DIFFERENT`

The prompt follows the shared reasoning template:

- `[Task]`
- image block
- `[Rules]`
- `[Question]`
- `[Answer Format]`

Canonical answers look like:

```json
{"answer": "RED, BLUE, GREEN"}
{"answer": "CYCLIC_ROTATION"}
```

## Layout

- `metadata.json`: task metadata
- `frontend/index.html`: gallery and renderer entry
- `frontend/src/bead-gallery.js`: main frontend app plus dataset generation helpers
- `frontend/src/bead-curve-library.js`: parametric curve generation
- `frontend/src/bead-renderer.js`: Three.js bead-string renderer
- `frontend/src/bead-difficulty-controller.js`: difficulty presets
- `backend/serve_frontend.py`: preview server for the frontend
- `backend/vite_server.py`: Vite dev-server launcher used by backend automation
- `backend/generate_samples.py`: headless dataset generation through the frontend batch API
- `backend/internvl3_benchmark.py`: standardized reasoning benchmark runner
- `backend/internvl3_config.py`: local/env/CLI config loader
- `backend/vlm_benchmark_bead.py`: shared task definitions, prompt logic, parsers, and scoring
- `backend/question_phrasings_bead.py`: canonical question text for the two tasks
- `output/`: generated samples written by `backend/generate_samples.py`

## Run

Start the frontend preview:

```bash
cd environments/order_bead_string
python3 backend/serve_frontend.py --port 8000
```

Then open:

```text
http://127.0.0.1:8000/order_bead_string/frontend/index.html
```

Or run the Vite dev server:

```bash
cd environments/order_bead_string/frontend
npm install
npm run dev
```

## Benchmark

Run a real model benchmark:

```bash
cd ..
python3 backend/internvl3_benchmark.py \
  --tasks all \
  --output-json output/internvl3_order_bead_string.json \
  --output-csv output/internvl3_order_bead_string.csv
```

Optional model selection sources, in precedence order:

- `--model ...`
- `.env` via `INTERNVL_MODEL` or `INTERNVL3_MODEL`
- `backend/internvl3_local_config.json` via `"model": "..."`

The benchmark runner is named `internvl3_benchmark.py`, but the request path is OpenAI-compatible. In practice you can point it at other compatible VLMs by changing `--base-url`, `--api-key`, and `--model`.

If you want to use a local config file, create `backend/internvl3_local_config.json` and omit `--config`, or pass a real custom path with `--config /abs/path/to/file.json`.

Run an oracle smoke test without calling the API:

```bash
cd ..
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
- `--max-tokens 300`
- `--output-json output/internvl3_order_bead_string.json`
- `--output-csv output/internvl3_order_bead_string.csv`
- `output/question.jsonl`
- `output/model_answer.jsonl`

## Outputs

The standardized benchmark writes:

- `output/internvl3_order_bead_string.json`
- `output/internvl3_order_bead_string.csv`
- `output/question.jsonl`
- `output/model_answer.jsonl`

This runner evaluates existing dataset images in place. The JSONL `images` fields therefore point back to files in the selected `--data-dir` using relative paths rather than copying images into the benchmark output directory.

The JSONL files follow the shared reasoning schema used across TopoBench tasks:

- `question.jsonl` stores `id`, `category`, `type`, `meta_info`, `question`, `images`, and `gt_answer` per sample
- `model_answer.jsonl` stores `id`, `category`, `type`, `meta_info`, `question`, `answer`, `correct`, `invalid_response`, `api_error`, and `raw_response_text` per sample

## Dataset Regeneration

Install frontend dependencies once:

```bash
cd environments/order_bead_string/frontend
npm install
```

Then generate samples through the frontend batch API:

```bash
cd ..
python3 backend/generate_samples.py \
  --samples-per-difficulty 217 \
  --difficulty easy,medium,hard \
  --cyclic-pairs 15 \
  --image-size 1024 \
  --workers 8
```

`--workers N` opens N parallel Playwright pages, each with its own three.js scene/WebGL context, and shards the per-difficulty index range by `i % N`. Cyclic-pair generation always runs on a single page after every shard finishes (it needs the full base-sample pool to sample from). Expect roughly 2–3× speedup on a typical laptop — the bottleneck is GPU/WebGL, not Python — so beyond ~4 workers there are diminishing returns. `--workers 1` is bit-equivalent to the previous integrated path for base samples; cyclic-pair output may differ slightly because the cyclic phase now uses its own seeded RNG.

The three difficulty tiers are:

- `easy`: moderate open curves (arc / s-curve / helix), 5–7 beads with mixed colors and partial occlusion (the previous "medium" recipe).
- `medium`: complex open / random-spline / helix curves, 8–10 beads with similar colors and heavy occlusion (the previous "hard" recipe).
- `hard`: torus-knot closed loops (`tangled_loop`, `T(p,q)`). The xy projection has multiple over/under crossings with bounded z separation, but the curve never self-intersects in 3D and has smooth (cusp-free) tangents — so a human can trace the bead order while a VLM has to reason about strand continuation through the crossings.

Default outputs:

- `output/dataset_metadata.json`
- `output/summary.json`
- `output/question.jsonl`
- `output/images/<question_id>/...png`

Notes:

- `generate_samples.py` writes images, `dataset_metadata.json`, `summary.json`, and `question.jsonl` in one pass.
- After question rows are built, images are reorganized so each row id has its own folder under `images/<question_id>/`. Pair-task rows (`T_BS02_pair_relationship`) carry both images side-by-side in their per-row folder.
- `--samples-per-difficulty` controls the number of base single-image samples per difficulty bucket.
- Final sample count is approximately `3 × samples-per-difficulty + actual_cyclic_pairs`.
- `T_BS01_describe_sequence` contributes one question per image, and the default `T_BS02_pair_relationship` expansion contributes about `ceil(N / 2)` pair questions, where `N` is the final sample count.
- So total `question.jsonl` rows are approximately `N + ceil(N / 2)`.
- `--cyclic-pairs` is a requested upper bound. The actual number can be slightly smaller because invalid cyclic rotations are skipped during generation.
- For about 1000 questions, `--samples-per-difficulty 217 --cyclic-pairs 15` usually lands at about `999` rows when all requested cyclic samples are realized, and stays close even if a few are skipped. Check `summary.json` for the exact final sample count.

The benchmark now reads these newly generated samples by default.

## Notes

- `T_BS01_describe_sequence` is a single-image task.
- `T_BS02_pair_relationship` is a two-image task and writes both image paths into the `images` list in `question.jsonl`.
- By default, pair generation now uses the minimum number of pair questions needed to cover every dataset image at least once, so each image participates in both task families unless you explicitly constrain the run with `--limit`.
- `order_bead_string` is now also registered under `lmms_eval_tasks/`, so once `output/question.jsonl` has been refreshed you can run it through the shared `lmms-eval` pipeline like the other static reasoning tasks.
