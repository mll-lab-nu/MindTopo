# Separation Objects

`separation_objects` is a TopoBench reasoning task. Each data item contains one complete-object image plus five option images. The complete-object image combines front upper right and back upper left 45-degree oblique views. The model chooses which option is the correct decomposition of the complete object.

Canonical answer format:

```json
{"answer":"A"}
```

## Difficulty

Difficulty is defined by the complete object's primitive `part_count`:

- easy: `part_count` 3-5
- medium: `part_count` 6-10
- hard: `part_count` 11+

There is no separate `difficulty` argument and no `part_count` argument. The `part_count` is the number of primitive components in the complete catalog object, and it is fixed for each object. Every option shows a two-part decomposition.

Option generation uses the `revised` mode only. The legacy and three-subassembly modes are disabled in the frontend, generation script, and benchmark script.

The `revised` mode keeps the same task format but uses harder distractors: two same-category component-replacement options built from different target partitions, one target option with one component removed from a different target partition, and one target option with one same-category component added to a different target partition. Here, same-category means another catalog object in the same category with a different object name; the other object's primitive part count does not need to match the target object's part count. The final A-E letters are shuffled, so the correct answer is not always A.

The target and distractor object pool is also filtered by complete-object visibility. Across the two fixed complete-object views, every primitive part must have at least `min(5 pixels, 5% of that part's solo rendered area)` visible. This keeps 85 eligible objects and excludes objects whose complete view hides at least one primitive part.

Generated datasets and direct-generation benchmark runs also apply an option-level visibility filter. After a scene is rendered, every primitive part that appears in each option subassembly must have at least `min(13 pixels, 13% of that part's solo rendered area)` visible in that option view. Scenes that fail this check are rejected and regenerated from the next scene seed.

Seed behavior is deterministic. The scripts initialize one RNG with `--seed 12345` by default, then keep consuming generated scene seeds from that same RNG. Accepted samples use the first subsequent scene seed whose option images pass the visibility filter; rejected scene seeds are skipped, and the RNG is not reset between objects. With `--category Bench,Chair --repeats 2`, generation order is:

- every eligible Bench object: repeats 1-2
- every eligible Chair object: repeats 1-2

With no `--category` and no `--object-name`, all eligible catalog objects are generated, and each object is repeated `--repeats` times.

## Frontend Debug

```bash
cd environments/separation_objects
conda activate ptm
python backend/serve_frontend.py --port 8000
```

Open:

```text
http://127.0.0.1:8000/separation_objects/frontend/index.html
```

The debug page exposes `Category`, `Object`, `Seed`, and a fixed disabled `Option Mode = Revised` selector. The displayed part count is read from the selected catalog object.

## Generate Question JSONL

Generate all eligible objects, each repeated once:

```bash
python backend/generate_samples.py
```

Generate all eligible objects, each repeated 10 times:

```bash
python backend/generate_samples.py \
  --repeats 10 \
  --seed 12345
```

Generate one easy object (`part_count` 3-5), repeated 100 times:

```bash
python backend/generate_samples.py \
  --category Bench \
  --object-name applaro \
  --repeats 100 \
  --seed 12345 \
  --output-json output/question_easy.jsonl \
  --images-dir output/images
```

Generate one medium object (`part_count` 6-10), repeated 100 times:

```bash
python backend/generate_samples.py \
  --category Bench \
  --object-name applaro_2 \
  --repeats 100 \
  --seed 12345 \
  --output-json output/question_medium.jsonl \
  --images-dir output/images
```

Generate one hard object (`part_count` 11+), repeated 100 times:

```bash
python backend/generate_samples.py \
  --category Bench \
  --object-name applaro_3 \
  --repeats 100 \
  --seed 12345 \
  --output-json output/question_hard.jsonl \
  --images-dir output/images
```

Generate every eligible object in one category, each repeated 10 times:

```bash
python backend/generate_samples.py \
  --category Bench \
  --repeats 10 \
  --seed 12345
```

Default generate args:

- `--repeats 1`
- `--seed 12345`
- `--option-mode revised`
- `--category ""`
- `--object-name ""`
- `--output-json output/question.jsonl`
- `--images-dir output/images`
- `--headless` enabled by default

## Benchmark

Load an existing generated question JSONL:

```bash
python backend/internvl3_benchmark.py \
  --oracle \
  --question-jsonl output/question.jsonl \
  --output-json output/internvl3_separation_objects.json \
  --output-csv output/internvl3_separation_objects.csv \
  --image-root output/images
```

Run direct generation plus benchmark:

```bash
python backend/internvl3_benchmark.py \
  --oracle \
  --repeats 1 \
  --seed 12345
```

Direct benchmark uses the same target traversal as `generate_samples.py`: no filters means every eligible object, `--category` means every eligible object in that category, and `--category ... --object-name ...` means one object. Each selected object is repeated `--repeats` times.

Default benchmark args:

- `--repeats 1`
- `--seed 12345`
- `--option-mode revised`
- `--category ""`
- `--object-name ""`
- `--max-tokens 0`
- `--output-json output/internvl3_separation_objects.json`
- `--output-csv output/internvl3_separation_objects.csv`
- `--image-root output/images`
- `--headless` enabled by default

Benchmark output includes:

- full JSON results
- full CSV results
- `_by_category.json/csv` and `_by_object.json/csv`
- `question.jsonl`
- `model_answer.jsonl`
- per-sample images and prompt files under `output/images/{data_id}/`

## Smoke Test

```bash
python backend/playwright_smoke.py --headless
```
