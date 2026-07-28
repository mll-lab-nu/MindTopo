# Continuity 3D Maze

`continuity_3d_maze` is a TopoBench reasoning task. It uses the ProcTHOR/AI2-THOR renderer to generate furnished houses as 3D mazes, samples labeled valid navigation points, renders five fixed views, and asks connectivity questions under the visible door states.

## Question Types

There are three CLI question modes controlled by `--question-types`:

- `subset_choice`: one 4-choice question. Each option is a list of two or more labeled points, and exactly one option lists points that are all mutually connected. The correct option position is shuffled with the scene RNG, so it is reproducible but not fixed to option `1`.
- `point_list`: one question for one target point in the scene. The target point is selected by the scene RNG, so it is reproducible with the same seed. The answer is the list of all other labeled points connected to that target point.
- `door_open`: one question for two target points that are not connected under the current door states. Generation first closes every actual openable door in the scene, finds a room-door graph with a unique minimum closed-door path, then places the target points in the endpoint rooms and verifies the answer with AI2-THOR navigation. The answer is a unique minimum set of 1-4 currently closed door colors. The `door_open` images color the currently closed candidate doors with distinct colors.

Accepted `--question-types` values are comma-separated subsets of:

- `subset_choice`
- `point_list`
- `door_open`

The default is `point_list`. In `--question-jsonl` benchmark mode, `--question-types` is only a filter over rows already present in the JSONL. Run `door_open` explicitly when you want the door-opening task.

Canonical answer formats:

```json
{"answer":"1"}
```

```json
{"answer":{ans}}
```

For `point_list`, replace `{ans}` with the actual JSON array of connected point names, for example `["B","D"]` or `[]`.

For `door_open`, replace `{ans}` with the actual JSON array of required door color names, for example `["blue","red"]`.

## Difficulty

For `point_list` and `subset_choice`, difficulty is defined directly by the requested `(room_count, door_count, point_count)` setup:

| level | setup tuples |
| --- | --- |
| easy | `(3,2,4)` |
| medium | `(4,3,4)`, `(5,4,4)` |
| hard | `(6,5,5)`, `(7,6,5)` |

For `door_open`, `--difficulty` uses a separate setup table:

| level | setup tuples |
| --- | --- |
| easy | `(4,3,4)` |
| medium | `(5,4,4)` |
| hard | `(6,5,5)` |

Passing `--difficulty easy`, `--difficulty medium`, or `--difficulty hard` makes generate/direct-benchmark mode sample one setup tuple per scene from that level using `--seed`. If `--difficulty` is omitted, the scripts keep the explicit `--room-counts * --door-counts * --point-counts * --repeats` Cartesian-product behavior.

`point_count` must be at least `3` in generate and benchmark code. The recommended dataset plan is to run each level separately and append all rows into one combined `output/question.jsonl`:

| difficulty | point/subset setup tuples | door_open setup tuples | repeats | scenes |
| --- | --- | ---: | ---: | ---: |
| easy | `(3,2,4)` | `(4,3,4)` | 150 | 150 |
| medium | `(4,3,4)`, `(5,4,4)` | `(5,4,4)` | 200 | 200 |
| hard | `(6,5,5)`, `(7,6,5)` | `(6,5,5)` | 300 | 300 |

With the current defaults, `--room-counts 4 --door-counts 2 --point-counts 3` is a custom setup unless `--difficulty` is set.

## Reproducibility

The default seed is `12345`. The scripts initialize one RNG from `--seed`, derive one start seed per generated scene, and then use the scene RNG for door states, point sampling, the `point_list` target point, and `subset_choice` option shuffling. `door_open` is deterministic under the same seed as well, but it always uses all-closed door states before planning the door-opening question.

Generation order is deterministic:

- first room/door/point setup
- all repeats for that setup
- next room/door/point setup

For example, `--room-counts 3,4 --door-counts 2 --point-counts 3 --repeats 2` generates:

- rooms 3, doors 2, points 3: repeats 1-2
- rooms 4, doors 2, points 3: repeats 1-2

`--workers` defaults to `1`. Generate mode and benchmark direct-generation mode both support `--workers > 1` with the default resilient candidate pool. Candidate seeds are assigned before workers run, each wave is resolved by candidate index, and the final JSONL rows are selected in deterministic seed order rather than completion order. In `--question-jsonl` mode, no environment generation happens, so `--workers` does not affect the loaded samples.

Oracle and JSONL-based evaluation are reproducible. Real model API outputs may still vary because the remote model/provider can change behavior and the script does not set sampling controls such as temperature.

## Debug Preview

The debug UI is the local Tk preview app:

```bash
cd /home/andrew/Desktop/topobench/environments/continuity_3d_maze
conda activate swing
python preview_house.py
```

The preview exposes `Seed`, `Room Count`, `Controllable Doors`, and `Random Point Count`. Its default seed is `12345`. The preview is for visual/debug inspection of maze points and pairwise connectivity; it does not use `--question-types` and still allows 2-point debug connectivity even though formal generate/benchmark samples require at least 3 points.

For a non-interactive one-house render:

```bash
python render_one_house.py \
  --count 1 \
  --start-seed 12345 \
  --room-count 4 \
  --door-count 2 \
  --door-state random \
  --output-dir outputs
```

## Generate Question JSONL

`generate_samples.py` renders scenes, writes images, and appends successful question rows to `--output-json`. This is useful for building one combined JSONL in several runs, such as easy first, then medium, then hard. Existing JSONL row ids are treated as already generated: rerunning the same command against the same output path will not append duplicate rows or overwrite the saved images/house files for those ids. Remove the old JSONL and image directory first if you want a clean dataset.

By default, generation uses `--scene-timeout-sec 180 --skip-failed-scenes --max-attempts 1 --platform CloudRendering`. Each candidate seed runs in an isolated subprocess; if AI2-THOR/ProcTHOR hangs or fails, that candidate is skipped and the setup keeps sampling deterministic replacement seeds until it reaches `--repeats` successful scenes. With `--workers > 1`, candidate seeds are tried in parallel but accepted in candidate-index order, so the same command and seed remain reproducible. Only successful generated samples are written to `question.jsonl`.

With the default `--question-types point_list`, each rendered scene emits one JSONL row. The three commands below produce:

- easy: 150 scenes, 150 rows
- medium: 200 scenes, 200 rows
- hard: 300 scenes, 300 rows
- total: 650 scenes, 650 rows

Generate easy data:

```bash
python backend/generate_samples.py \
  --difficulty easy \
  --repeats 150
```

Generate medium data:

```bash
python backend/generate_samples.py \
  --difficulty medium \
  --repeats 200
```

Generate hard data:

```bash
python backend/generate_samples.py \
  --difficulty hard \
  --repeats 300
```

Default generate args:

- `--room-counts 4`
- `--door-counts 2`
- `--point-counts 3`
- `--question-types point_list`
- `--difficulty ""`
- `--repeats 1`
- `--seed 12345`
- `--split train`
- `--door-state random`
- `--max-attempts 1`
- `--platform CloudRendering`
- `--render-width 1920`
- `--render-height 1080`
- `--render-quality Ultra`
- `--workers 1`
- `--scene-timeout-sec 180`
- `--skip-failed-scenes true`
- `--max-seed-multiplier 10`
- `--output-json output/question.jsonl`
- `--image-root output/images`

Each image folder is named with the corresponding JSONL data `id`:

```text
output/images/{data_id}/topdown.png
```

Each generated sample contains five images: `topdown`, `front45`, `right45`, `rear45`, and `left45`.

Question row counts per rendered scene:

- `--question-types subset_choice`: 1 row
- `--question-types point_list`: 1 row
- `--question-types door_open`: 1 row
- `--question-types point_list,subset_choice`: 2 rows
- `--question-types point_list,door_open`: 2 rows
- `--question-types point_list,subset_choice,door_open`: 3 rows

## Benchmark

`internvl3_benchmark.py` has two modes.

Load existing rendered samples:

```bash
python backend/internvl3_benchmark.py \
  --config backend/internvl3_local_config.json \
  --question-jsonl output/question.jsonl \
  --output-json output/internvl3_continuity_3d_maze.json \
  --output-csv output/internvl3_continuity_3d_maze.csv \
  --image-root output/images \
  --max-tokens 64
```

In this mode, the script does not call AI2-THOR or generate a new environment. It loads `question`, `images`, and `gt_answer` from the JSONL, resolves the image files, sends the stored prompt/images to the model, and writes fresh results. If `--question-types` is omitted, every row in the JSONL is evaluated; pass `--question-types door_open` or another comma-separated subset only when you want to filter. The source `--question-jsonl` is not rewritten; the benchmark summary, CSV, and `model_answer.jsonl` in the output directory are overwritten for the evaluated rows.

Run direct generation plus benchmark:

```bash
python backend/internvl3_benchmark.py \
  --config backend/internvl3_local_config.json \
  --difficulty easy \
  --repeats 1 \
  --max-tokens 64
```

In direct-generation mode, the script generates scenes and immediately evaluates them. It uses the same default generation stability settings as `generate_samples.py`: `--scene-timeout-sec 180 --skip-failed-scenes --max-attempts 1 --platform CloudRendering`. With `--workers > 1`, scene candidates are generated in parallel and then evaluated in deterministic candidate order. It upserts `question.jsonl` and `model_answer.jsonl`: matching sample ids are replaced, and new sample ids are appended.

Use `--oracle` for a no-API sanity check:

```bash
python backend/internvl3_benchmark.py \
  --oracle \
  --question-jsonl output/question.jsonl \
  --output-json output/internvl3_continuity_3d_maze.json \
  --output-csv output/internvl3_continuity_3d_maze.csv \
  --image-root output/images
```

Default benchmark args:

- `--room-counts 4`
- `--door-counts 2`
- `--point-counts 3`
- `--question-types point_list`
- `--difficulty ""`
- `--repeats 1`
- `--seed 12345`
- `--split train`
- `--door-state random`
- `--max-attempts 1`
- `--platform CloudRendering`
- `--render-width 1920`
- `--render-height 1080`
- `--render-quality Ultra`
- `--workers 1`
- `--scene-timeout-sec 180`
- `--skip-failed-scenes true`
- `--max-seed-multiplier 10`
- `--question-jsonl ""`
- `--max-tokens 0`
- `--oracle false`
- `--output-json output/internvl3_continuity_3d_maze.json`
- `--output-csv output/internvl3_continuity_3d_maze.csv`
- `--image-root output/images`

API config can come from CLI flags, environment variables, or a config file. The priority is CLI value, then environment variable, then config file, then code default. If `--config` is omitted, the script looks for `backend/internvl3_local_config.json`.

Example config:

```json
{
  "api_key": "...",
  "base_url": "...",
  "model": "internvl3.5-latest"
}
```

## JSONL And Prompt Format

Generated rows follow the meta JSONL shape:

```json
{
  "id": "...",
  "category": ["continuity", "continuity_3d_maze", "connected_point_list"],
  "type": "reasoning",
  "meta_info": {...},
  "question": "[Task]...",
  "images": ["images/.../topdown.png", "..."],
  "gt_answer": []
}
```

The `question` field is a meta prompt with `[Task]`, `[Image n]`, `[Rules]`, `[Question]`, and `[Answer Format]` sections. The benchmark can split this prompt around the image markers and send it as one multi-image user message.

The model request uses OpenAI-compatible Chat Completions content parts:

- one text part before the images
- five `image_url` parts containing base64 data URLs
- one text part after the images

## Answer Parsing

`subset_choice` ground truth is a string option label such as `"4"`. The parser accepts JSON answers, embedded JSON objects, `option 4` / `choice 4`, bare `"4"`, and as a final fallback scans from the end of the response for a legal option number.

`point_list` ground truth is a JSON array such as `[]` or `["B","D"]`. Evaluation uses set equality over parsed point-name lists, so order does not matter. The parser accepts JSON answers, embedded JSON objects, boxed answers, and as a final fallback scans from the end of the response for the last bracketed list.

`door_open` ground truth is a JSON array of door color names such as `["blue","red"]`. Evaluation uses set equality over parsed color-name lists, so order does not matter.

## Notes

- This task is reasoning-only, so there is no `gym/` wrapper.
- If a strict room/door setup is rare, increase `--workers` for faster parallel candidate search or increase `--max-seed-multiplier` to allow more replacement seeds. Increase `--max-attempts` only if you specifically want each candidate seed to try neighboring house seeds before being skipped.
- `--max-tokens` defaults to `0`; for real model API evaluation, pass a positive value such as `--max-tokens 64`.
- The renderer is AI2-THOR/ProcTHOR rather than Three.js because the environment depends on physically valid furnished indoor navigation and live door-state reachability.
