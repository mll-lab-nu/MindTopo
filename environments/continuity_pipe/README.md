# Continuity Pipe

`continuity_pipe` is a TopoBench interactive planning task in the `continuity` category.

The board is a square grid where some cells contain rotatable pipe pieces and some cells may be empty. One source pipe is green. Pipes connected to the source are green, while disconnected pipes are blue. At each turn, the agent chooses one non-empty pipe cell and that pipe rotates clockwise by 90 degrees. The task succeeds when every pipe is connected to the source.

The benchmark presets keep the solved pipe network as a tree: no loops and no four-way junctions are added. Difficulty is determined by a combination of board size, active pipe count, three-way branch junction count, and oracle solution length:

| Difficulty | Grid | Active pipes | Branch junctions | Oracle rotations |
| --- | ---: | ---: | ---: | ---: |
| `easy` | 4x4 | 10-13 | 2-3 | 13-17 |
| `medium` | 5x5 | 13-17 | 3-5 | 17-23 |
| `hard` | 5x5 | 17-23 | 5-7 | 23-27 |

The generator first builds a connected tree-shaped pipe network over a subset of grid cells, then rejects any sample with a four-way junction and scrambles non-empty pipe cells by applying seeded random rotations, so every sampled board has a solvable connected target.
Default generation uses `--difficulties easy,medium,hard`, `--repeats 1`, `--seed-start 1`, and writes one file at `continuity_pipe/output/question.jsonl`. The fixed benchmark set uses `--repeats 200`, producing 600 rows in that same JSONL. Legacy bounded `--solution-steps` ranges are still supported when `--grid-sizes` or `--solution-steps` is provided manually.
The first seed defaults to `1` via `--seed-start`; each `(difficulty, repeat)` or legacy `(grid_size, solution_steps, repeat)` uses the next integer seed. The frontend uses the same seeded sampler as the Python generator/evaluator, so the same seed and difficulty produce the same initial state. The action budget is `ceil(solution_ticks * 1.3)`.

## Run

Environment smoke test:

```bash
# From the repository root:
cd environments
uv run python continuity_pipe/gym/test_env.py --headless
```

Protocol smoke test:

```bash
# From the repository root:
cd environments
uv run python continuity_pipe/gym/test_protocol.py --headless
```

Generate interactive `question.jsonl` rows:

```bash
# From the repository root:
cd environments
uv run python continuity_pipe/backend/generate_samples.py \
  --difficulties easy,medium,hard \
  --repeats 200
```

Generate a specific difficulty bucket:

```bash
# From the repository root:
cd environments

# easy: 4x4, 10-13 pipes, 2-3 three-way branch junctions, 13-17 oracle rotations
uv run python continuity_pipe/backend/generate_samples.py \
  --difficulties easy \
  --repeats 10

# medium: 5x5, 13-17 pipes, 3-5 three-way branch junctions, 17-23 oracle rotations
uv run python continuity_pipe/backend/generate_samples.py \
  --difficulties medium \
  --repeats 10

# hard: 5x5, 17-23 pipes, 5-7 three-way branch junctions, 23-27 oracle rotations
uv run python continuity_pipe/backend/generate_samples.py \
  --difficulties hard \
  --repeats 10
```

Run a generated benchmark with the oracle solver:

```bash
# From the repository root:
cd environments
uv run python continuity_pipe/backend/internvl3_benchmark.py \
  --oracle \
  --question-jsonl continuity_pipe/output/question.jsonl
```

Run a small deterministic oracle smoke directly from difficulty presets:

```bash
# From the repository root:
cd environments
uv run python continuity_pipe/backend/internvl3_benchmark.py \
  --oracle \
  --repeats 1
```

Frontend preview:

```bash
# From the repository root:
cd environments
uv run python continuity_pipe/backend/serve_frontend.py --port 8000
```

Then open `http://127.0.0.1:8000/`. The standalone frontend supports clicking a pipe cell directly to rotate it.
The debug UI exposes `Grid size`, `Seed`, and `Difficulty`; generated state metadata reports difficulty, branch junction count, oracle rotations, and max steps.

## Outputs

The generator writes:

- `continuity_pipe/output/question.jsonl`

The standalone benchmark defaults to `continuity_pipe/output/internvl3_continuity_pipe.json`, `continuity_pipe/output/internvl3_continuity_pipe.csv`, `continuity_pipe/output/model_answer.jsonl`, and per-step artifacts under `continuity_pipe/output/{id}/`. When `--question-jsonl` is omitted it also writes a generated `continuity_pipe/output/question.jsonl`.

The benchmark writes:

- `continuity_pipe/output/model_answer.jsonl`
- per-step images and prompts under `continuity_pipe/output/{id}/`

The model answer format is:

```json
{"answer":{"x": 1, "y": 2}}
```

The x labels are shown above the grid and y labels are shown on the left side of the grid.
