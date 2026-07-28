# TopoBench

TopoBench is a benchmark for topology-aware visual reasoning. This minimal
snapshot keeps four runnable evaluation paths: static visual QA, interactive
planning, interleaved image-generation-in-the-loop evaluation, and a minimal
cached-video E2E evaluation across five topological categories: continuity,
enclosure, knots, order, and separation.

## What Is Included

| Category | Static reasoning tasks | Interactive planning tasks | Interleaved tasks | Cached-video E2E tasks |
| --- | --- | --- | --- | --- |
| Continuity | `continuity_2d_maze`, `continuity_3d_maze` | `continuity_pipe` | `continuity_2d_maze`, `continuity_pipe` | `continuity_pipe` |
| Enclosure | `enclosure_sheep`, `enclosure_hole_detection` | `enclosure_chat_noir` | `enclosure_sheep` | - |
| Knots | `knots_static` | `knots_untangle` | `knots_untangle` | `knots_untangle` |
| Order | `order_bead_string`, `order_origami` | `order_swap_2d_puzzle` | - | - |
| Separation | `separation_objects` | `separation_one_stroke` | `separation_one_stroke` | `separation_one_stroke` |

## Setup

Install the benchmark environment with `uv`:

```bash
uv sync --extra openai --extra lmms-eval --extra dev
```

For interactive planning tasks, install the browser runtime and build the task
frontends:

```bash
uv run python -m playwright install chromium
bash bin/setup_frontends.sh
```

For API-backed models, create a local `.env` from the template:

```bash
cp .env.example .env
```

Then add the provider keys you plan to use. Comma-separated key pools are
supported, for example:

```dotenv
INTERNVL_API_KEYS=key1,key2,key3
NVIDIA_NIM_API_KEYS=key1,key2
OPENAI_PAID_API_KEYS=key1
GOOGLE_PAID_API_KEYS=key1
```

No API key is required for the local planning baselines `oracle` and `random`.
Those local baselines also work as no-key passthrough checks for the interleaved
runner.

## Quick Start

Run one planning task with an API-backed model:

```bash
bash bin/run_planning_task.sh --env knots_untangle --model internvl
```

Run selected static tasks with the convenience wrapper:

```bash
bash bin/run_reasoning_task.sh \
  --env knots_static,enclosure_sheep,separation_objects \
  --model internvl
```

Run a no-key interleaved smoke test:

```bash
uv run python interleaved_eval_tasks/agent_runner.py \
  env=knots_untangle \
  model=oracle \
  run.manifest_limit=1 \
  run.output_dir=logs/smoke/interleaved_knots_oracle
```

Run one cached-video E2E episode:

```bash
uv run python video_eval_tasks/agent_runner.py \
  env=<env_name> \
  model=<video_model_id> \
  run.manifest_limit=1 \
  run.output_dir=logs/video_smoke/<run_label>
```

## Models

Model IDs are configured in `topobench_eval/providers.yaml` and
`planning_eval_tasks/configs/model/`.

Local planning baselines:

- `oracle`
- `random`

Remote/API-backed models in this snapshot include:

- `internvl`
- `qwen`
- `nemotron`
- `llama`
- `mistral_14b`
- `gemini_3_1_flash_lite`
- `gemini_3_1_pro`
- `gpt_5_4_mini`
- `gpt_5_5`
- `gpt_5_6_luna`
- `gemma`
- `cosmos`
- `bagel`
- `thinkmorph`

Self-hosted model IDs expect an OpenAI-compatible server and the matching
`*_BASE_URL` / `*_API_KEYS` values in `.env`.

## Outputs

Planning, interleaved, and video runs write generated artifacts under `logs/` by
default. These outputs are intentionally ignored by git and are not part of the
minimal repository.

Planning outputs include:

```text
logs/planning_eval/
├── summary.json
├── resolved_config.yaml
├── model_answer.jsonl
└── images/
```

Static evaluation outputs are written by `lmms-eval` through
`bin/topobench-eval`, normally under `lmms_eval_tasks/logs/`.

Interleaved outputs include the planning-style `model_answer.jsonl`,
`summary.json`, and imagined-image artifacts under
`logs/interleaved_eval/` unless `run.output_dir` is overridden.

Cached-video E2E outputs use the same planning-style summaries and store the
generated clip and extracted frames beside the episode screenshots.

## Repository Layout

```text
topobench-minimal/
├── environments/           # task implementations and fixed benchmark data
├── lmms_eval_tasks/        # static reasoning adapters for lmms-eval
├── planning_eval_tasks/    # Hydra + Playwright planning rollout runner
├── interleaved_eval_tasks/ # image-generation-in-the-loop runner
├── video_eval_tasks/       # one cached video per planning episode
├── topobench_eval/         # shared provider registry, parsing, and scoring
├── bin/                    # public command wrappers
├── pyproject.toml          # Python dependency source of truth
└── uv.lock
```

More detailed evaluation notes live in `topobench_eval/README.md`.

## Citation

Citation information will be added with the paper release.
