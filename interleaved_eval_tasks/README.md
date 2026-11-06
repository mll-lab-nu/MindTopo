# Interleaved Evaluation

This directory contains TopoBench's image-generation-in-the-loop evaluation
path. It reuses the planning runner and static answer parser, but wraps each
model call in a two-phase protocol:

1. The text model reasons over the current observation and emits an
   `<image_prompt>...</image_prompt>`.
2. The harness sends that prompt to the configured image-generation endpoint.
3. The text model sees the original observation plus the imagined image and
   commits to the final action or answer.

## Included Tasks

This minimal snapshot keeps five wired interleaved tasks:

| Mode | Tasks | Data source |
| --- | --- | --- |
| Planning rollout | `continuity_pipe`, `knots_untangle`, `separation_one_stroke` | `environments/<task>/output/question.jsonl` plus checked-in episode-id slices |
| Static QA | `enclosure_sheep`, `continuity_2d_maze` | `environments/<task>/output/question.jsonl` |

The planning tasks reuse `planning_eval_tasks/env_adapters.py`,
`RolloutRunner`, Playwright screenshots, and the task frontends. The static QA
tasks reuse `topobench_eval.answer_parser` for answer normalization and scoring.

## Layout

```text
interleaved_eval_tasks/
├── agent_runner.py        # Hydra entry point
├── interleaved_client.py  # phase 1 -> image generation -> phase 2 client
├── image_gen_client.py    # OpenAI-compatible image-generation client
├── reasoning_loader.py    # static question.jsonl loader
├── reasoning_runner.py    # static-QA evaluation loop
└── configs/
    ├── config.yaml
    ├── env/
    └── model/
```

## Setup

Install the Python environment and build the interactive frontends from the repo
root:

```bash
uv sync --extra openai --extra lmms-eval --extra dev
uv run python -m playwright install chromium
bash bin/setup_frontends.sh
```

For API-backed interleaved models, create `.env` from `.env.example` and set the
text-model and image-generation provider keys required by the model config.
For planning tasks, local passthrough models such as `oracle` and `random` do
not require keys and bypass image generation. Static QA tasks require a model
client; these local policies do not provide static answers.

## Quick Checks

Run a no-key planning smoke test:

```bash
uv run python interleaved_eval_tasks/agent_runner.py \
  env=knots_untangle \
  model=oracle \
  run.manifest_limit=1 \
  run.output_dir=logs/smoke/interleaved_knots_oracle
```

Run one API-backed interleaved episode:

```bash
uv run python interleaved_eval_tasks/agent_runner.py \
  env=<env_name> \
  model=<model_id>_imagined \
  run.manifest_limit=1 \
  run.step_budget=10 \
  run.output_dir=logs/smoke/interleaved_<run_label>
```

Planning env configs may select a checked-in subset through
`env.episode_ids_path`. To compare an interleaved result against its required
VLM-only baseline, use the same model family, env, episode-ID file, and step
budget:

```bash
uv run python planning_eval_tasks/agent_runner.py \
  env=<env_name> model=<model_id> \
  +env.episode_ids_path=<episode_ids_path> \
  run.output_dir=logs/baseline/<run_label>
```

Set `env.episode_ids_path=null` to use the complete source pool when the env
config defines a subset by default.

Run a static-QA interleaved sample:

```bash
uv run python interleaved_eval_tasks/agent_runner.py \
  env=<perception_env> \
  model=<model_id>_imagined \
  run.manifest_limit=1 \
  run.output_dir=logs/smoke/interleaved_<run_label>
```

The static-QA runner skips rows whose referenced images are missing. If all rows
are missing images, it raises instead of producing an empty score. This minimal
snapshot keeps only `question.jsonl`, so restore or regenerate referenced images
before running `enclosure_sheep` or `continuity_2d_maze` in interleaved static
QA mode.

## Useful Overrides

| Override | Purpose |
| --- | --- |
| `env=<name>` | Select one of `continuity_pipe`, `knots_untangle`, `separation_one_stroke`, `enclosure_sheep`, `continuity_2d_maze` |
| `model=<id>` | Select a config from `configs/model/` |
| `run.manifest_limit=N` | Cap episodes or samples for smoke runs |
| `run.manifest_offset=N` | Skip the first N rows |
| `run.step_budget=N` | Cap planning rollout steps |
| `run.parallel_sessions=N` | Run multiple planning workers |
| `run.output_dir=PATH` | Choose the artifact directory |
| `env.source_question_jsonl=PATH` | Override planning task row source |
| `env.episode_ids_path=PATH` | Select and order a fixed planning subset |
| `env.question_jsonl=PATH` | Override static-QA row source |

## Outputs

Interleaved runs write:

```text
logs/interleaved_eval/
├── summary.json
├── resolved_config.yaml
├── model_answer.jsonl
└── images/ or source/
```

Planning tasks store imagined images next to the environment screenshots.
Static-QA tasks store source and imagined images under
`<output_dir>/source/<sample_id>/`. The relative paths are also recorded in each
`model_answer.jsonl` row.

Resume is automatic when the same `run.output_dir` is reused. Completed IDs in
`model_answer.jsonl` are skipped.
