# TopoBench

TopoBench is a benchmark for topology-aware visual reasoning. This minimal
snapshot keeps four evaluation paths: static visual QA, interactive
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
OPENAI_API_KEYS=key1
GOOGLE_PAID_API_KEYS=key1
```

No API key is required for the local planning baselines `oracle` and `random`.
Those local baselines also work as no-key passthrough checks for interleaved
planning tasks; interleaved static QA requires a model client.

## Core Set and Extended Set

The release has two disjoint data partitions across all 13 tasks:

| Partition | Instances | Purpose |
| --- | ---: | --- |
| `core` | 3,900 | Primary paper evaluation, 300 per task, with 150 Easy and 150 Hard |
| `extended` | 8,780 | Additional archived instances after exact content deduplication |

Static reasoning and planning default to `core`. Select `--split extended`
explicitly for the Extended Set. Core difficulty is stored as
`meta_info.benchmark_level`; the original generator's `difficulty` and reset
configuration are preserved for faithful execution. Extended keeps its source
difficulty labels. The 150 older Origami variants with reused IDs are retained
with `legacy:` release IDs and their original `meta_info.source_id`.

Canonical JSONL files live under `data/<split>/<task>/question.jsonl`, and
`data/manifests/` records the fixed paper lists, content hashes and counts.
Images live under `data/assets/` with content-addressed filenames. They are
excluded from Git and must be restored with the matching asset package.
Missing or corrupt images stop evaluation before provider initialization.
They never silently reduce the benchmark sample count.

The authors can reconstruct the local release from its recorded original
exports without regenerating any scenes or labels:

```bash
python bin/export_release.py \
  --paper-root /path/to/topobench_arxiv \
  --benchmark-root /path/to/topobench
uv run python bin/check_release.py
```

`bin/check_release.py --skip-images` validates JSONL and manifests in a clone
without image assets. The normal check verifies every image hash as well.

Interleaved and video defaults retain the existing `legacy_probing` protocol.
Their frozen question snapshots are in `data/probing/`, and the original
105-ID lists remain in `data/slices/balanced_105/`. These protocol views are
separate from the two public data partition choices. To run a new Core or
Extended probing experiment through Hydra, set `data.split=core` or
`data.split=extended` and explicitly clear `env.episode_ids_path=null` where
a balanced-105 list is configured. Explicit custom `env.source_question_jsonl`
paths remain supported and are labelled `custom`, rather than Core/Extended.

The legacy files under `environments/<task>/output/` are retained for old
generation/examples. Public benchmark commands load the canonical data paths
above. Updating the main default does not replace the probing question sets.

## Quick Start

Run one planning task with an API-backed model:

```bash
bash bin/run_planning_task.sh --env knots_untangle --model internvl --split core
```

Run selected static tasks with the convenience wrapper:

```bash
bash bin/run_reasoning_task.sh \
  --env knots_static,enclosure_sheep,separation_objects \
  --model internvl --split extended
```

Validate main evaluation inputs without API keys or model calls:

```bash
bash bin/run_reasoning_task.sh --check --env order_origami --split core
bash bin/run_planning_task.sh --check --model oracle --env order_swap_2d_puzzle --split extended
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
  run.output_dir=logs/video_eval/<env_name>/<run_label>
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
logs/planning_eval/<split>/<model>/<task>/
├── summary.json
├── resolved_config.yaml
├── model_answer.jsonl
└── images/
```

Static evaluation outputs are written by `lmms-eval` through
`bin/topobench-eval`, normally under `lmms_eval_tasks/logs/<split>/`. Caches are separated by split too.

Interleaved outputs include the planning-style `model_answer.jsonl`,
`summary.json`, and imagined-image artifacts under
`logs/interleaved_eval/<protocol>/` unless `run.output_dir` is overridden.

Cached-video E2E outputs use the same planning-style summaries and store the
generated clip and extracted frames beside the episode screenshots. The same
run directory is the direct input to the video-metrics evaluator
(`uv run python -m video_eval_tasks.video_metrics evaluate --run-dir <dir>`),
which writes its report to `<dir>/reports/video_metrics/` — see
`video_eval_tasks/README.md`.

## Repository Layout

```text
topobench-minimal/
├── data/                   # core, extended, probing, manifests, and ignored assets
├── environments/           # task implementations and legacy generator exports
├── lmms_eval_tasks/        # static reasoning adapters for lmms-eval
├── planning_eval_tasks/    # Hydra + Playwright planning rollout runner
├── interleaved_eval_tasks/ # image-generation-in-the-loop runner
├── video_eval_tasks/       # one cached video per planning episode
│   ├── video_metrics/      #   CV-based scoring of completed video runs
│   └── cv_backend/         #   internal task-specific CV detectors
├── topobench_eval/         # shared provider registry, parsing, and scoring
├── bin/                    # public command wrappers
├── pyproject.toml          # Python dependency source of truth
└── uv.lock
```

More detailed evaluation notes live in `topobench_eval/README.md`.

## License

This project is released under the MIT License; see [LICENSE](LICENSE) for details.

## Citation

Citation information will be added with the paper release.
