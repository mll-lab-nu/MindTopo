# Cached Video E2E Evaluation

This directory contains TopoBench's minimal video-generation-in-the-loop path.
It reuses the planning runner and action parsers, but wraps each episode in a
two-phase protocol:

1. The text model reads the initial observation and emits a
   `<video_prompt>...</video_prompt>`.
2. The configured video backend generates one image-conditioned clip.
3. The harness extracts frames with `ffmpeg`.
4. The text model uses the live observation and cached frames to commit one
   action per environment step.

The clip is generated once per episode and reused on subsequent steps. The
existing `RolloutRunner` remains responsible for stepping, parsing, resume,
summaries, and artifacts.

## Included Tasks

| Mode | Tasks | Data source |
| --- | --- | --- |
| Planning rollout | `continuity_pipe`, `knots_untangle`, `separation_one_stroke` | `environments/<task>/output/question.jsonl` plus optional episode-ID slices |

The minimal implementation supports the Ark asynchronous task API. Direct
video-only evaluation, multi-video cycles, and additional provider adapters are
outside this snapshot.

## Layout

```text
video_eval_tasks/
├── agent_runner.py                 # Hydra entry point
├── video_interleaved_client.py     # phase 1 -> video -> frames -> phase 2
├── video_gen_client.py             # small backend factory and result type
├── video_gen_client_ark.py         # Ark submit, poll, and download client
├── frame_extractor.py              # mp4 -> evenly spaced PNG frames
└── configs/
    ├── config.yaml
    ├── env/
    └── model/
```

## Setup

Install the normal repository dependencies, Playwright Chromium, task
frontends, `ffmpeg`, and `ffprobe`. Create `.env` from `.env.example` and set
the text-model and video-provider keys referenced by the selected model config.

## Quick Check

```bash
uv run python video_eval_tasks/agent_runner.py \
  env=<env_name> \
  model=<video_model_id> \
  run.manifest_limit=1 \
  run.output_dir=logs/video_eval/<env_name>/<run_label>
```

## Useful Overrides

| Override | Purpose |
| --- | --- |
| `env=<name>` | Select a config from `configs/env/` |
| `model=<id>` | Select a config from `configs/model/` |
| `run.manifest_limit=N` | Cap episodes for smoke runs |
| `run.manifest_offset=N` | Skip the first N selected episodes |
| `run.step_budget=N` | Clamp the environment step budget |
| `run.num_extracted_frames=N` | Choose how many clip frames phase 2 receives |
| `run.output_dir=PATH` | Choose the artifact and resume directory |
| `env.source_question_jsonl=PATH` | Override the source question pool |
| `env.episode_ids_path=PATH` | Select and order a fixed subset |

Set `env.episode_ids_path=null` to use the complete source pool when an env
config defines a subset by default.

## Outputs

Cached-video runs write the planning-style `summary.json`,
`model_answer.jsonl`, `resolved_config.yaml`, prompt artifacts, environment
screenshots, one imagined MP4 per episode, and its extracted frames.

Resume is automatic when the same `run.output_dir` is reused. Completed IDs in
`model_answer.jsonl` are skipped.

## Video metrics

Score a completed cached-video run with the CV-based evaluator. The input is
the run directory the runner wrote — no copying or renaming:

```bash
uv sync --extra video-metrics
uv run python -m video_eval_tasks.video_metrics evaluate \
  --run-dir logs/video_eval/<env_name>/<run_label> \
  --overlays failures
```

The evaluator reads only `<run-dir>/model_answer.jsonl`. Task identity comes
from `meta_info.task_name`, generated clips from `trajectory[].imagined_video`
(a path relative to the run dir), and oracle inputs from the configured source
question JSONL, falling back to the tracked
`environments/<task>/output/question.jsonl`. Absolute or traversing video
paths are rejected; the evaluator never scans directories.

Wired tasks in this snapshot: `continuity_pipe`, `knots_untangle`, and
`separation_one_stroke`.

The report is replaced atomically at `<run-dir>/reports/video_metrics/`:

```text
reports/video_metrics/
├── report.json          # versioned machine interface
├── report.csv           # one row per generated video
├── report.md            # human summary
├── manual_review.csv    # ordered by verifier confidence (never changes scores)
├── episodes/<episode-id>.json
├── overlays/<episode-id>/{static,dynamic}/...
└── artifacts/...
```

The primary score is `overall_video_accuracy`: outcome, static, and dynamic
checks must all pass; unscorable videos count as failures.
`conditional_overall_accuracy` and `scorable_rate` are reported separately so
detector failures stay visible. Use `--output-dir` to relocate the tree and
`--overlays none|failures|all` to control evidence generation.

Cross-run reporting:

```bash
uv run python -m video_eval_tasks.video_metrics aggregate \
  --runs logs/video_eval/<env>/<run_a> logs/video_eval/<env>/<run_b> \
  --output-dir logs/reports/<benchmark_name>
```

Loose historical MP4s (not produced by this runner) enter only through an
explicit JSONL manifest; the importer copies them into a standard run layout:

```bash
uv run python -m video_eval_tasks.video_metrics import-manifest \
  --manifest path/to/videos.jsonl --output-run-dir logs/video_eval/imported/<name>
```

The task-specific CV detectors live in `video_eval_tasks/cv_backend/` — an
internal backend, not a second CLI; always enter through the evaluator above.
