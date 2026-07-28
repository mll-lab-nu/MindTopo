# TopoBench Evaluation Guide

This directory contains the shared model registry, answer contracts, parsing,
and scoring utilities used by the retained benchmark pipelines:

| Pipeline | Tasks | Entry point |
| --- | --- | --- |
| Static reasoning | 8 VQA-style tasks | `bin/topobench-eval` |
| Interactive planning | 5 gym-style tasks | `planning_eval_tasks/agent_runner.py` |
| Interleaved evaluation | 4 image-generation-in-the-loop tasks | `interleaved_eval_tasks/agent_runner.py` |

## API Keys

The pipelines load `.env` from the repository root. Key pools may be
comma-separated:

```dotenv
INTERNVL_API_KEYS=key1,key2,key3
NVIDIA_NIM_API_KEYS=key1,key2
OPENAI_PAID_API_KEYS=key1
GOOGLE_PAID_API_KEYS=key1
```

Provider routing lives in `providers.yaml`. Planning model configs live in
`../planning_eval_tasks/configs/model/`. Interleaved composite model configs
live in `../interleaved_eval_tasks/configs/model/`.

## Static Reasoning

Static tasks are exposed through `lmms-eval` under the `topobench` task group.
The retained tasks are:

- `continuity_2d_maze`
- `continuity_3d_maze`
- `enclosure_hole_detection`
- `enclosure_sheep`
- `knots_static`
- `order_bead_string`
- `order_origami`
- `separation_objects`

Example:

```bash
./bin/topobench-eval \
  --num-cpus 1 \
  --workers-per-key 1 \
  --max-shard-workers 1 \
  --no-adaptive-concurrency \
  internvl knots_static --limit 1
```

The static task adapters read fixed data from
`../environments/<task>/output/question.jsonl` and the image paths referenced by
those rows. This minimal snapshot keeps the `question.jsonl` files only; restore
or regenerate the referenced image files before running static visual QA.

## Interactive Planning

Planning tasks are driven by Hydra configs and Playwright-backed environments.
The retained tasks are:

- `knots_untangle`
- `continuity_pipe`
- `separation_one_stroke`
- `order_swap_2d_puzzle`
- `enclosure_chat_noir`

No-key smoke test:

```bash
.venv/bin/python planning_eval_tasks/agent_runner.py \
  env=knots_untangle \
  model=oracle \
  run.sample_limit=1 \
  run.output_dir=logs/smoke/knots_oracle
```

Useful overrides:

| Override | Purpose |
| --- | --- |
| `run.sample_limit=1` | Run only the first episode |
| `run.sample_offset=N` | Skip the first N episodes |
| `run.parallel_sessions=N` | Set worker count manually |
| `run.output_dir=...` | Choose an explicit output directory |

## Interleaved Evaluation

Interleaved evaluation runs a two-phase loop: the model first reasons and
requests an imagined image, the harness generates that image, and the model then
answers using both the original observation and the imagined image. The retained
interleaved tasks are:

- `knots_untangle`
- `separation_one_stroke`
- `enclosure_sheep`
- `continuity_2d_maze`

No-key passthrough smoke test:

```bash
.venv/bin/python interleaved_eval_tasks/agent_runner.py \
  env=knots_untangle \
  model=oracle \
  run.manifest_limit=1 \
  run.output_dir=logs/smoke/interleaved_knots_oracle
```

API-backed interleaved model configs use the suffix `_imagined`, for example
`gpt_5_4_mini_imagined` or `internvl3_5_imagined`. Those runs require both the
text-model key and the configured image-generation key.

## Shared Modules

| Module | Responsibility |
| --- | --- |
| `provider_registry.py` | Model ID to provider/base URL/key-env resolution |
| `answer_contract.py` | Legal answer domains |
| `contract_resolver.py` | Dataset row to answer contract |
| `structured_output.py` | JSON/fence/integer extraction primitives |
| `answer_parser.py` | Answer parsing orchestration |
| `answer_scoring.py` | Canonicalization and equality |

Generated outputs are written under `logs/` or `lmms_eval_tasks/logs/` and are
not part of the minimal repository.
