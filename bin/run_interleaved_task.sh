#!/usr/bin/env bash
# One-line launcher for the interleaved (image-gen-in-the-loop) pipeline.
# Mirrors bin/run_planning_task.sh: resolves the uv .venv python, .env export,
# canonical output paths, and looping over a four-task subset by default.
#
#   bash bin/run_interleaved_task.sh --model <model> [--env <env>] [...]
#
# Output goes to logs/interleaved_eval/<model>/<env>/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

ENV_NAME=""
MODEL_ID=""
PARALLEL_SESSIONS=""
MANIFEST_LIMIT=""
STEP_BUDGET=""
# Default batch subset; continuity_pipe is also supported via --env.
INTERLEAVED_ENVS=(
  knots_untangle
  separation_one_stroke
  enclosure_sheep
  continuity_2d_maze
)

usage() {
  cat <<'EOF'
usage: bash bin/run_interleaved_task.sh --model <model> [--env <env>] [--parallel-sessions N] [--manifest-limit N] [--step-budget N]

  --model               Required. e.g. internvl_imagined, bagel_imagined,
                        gpt_5_4_mini_imagined, oracle, random, greedy, ...
  --env                 Optional. Run one env (including continuity_pipe) instead of the
                        default four-task batch (knots_untangle, separation_one_stroke,
                        enclosure_sheep, continuity_2d_maze).
  --parallel-sessions   Optional. Default: omit (runner picks 0 = #API keys
                        for gym; clamps to 1 for reasoning).
  --manifest-limit      Optional. Cap episodes per env (useful for canaries).
  --step-budget         Optional. Override max steps per gym episode.
Output goes to logs/interleaved_eval/<model>/<env>/.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      [[ -n "${2-}" ]] || { printf 'error: --env requires a value\n' >&2; exit 2; }
      ENV_NAME="$2"; shift 2 ;;
    --model)
      [[ -n "${2-}" ]] || { printf 'error: --model requires a value\n' >&2; exit 2; }
      MODEL_ID="$2"; shift 2 ;;
    --parallel-sessions)
      [[ "${2-}" =~ ^[0-9]+$ ]] || { printf 'error: --parallel-sessions requires a non-negative integer\n' >&2; exit 2; }
      PARALLEL_SESSIONS="$2"; shift 2 ;;
    --manifest-limit)
      [[ "${2-}" =~ ^[1-9][0-9]*$ ]] || { printf 'error: --manifest-limit requires a positive integer\n' >&2; exit 2; }
      MANIFEST_LIMIT="$2"; shift 2 ;;
    --step-budget)
      [[ "${2-}" =~ ^[1-9][0-9]*$ ]] || { printf 'error: --step-budget requires a positive integer\n' >&2; exit 2; }
      STEP_BUDGET="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      printf 'error: unknown argument %q\n' "$1" >&2
      usage; exit 2 ;;
  esac
done

if [[ -z "$MODEL_ID" ]]; then
  printf 'error: --model is required\n' >&2
  usage
  exit 2
fi

# Use the uv-managed virtualenv created by `uv sync` (README setup). Override
# with TOPOBENCH_PYTHON to point at a different interpreter.
PYTHON="${TOPOBENCH_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
  printf 'error: no Python interpreter found.\n' >&2
  printf '       Run: uv sync --extra openai --extra lmms-eval --extra dev\n' >&2
  exit 1
fi

if [[ -f "$ROOT/.env" ]]; then
  eval "$("$PYTHON" - "$ROOT/.env" <<'PY'
import shlex
import sys
from dotenv import dotenv_values

for key, value in dotenv_values(sys.argv[1]).items():
    if not key or value is None:
        continue
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"
fi

cd "$ROOT"

run_env() {
  local env_name="$1"
  local output_dir="logs/interleaved_eval/${MODEL_ID}/${env_name}"

  printf '\n==> Running interleaved task: env=%s model=%s\n' \
    "$env_name" "$MODEL_ID"

  local args=(
    "env=${env_name}"
    "model=${MODEL_ID}"
    "run.output_dir=${output_dir}"
  )
  [[ -n "$PARALLEL_SESSIONS" ]] && args+=("run.parallel_sessions=${PARALLEL_SESSIONS}")
  [[ -n "$MANIFEST_LIMIT" ]] && args+=("run.manifest_limit=${MANIFEST_LIMIT}")
  [[ -n "$STEP_BUDGET" ]] && args+=("run.step_budget=${STEP_BUDGET}")
  "$PYTHON" interleaved_eval_tasks/agent_runner.py "${args[@]}"
}

if [[ -n "$ENV_NAME" ]]; then
  run_env "$ENV_NAME"
else
  for env_name in "${INTERLEAVED_ENVS[@]}"; do
    run_env "$env_name"
  done
fi
