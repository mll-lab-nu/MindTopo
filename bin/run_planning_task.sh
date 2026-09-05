#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

ENV_NAME=""
MODEL_ID=""
PARALLEL_SESSIONS=16
PARALLEL_SESSIONS_EXPLICIT=0
WORKERS_PER_KEY=16
WORKERS_PER_KEY_EXPLICIT=0
MODEL_OVERRIDES=()
RETAIN_STEP_ARTIFACTS=1
PLANNING_ENVS=(
  order_swap_2d_puzzle
  enclosure_chat_noir
  knots_untangle
  separation_one_stroke
  continuity_pipe
)

usage() {
  cat <<'EOF'
usage: bash bin/run_planning_task.sh --model <model> [--env <env>] [--parallel-sessions N|auto] [--workers-per-key N] [--no-step-artifacts]

  --model               Required. e.g. internvl, qwen, qvq, gpt_5_5, gemini_3_1_pro, ...
                        BAGEL accepts: bagel or ByteDance-Seed/BAGEL-7B-MoT
  --env                 Optional. Run one env instead of the default 5:
                        order_swap_2d_puzzle, enclosure_chat_noir, knots_untangle,
                        separation_one_stroke, continuity_pipe
  --parallel-sessions   Optional. Default 16 before model-specific limits.
                        Use "auto" to use model/key-pool capacity.
  --workers-per-key     Optional. Concurrent workers allowed per API key.
                        Default 16; InternVL uses 8, NVIDIA NIM uses 2.
                        BAGEL/Gemma/ThinkMorph/Cosmos use 1.
  --no-step-artifacts   Keep scoring JSONL/summaries but prune each completed
                        episode's PNG/GIF/prompt files to limit disk usage.

Output goes to logs/planning_eval/<model>/<env>/
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
      if [[ "${2-}" == "auto" || "${2-}" == "0" ]]; then
        PARALLEL_SESSIONS=0
      else
        [[ "${2-}" =~ ^[1-9][0-9]*$ ]] || { printf 'error: --parallel-sessions requires a positive integer or auto\n' >&2; exit 2; }
        PARALLEL_SESSIONS="$2"
      fi
      PARALLEL_SESSIONS_EXPLICIT=1
      shift 2 ;;
    --workers-per-key)
      [[ "${2-}" =~ ^[1-9][0-9]*$ ]] || { printf 'error: --workers-per-key requires a positive integer\n' >&2; exit 2; }
      WORKERS_PER_KEY="$2"
      WORKERS_PER_KEY_EXPLICIT=1
      shift 2 ;;
    --no-step-artifacts)
      RETAIN_STEP_ARTIFACTS=0
      shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      printf 'error: unknown argument %q\n' "$1" >&2
      usage; exit 2 ;;
  esac
done

if [[ "$RETAIN_STEP_ARTIFACTS" == "0" ]]; then
  MODEL_OVERRIDES+=(
    "run.write_prompt_artifacts=false"
    "run.write_step_images=false"
  )
fi

if [[ -z "$MODEL_ID" ]]; then
  printf 'error: --model is required\n' >&2
  usage
  exit 2
fi

case "$MODEL_ID" in
  internvl)
    if [[ "$WORKERS_PER_KEY_EXPLICIT" == "0" ]]; then
      WORKERS_PER_KEY=8
    fi
    if [[ "$PARALLEL_SESSIONS_EXPLICIT" == "0" ]]; then
      PARALLEL_SESSIONS=8
    fi
    printf 'note: %s detected — using planning parallel_sessions=%s workers-per-key=%s\n' \
      "$MODEL_ID" "$PARALLEL_SESSIONS" "$WORKERS_PER_KEY" >&2
    ;;
  bagel|gemma|thinkmorph|cosmos)
    if [[ "$WORKERS_PER_KEY_EXPLICIT" == "0" ]]; then
      WORKERS_PER_KEY=1
    fi
    if [[ "$PARALLEL_SESSIONS_EXPLICIT" == "0" ]]; then
      PARALLEL_SESSIONS=8
    fi
    printf 'note: %s detected — using planning parallel_sessions=%s workers-per-key=%s\n' \
      "$MODEL_ID" "$PARALLEL_SESSIONS" "$WORKERS_PER_KEY" >&2
    ;;
  nemotron)
    if [[ "$WORKERS_PER_KEY_EXPLICIT" == "0" ]]; then
      WORKERS_PER_KEY=2
    fi
    if [[ "$PARALLEL_SESSIONS_EXPLICIT" == "0" ]]; then
      PARALLEL_SESSIONS=2
    fi
    printf 'note: NVIDIA NIM model %s detected — using planning parallel_sessions=%s workers-per-key=%s\n' \
      "$MODEL_ID" "$PARALLEL_SESSIONS" "$WORKERS_PER_KEY" >&2
    ;;
esac

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
import os
import shlex
import sys
from dotenv import dotenv_values

for key, value in dotenv_values(sys.argv[1]).items():
    if not key or value is None:
        continue
    if key in os.environ:
        continue
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"
fi

cd "$ROOT"
export PW_TEST_SCREENSHOT_NO_FONTS_READY="${PW_TEST_SCREENSHOT_NO_FONTS_READY:-1}"
if [[ -n "$WORKERS_PER_KEY" ]]; then
  export TOPOBENCH_WORKERS_PER_KEY="$WORKERS_PER_KEY"
fi

HEAVY_ENV_LOCK_ROOT="logs/planning_eval/_heavy_env_locks"
HEAVY_ENV_LOCK=""
HEAVY_ENV_LOCK_HELD=0

release_heavy_env_lock() {
  if [[ "$HEAVY_ENV_LOCK_HELD" == "1" && -d "$HEAVY_ENV_LOCK" ]]; then
    local owner=""
    owner="$(cat "$HEAVY_ENV_LOCK/owner.pid" 2>/dev/null || true)"
    if [[ "$owner" == "$$" ]]; then
      rm -f "$HEAVY_ENV_LOCK/owner.pid" "$HEAVY_ENV_LOCK/owner.txt"
      rmdir "$HEAVY_ENV_LOCK" 2>/dev/null || true
    fi
  fi
  HEAVY_ENV_LOCK_HELD=0
}

acquire_heavy_env_lock() {
  local env_name="$1"
  local owner=""
  mkdir -p "$HEAVY_ENV_LOCK_ROOT"
  HEAVY_ENV_LOCK="${HEAVY_ENV_LOCK_ROOT}/${env_name}.lock"
  while ! mkdir "$HEAVY_ENV_LOCK" 2>/dev/null; do
    owner="$(cat "$HEAVY_ENV_LOCK/owner.pid" 2>/dev/null || true)"
    if [[ -n "$owner" ]] && ! kill -0 "$owner" 2>/dev/null; then
      rm -f "$HEAVY_ENV_LOCK/owner.pid" "$HEAVY_ENV_LOCK/owner.txt"
      rmdir "$HEAVY_ENV_LOCK" 2>/dev/null || true
      continue
    fi
    printf 'note: waiting for heavy planning env lock: env=%s model=%s owner=%s\n' \
      "$env_name" "$MODEL_ID" "${owner:-unknown}" >&2
    sleep 10
  done
  printf '%s\n' "$$" > "$HEAVY_ENV_LOCK/owner.pid"
  printf 'env=%s model=%s started=%s\n' "$env_name" "$MODEL_ID" "$(date -u +%FT%TZ)" \
    > "$HEAVY_ENV_LOCK/owner.txt"
  HEAVY_ENV_LOCK_HELD=1
  printf 'note: acquired heavy planning env lock: env=%s model=%s pid=%s\n' \
    "$env_name" "$MODEL_ID" "$$" >&2
}

trap release_heavy_env_lock EXIT INT TERM

run_env() {
  local env_name="$1"
  local output_dir="logs/planning_eval/${MODEL_ID}/${env_name}"
  local workers_per_key_label="${TOPOBENCH_WORKERS_PER_KEY:-2(default)}"
  local status=0

  case "$env_name" in
    separation_one_stroke|continuity_pipe)
      acquire_heavy_env_lock "$env_name"
      ;;
  esac

  printf '\n==> Running planning task: env=%s model=%s parallel_sessions=%s workers_per_key=%s\n' \
    "$env_name" "$MODEL_ID" "$PARALLEL_SESSIONS" "$workers_per_key_label"
  if "$PYTHON" planning_eval_tasks/agent_runner.py \
      "env=${env_name}" \
      "model=${MODEL_ID}" \
      "run.parallel_sessions=${PARALLEL_SESSIONS}" \
      "run.output_dir=${output_dir}" \
      ${MODEL_OVERRIDES[@]+"${MODEL_OVERRIDES[@]}"}; then
    status=0
  else
    status=$?
  fi
  release_heavy_env_lock
  return "$status"
}

if [[ -n "$ENV_NAME" ]]; then
  run_env "$ENV_NAME"
else
  for env_name in "${PLANNING_ENVS[@]}"; do
    run_env "$env_name"
  done
fi
