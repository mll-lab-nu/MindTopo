#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

MODEL_ID="internvl"
NUM_CPUS=16
WORKERS_PER_KEY=2
MAX_SHARD_WORKERS=16
SHARD_SIZE="${TOPOBENCH_REASONING_SHARD_SIZE:-0}"
ADAPTIVE_CONCURRENCY=0
EXTRA_LMMS_ARGS=()
ENV_NAMES="knots_static,separation_objects,enclosure_hole_detection,enclosure_sheep,order_bead_string,continuity_2d_maze,continuity_3d_maze,order_origami"
SUPPORTED_REASONING_ENVS=(
  knots_static
  separation_objects
  enclosure_hole_detection
  enclosure_sheep
  order_bead_string
  continuity_2d_maze
  continuity_3d_maze
  order_origami
)
PLANNING_ONLY_ENVS=(
  continuity_pipe
  separation_one_stroke
  order_swap_2d_puzzle
  enclosure_chat_noir
  knots_untangle
)

usage() {
  cat <<'EOF'
usage: bash bin/run_reasoning_task.sh [--model <id>] [--env csv] [--num-cpus N] [--workers-per-key N] [--max-shard-workers N] [--shard-size N] [--adaptive-concurrency]

  --model              Default internvl
                       BAGEL accepts: bagel or ByteDance-Seed/BAGEL-7B-MoT
	  --env                Comma-separated env names. Default: 8 reasoning envs
	                       (knots_static, separation_objects, enclosure_hole_detection,
	                        enclosure_sheep, order_bead_string,
	                        continuity_2d_maze, continuity_3d_maze, order_origami)
	                       Planning-only envs such as order_swap_2d_puzzle must use
	                       bin/run_planning_task.sh instead.
  --num-cpus           Default 16; API concurrency inside each lmms-eval shard
  --workers-per-key    Default 2; key-pool slots per API key
  --max-shard-workers  Default 16; local lmms-eval subprocess cap
  --shard-size         Optional fixed samples per lmms-eval subprocess. Default
                       auto; long-running API models use 50 unless overridden.
                       Set TOPOBENCH_REASONING_SHARD_SIZE to change the default.
  --adaptive-concurrency
                       Enable lmms-eval adaptive API concurrency. Default off.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ -n "${2-}" ]] || { printf 'error: --model requires a value\n' >&2; exit 2; }
      MODEL_ID="$2"; shift 2 ;;
    --num-cpus)
      [[ "${2-}" =~ ^[1-9][0-9]*$ ]] || { printf 'error: --num-cpus requires a positive integer\n' >&2; exit 2; }
      NUM_CPUS="$2"; shift 2 ;;
    --workers-per-key)
      [[ "${2-}" =~ ^[1-9][0-9]*$ ]] || { printf 'error: --workers-per-key requires a positive integer\n' >&2; exit 2; }
      WORKERS_PER_KEY="$2"; shift 2 ;;
    --max-shard-workers)
      [[ "${2-}" =~ ^[1-9][0-9]*$ ]] || { printf 'error: --max-shard-workers requires a positive integer\n' >&2; exit 2; }
      MAX_SHARD_WORKERS="$2"; shift 2 ;;
    --shard-size)
      [[ "${2-}" =~ ^[1-9][0-9]*$ ]] || { printf 'error: --shard-size requires a positive integer\n' >&2; exit 2; }
      SHARD_SIZE="$2"; shift 2 ;;
    --adaptive-concurrency)
      ADAPTIVE_CONCURRENCY=1; shift ;;
    --env)
      [[ -n "${2-}" ]] || { printf 'error: --env requires a value\n' >&2; exit 2; }
      ENV_NAMES="$2"; shift 2 ;;
    --tasks)
      printf 'error: --tasks was renamed to --env\n' >&2
      usage; exit 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      printf 'error: unknown argument %q\n' "$1" >&2
      usage; exit 2 ;;
	esac
	done

contains_env() {
  local needle="$1"; shift
  local item
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

IFS=',' read -r -a REQUESTED_ENVS <<< "$ENV_NAMES"
for raw_env in "${REQUESTED_ENVS[@]}"; do
  env_name="$(printf '%s' "$raw_env" | tr -d '[:space:]')"
  [[ -n "$env_name" ]] || continue
  if contains_env "$env_name" "${SUPPORTED_REASONING_ENVS[@]}"; then
    continue
  fi
  if contains_env "$env_name" "${PLANNING_ONLY_ENVS[@]}"; then
    printf 'error: env %q is an interactive planning env, not a reasoning/lmms-eval task.\n' "$env_name" >&2
    printf '       Use: bash bin/run_planning_task.sh --model %s --env %s\n' "$MODEL_ID" "$env_name" >&2
    exit 2
  fi
  printf 'error: unknown reasoning env %q\n' "$env_name" >&2
  printf '       Supported reasoning envs: %s\n' "${SUPPORTED_REASONING_ENVS[*]}" >&2
  exit 2
done

case "$MODEL_ID" in
  ByteDance-Seed/BAGEL-7B-MoT)
    MODEL_ID="bagel"
    ;;
esac

# InternVL is a heavy reasoning model (long CoT, rate-limited API). Force gentle
# concurrency and a larger token budget so answers aren't truncated mid-thought.
if [[ "$MODEL_ID" == "internvl" || "$MODEL_ID" == "gemma" || "$MODEL_ID" == "thinkmorph" || "$MODEL_ID" == "cosmos" ]]; then
  NUM_CPUS=8
  WORKERS_PER_KEY=1
  MAX_SHARD_WORKERS=8
  if [[ "$SHARD_SIZE" == "0" ]]; then
    SHARD_SIZE=50
  fi
  printf 'note: %s detected — forcing num-cpus=8 workers-per-key=1 max-shard-workers=8 shard-size=%s\n' "$MODEL_ID" "$SHARD_SIZE" >&2
fi

if [[ "$MODEL_ID" == "bagel" ]]; then
  NUM_CPUS=1
  WORKERS_PER_KEY=1
  MAX_SHARD_WORKERS=1
  if [[ "$SHARD_SIZE" == "0" ]]; then
    SHARD_SIZE=50
  fi
  printf 'note: %s detected — forcing num-cpus=1 workers-per-key=1 max-shard-workers=1 shard-size=%s\n' "$MODEL_ID" "$SHARD_SIZE" >&2
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
export PYTHON

cd "$ROOT"
ADAPTIVE_ARGS=(--no-adaptive-concurrency)
if [[ "$ADAPTIVE_CONCURRENCY" == "1" ]]; then
  ADAPTIVE_ARGS=(--adaptive-concurrency)
fi
SHARD_ARGS=()
if [[ "$SHARD_SIZE" != "0" ]]; then
  SHARD_ARGS=(--shard-size "$SHARD_SIZE")
fi

exec ./bin/topobench-eval \
  --num-cpus "$NUM_CPUS" \
  --workers-per-key "$WORKERS_PER_KEY" \
  --max-shard-workers "$MAX_SHARD_WORKERS" \
  ${SHARD_ARGS[@]+"${SHARD_ARGS[@]}"} \
  "${ADAPTIVE_ARGS[@]}" \
  "$MODEL_ID" "$ENV_NAMES" \
  ${EXTRA_LMMS_ARGS[@]+"${EXTRA_LMMS_ARGS[@]}"}
