#!/usr/bin/env bash
# setup_frontends.sh — install + build all gym env frontends with bun.
#
# Required for envs that serve static built assets (knots_untangle uses
# SimpleHTTPRequestHandler so it needs `frontend/dist/`). For envs that run
# Vite in dev mode (one_stroke, continuity_pipe, …), the install alone is
# enough, but building is harmless.
#
# Prerequisite: install bun (https://bun.sh).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

if ! command -v bun >/dev/null 2>&1; then
  echo "Error: \`bun\` not found on PATH. Install from https://bun.sh." >&2
  exit 1
fi

GYM_ENVS=(
  knots_untangle
  continuity_pipe
  separation_one_stroke
  order_swap_2d_puzzle
  enclosure_chat_noir
)

setup_one() {
  local env="$1"
  local fdir="$ROOT/environments/$env/frontend"
  if [ ! -f "$fdir/package.json" ]; then
    echo "[skip] $env: no $fdir/package.json"
    return 0
  fi
  echo "[install] $env"
  (cd "$fdir" && bun install) || { echo "[fail] $env: bun install"; return 1; }
  if grep -q '"build"' "$fdir/package.json" 2>/dev/null; then
    echo "[build] $env"
    (cd "$fdir" && bun run build) || echo "[note] $env: build step failed (ok if env uses vite dev)"
  fi
}

# Run all envs in parallel — bun install + vite build is IO-bound, so 6× speedup.
pids=()
for env in "${GYM_ENVS[@]}"; do
  setup_one "$env" &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail + 1))
done

if [ "$fail" -gt 0 ]; then
  echo "Done with $fail failure(s). Static-served envs need their dist/ to run; vite-dev envs need node_modules." >&2
  exit 1
fi
echo "Done. Static-served envs now have dist/; vite-dev envs have node_modules ready."
