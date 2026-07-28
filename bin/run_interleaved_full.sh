#!/usr/bin/env bash
# Launch the full interleaved benchmark (4 envs concurrent, gpt_5_4_mini_imagined,
# 60 episodes/samples per env) and emit an aggregate summary on completion.
#
# Output:
#   logs/interleaved_eval/gpt_5_4_mini_imagined/<env>/{summary.json,model_answer.jsonl,...}
#   logs/interleaved_eval/gpt_5_4_mini_imagined/<env>.log
#   logs/interleaved_eval/gpt_5_4_mini_imagined/aggregate.{txt,json}
#
# Live monitor (run in a separate terminal):
#   .venv/bin/python bin/monitor_interleaved.py logs/interleaved_eval/gpt_5_4_mini_imagined
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

MODEL_ID="${MODEL_ID:-gpt_5_4_mini_imagined}"
# Override per-run with: RUN_DIR=logs/interleaved_eval/foo bash bin/run_interleaved_full.sh
RUN_DIR="${RUN_DIR:-logs/interleaved_eval/$MODEL_ID}"
if [ -e "$RUN_DIR" ] && [ "$(ls -A "$RUN_DIR" 2>/dev/null)" ]; then
  echo "RUN_DIR $RUN_DIR is non-empty — resume mode (existing rows will be skipped)." >&2
fi
mkdir -p "$RUN_DIR"

# .env auto-load + paid alias for the gpt_5_4_mini text key var.
set -a; . ./.env; set +a
export OPENAI_PAID_API_KEYS="${OPENAI_PAID_API_KEYS:-$OPENAI_API_KEYS}"
[ -n "${OPENAI_PAID_API_KEYS:-}" ] || { echo "no OPENAI key set" >&2; exit 1; }

ENVS=(knots_untangle separation_one_stroke enclosure_sheep continuity_2d_maze)

RES_OVERRIDES=(
  model.text.api_retries=5
  model.text.retry_sleep_seconds=5
  # Per-process RPM cap. 4 procs share one OPENAI key, so the per-process cap
  # × 4 is the total pressure. Was 200; lowered after v2 hit PoolExhausted.
  model.text.requests_per_minute=50
  model.text.request_timeout_seconds=180
  # Wall-clock cap on the entire generate() call (key acquire + retries).
  # Default is 180s. With reasoning enabled each call is much longer, so we
  # need more headroom or the local KeyPool times out. `+` adds the field
  # since it's not in the model yaml.
  +model.text.step_deadline_seconds=600
  model.image_gen.api_retries=5
  model.image_gen.retry_sleep_seconds=5
  model.image_gen.requests_per_minute=10
  model.image_gen.request_timeout_seconds=300
  +model.image_gen.step_deadline_seconds=600
)

START_TS=$(date +%s)
echo "$(date -Iseconds) START $RUN_DIR" | tee "$RUN_DIR/launch.log"

pids=()
for env in "${ENVS[@]}"; do
  out="$RUN_DIR/$env"
  log="$RUN_DIR/$env.log"
  mkdir -p "$out"
  echo "spawning $env -> $out" | tee -a "$RUN_DIR/launch.log"
  .venv/bin/python interleaved_eval_tasks/agent_runner.py \
    env="$env" model="$MODEL_ID" \
    "${RES_OVERRIDES[@]}" \
    run.output_dir="$out" \
    > "$log" 2>&1 &
  pids+=("$!")
done

set +e
exit_codes=()
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"
  rc=$?
  env="${ENVS[$i]}"
  exit_codes+=("$rc")
  echo "$(date -Iseconds) DONE $env exit=$rc" | tee -a "$RUN_DIR/launch.log"
done
set -e

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

# Aggregate summary
.venv/bin/python - "$RUN_DIR" "$ELAPSED" <<'PY' | tee "$RUN_DIR/aggregate.txt"
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
elapsed = int(sys.argv[2])
envs = ["knots_untangle", "separation_one_stroke", "enclosure_sheep", "continuity_2d_maze"]

def fmt_dur(s):
    h = s // 3600; m = (s % 3600) // 60; sec = s % 60
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s"

agg = {"run_dir": str(run_dir), "wall_clock_seconds": elapsed, "envs": {}}
total_done = total_correct = total_api_errors = total_img_skips = 0
print(f"=== INTERLEAVED EVAL RESULTS ===  wall: {fmt_dur(elapsed)}  dir: {run_dir}")
print()
for env in envs:
    s_path = run_dir / env / "summary.json"
    if not s_path.exists():
        print(f"{env:<26} (no summary.json)")
        agg["envs"][env] = {"status": "missing"}
        continue
    s = json.loads(s_path.read_text())
    eval_kind = s.get("eval_kind", "planning")
    api_err = s.get("api_errors", 0)
    img_skip = sum((s.get("image_gen_skip_reasons") or {}).values())
    if eval_kind == "reasoning":
        # Current runner emits gym-style keys for reasoning too; older runs
        # used total_samples/correct/accuracy. Read whichever is present.
        n = s.get("total_samples") or s.get("total_episodes") or 0
        correct = s.get("correct") if "correct" in s else s.get("successes", 0)
        rate = s.get("accuracy") if "accuracy" in s else s.get("success_rate", 0.0)
        per_type = s.get("per_type_accuracy", {})
        per_type_str = "  ".join(f"{k}:{v:.0%}" for k, v in per_type.items()) if per_type else ""
        print(f"{env:<26} (perc) {correct:>3}/{n:<3} correct ({rate:.0%})  api_err={api_err}  img_skip={img_skip}")
        if per_type_str:
            print(f"{'':<26}        per_type: {per_type_str}")
        agg["envs"][env] = {"kind": "reasoning", "n": n, "correct": correct, "accuracy": rate,
                             "per_type_accuracy": per_type, "api_errors": api_err, "image_gen_skips": img_skip}
        total_done += n; total_correct += correct
    else:
        n = s.get("total_episodes", 0)
        succ = s.get("successes", 0)
        rate = s.get("success_rate", 0.0)
        avg_steps = s.get("avg_steps_all", 0)
        print(f"{env:<26} (gym)  {succ:>3}/{n:<3} success ({rate:.0%})  avg_steps={avg_steps:.1f}  api_err={api_err}  img_skip={img_skip}")
        agg["envs"][env] = {"kind": "gym", "n": n, "successes": succ, "success_rate": rate,
                             "avg_steps_all": avg_steps, "api_errors": api_err, "image_gen_skips": img_skip}
        total_done += n; total_correct += succ
    total_api_errors += api_err; total_img_skips += img_skip

print()
overall = total_correct / total_done if total_done else 0
agg["total"] = {"n": total_done, "correct_or_success": total_correct, "rate": overall,
                "api_errors": total_api_errors, "image_gen_skips": total_img_skips}
print(f"{'TOTAL':<26}        {total_correct:>3}/{total_done:<3} ({overall:.0%})  api_err={total_api_errors}  img_skip={total_img_skips}")

(run_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
PY

echo "$(date -Iseconds) AGGREGATED $RUN_DIR/aggregate.{txt,json}" | tee -a "$RUN_DIR/launch.log"

# Reflect any non-zero per-env exit by failing the whole script.
for rc in "${exit_codes[@]}"; do
  [ "$rc" -eq 0 ] || exit "$rc"
done
