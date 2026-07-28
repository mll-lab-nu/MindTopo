#!/usr/bin/env python3
"""Live monitor for an interleaved eval run.

Usage:
  python bin/monitor_interleaved.py <run_dir> [--interval 5] [--once]

Reads model_answer.jsonl row counts and per-env stdout logs to render a
live per-worker progress table. Defaults to refresh every 5s; pass --once
for a single snapshot.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ENVS = ["knots_untangle", "separation_one_stroke", "enclosure_sheep", "continuity_2d_maze"]
TOTAL_PER_ENV = 60
GRAND_TOTAL = TOTAL_PER_ENV * len(ENVS)


def count_done(env_dir: Path) -> int:
    f = env_dir / "model_answer.jsonl"
    if not f.exists():
        return 0
    return sum(1 for line in f.read_text(errors="replace").splitlines() if line.strip())


def count_signals(env_dir: Path) -> tuple[int, int, int]:
    """(api_errors, image_skips, illegal_or_invalid)"""
    f = env_dir / "model_answer.jsonl"
    if not f.exists():
        return 0, 0, 0
    api = img = bad = 0
    for line in f.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for step in row.get("trajectory", []):
            if step.get("api_error"):
                api += 1
            if step.get("image_gen_skip_reason"):
                img += 1
            if step.get("illegal") or step.get("invalid_response"):
                bad += 1
    return api, img, bad


def tail(path: Path, n: int = 1, width: int = 90) -> str:
    if not path.exists():
        return ""
    try:
        lines = [l for l in path.read_text(errors="replace").splitlines() if l.strip()]
    except Exception:
        return ""
    return " ".join(lines[-n:])[-width:]


def status(env_dir: Path, done: int) -> str:
    if done >= TOTAL_PER_ENV:
        return "DONE"
    return "RUN"


def fmt_bar(done: int, total: int, width: int = 24) -> str:
    pct = done / total if total else 0
    fill = int(pct * width)
    return "[" + "#" * fill + "-" * (width - fill) + f"] {done:>3}/{total}"


def fmt_dur(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def render(run_dir: Path, t0: float, clear: bool) -> None:
    if clear:
        sys.stdout.write("\033[H\033[J")
    elapsed = time.time() - t0
    total_done = 0
    total_api = total_img = total_bad = 0
    print(f"== INTERLEAVED EVAL MONITOR ==  wall: {fmt_dur(elapsed)}  dir: {run_dir}")
    print(f"{'env':<24} {'state':<5} {'progress':<34} {'api_err':>7} {'img_skip':>8} {'illegal':>7}  tail")
    print("-" * 160)
    for env in ENVS:
        env_dir = run_dir / env
        log_path = run_dir / f"{env}.log"
        done = count_done(env_dir)
        api, img, bad = count_signals(env_dir)
        state = status(env_dir, done)
        last = tail(log_path)
        print(
            f"{env:<24} {state:<5} {fmt_bar(done, TOTAL_PER_ENV)} "
            f"{api:>7} {img:>8} {bad:>7}  {last[:80]}"
        )
        total_done += done
        total_api += api
        total_img += img
        total_bad += bad
    pct = 100 * total_done / GRAND_TOTAL
    rate = total_done / elapsed if elapsed > 0 else 0  # samples/sec
    eta_sec = (GRAND_TOTAL - total_done) / rate if rate > 0 else float("inf")
    eta_str = fmt_dur(eta_sec) if eta_sec < float("inf") else "?"
    print("-" * 160)
    print(
        f"{'TOTAL':<24} {'':<5} {fmt_bar(total_done, GRAND_TOTAL)} "
        f"{total_api:>7} {total_img:>8} {total_bad:>7}  ({pct:.1f}%)  ETA {eta_str}"
    )
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="logs/interleaved_eval/<model>/")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    if not args.run_dir.is_dir():
        sys.exit(f"run_dir not found: {args.run_dir}")
    t0 = time.time()
    try:
        while True:
            render(args.run_dir, t0, clear=not args.once)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
