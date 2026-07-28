"""Record demo MP4 videos of Knots Untangle episodes for each difficulty.

Strategy:
- Open the frontend with ``animate=True`` and ``physicsFramesPerStep=1``
  so the Three.js render loop drives the XPBD physics via
  requestAnimationFrame.
- Between agent actions, sample screenshots at ~30 FPS for a short window
  so the final video shows the rope settling.
- Use a simple row-aligned heuristic to bias the legal-action choice.
- Writes raw PNG frames to a temp dir, then stitches with ``ffmpeg``.

Output layout (default):
    gym/results/demo/<difficulty>_seed<seed>[_success|_fail].mp4
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from env import KnotsUntangleEnv


FRAME_FPS = 30
FRAMES_PER_ACTION = 36  # ~1.2s of physics per action at 30fps
FRAMES_INITIAL = 45     # ~1.5s static initial hold
FRAMES_FINAL = 60       # ~2s final hold


def legal_actions_from_state(env: KnotsUntangleEnv, state: dict) -> List[int]:
    holes = {h["id"]: h for h in state.get("holes", [])}
    rope_ends = {
        r["id"]: {r["startHole"]["id"], r["endHole"]["id"]}
        for r in state.get("ropes", [])
    }
    legal: List[int] = []
    for idx, (rope_idx, hole_idx) in enumerate(env.action_map):
        hole = holes.get(hole_idx)
        if hole is None or hole["occupied"]:
            continue
        if hole_idx in rope_ends.get(rope_idx, set()):
            continue
        legal.append(idx)
    return legal


def _row_delta_score(env: KnotsUntangleEnv, state: dict, action_idx: int) -> int:
    """Lower is better: prefer targets on the same row as the rope's other end."""
    rope_idx, hole_idx = env.action_map[action_idx]
    row = hole_idx // env.grid_size
    rope = state["ropes"][rope_idx]
    return abs(row - rope["endHole"]["row"]) + abs(row - rope["startHole"]["row"])


def pick_action(env: KnotsUntangleEnv, rng: random.Random) -> Optional[int]:
    state = env.get_state()
    legal = legal_actions_from_state(env, state)
    if not legal:
        return None
    rng.shuffle(legal)
    legal.sort(key=lambda a: _row_delta_score(env, state, a))
    return legal[0]


def capture_frames(
    env: KnotsUntangleEnv,
    out_dir: Path,
    count: int,
    start_idx: int,
) -> int:
    delay = 1.0 / FRAME_FPS
    idx = start_idx
    for _ in range(count):
        env.screenshot(path=str(out_dir / f"frame_{idx:06d}.png"))
        idx += 1
        time.sleep(delay)
    return idx


def run_episode(
    difficulty: str,
    seed: int,
    max_steps: int,
    headless: bool,
    out_dir: Path,
    rng: random.Random,
) -> Tuple[bool, int, int]:
    env = KnotsUntangleEnv(
        difficulty=difficulty,
        seed=seed,
        headless=headless,
        animate=True,
        physics_frames_per_step=1,
        max_steps=max_steps,
    )
    try:
        env.reset(seed=seed)
        frame_idx = 0
        frame_idx = capture_frames(env, out_dir, FRAMES_INITIAL, frame_idx)

        success = False
        steps = 0
        for step_idx in range(max_steps):
            action = pick_action(env, rng)
            if action is None:
                break
            _, reward, terminated, truncated, info = env.step(action)
            steps += 1
            frame_idx = capture_frames(env, out_dir, FRAMES_PER_ACTION, frame_idx)
            print(
                f"  [{difficulty} seed={seed}] step={step_idx:02d} "
                f"action={action} reward={reward:+.3f} "
                f"crossings={info.get('crossings')} done={terminated}"
            )
            if terminated:
                success = True
                break
            if truncated:
                break

        frame_idx = capture_frames(env, out_dir, FRAMES_FINAL, frame_idx)
        return success, steps, frame_idx
    finally:
        env.close()


def stitch_mp4(frames_dir: Path, output_mp4: Path) -> None:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(FRAME_FPS),
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(output_mp4),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:", result.stderr[-500:], file=sys.stderr)
        raise RuntimeError(f"ffmpeg failed: {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record demo videos of Knots Untangle episodes.")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "demo",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["easy", "medium", "hard"],
        choices=["easy", "medium", "hard"],
    )
    parser.add_argument("--seeds-per-difficulty", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep per-episode PNG frame directories after stitching.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed_base)

    manifest_lines: List[str] = []
    for difficulty in args.difficulties:
        for i in range(args.seeds_per_difficulty):
            seed = args.seed_base + i
            episode_name = f"{difficulty}_seed{seed:02d}"
            frames_dir = args.out_dir / f"_frames_{episode_name}"
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== recording {episode_name} ===")
            try:
                success, steps, frames = run_episode(
                    difficulty=difficulty,
                    seed=seed,
                    max_steps=args.max_steps,
                    headless=args.headless,
                    out_dir=frames_dir,
                    rng=rng,
                )
            except Exception as e:
                print(f"  FAILED: {e}")
                if not args.keep_frames and frames_dir.exists():
                    shutil.rmtree(frames_dir)
                continue

            tag = "success" if success else "fail"
            mp4_path = args.out_dir / f"{episode_name}_{tag}.mp4"
            stitch_mp4(frames_dir, mp4_path)
            manifest_lines.append(
                f"{episode_name}\t{tag}\tsteps={steps}\tframes={frames}\t{mp4_path.name}"
            )
            print(f"  -> {mp4_path}")

            if not args.keep_frames:
                shutil.rmtree(frames_dir)

    manifest_path = args.out_dir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"\nmanifest -> {manifest_path}")


if __name__ == "__main__":
    main()
