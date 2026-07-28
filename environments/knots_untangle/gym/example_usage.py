from __future__ import annotations

import argparse
import random

from env import KnotsUntangleEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Example usage for the Knots Untangle environment."
    )
    parser.add_argument("--headless", action="store_true", help="Run the browser in headless mode.")
    parser.add_argument("--animate", action="store_true", help="Enable frontend move animation.")
    parser.add_argument(
        "--difficulty",
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Level preset (easy=3x3/3 ropes, medium=4x4/4 ropes, hard=5x5/5 ropes).",
    )
    parser.add_argument("--steps", type=int, default=8, help="Number of random steps to execute.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = KnotsUntangleEnv(
        difficulty=args.difficulty,
        seed=args.seed,
        animate=args.animate,
        headless=args.headless,
        max_steps=32,
    )
    try:
        obs, info = env.reset(seed=args.seed)
        state = info["state"]
        print(
            f"reset rgb_shape={obs.shape} "
            f"difficulty={args.difficulty} "
            f"grid={env.grid_size}x{env.grid_size} "
            f"ropes={env.rope_count} "
            f"crossings={state.get('crossings')} "
            f"action_space={env.action_space.n}"
        )
        for step_idx in range(args.steps):
            action = rng.randrange(env.action_space.n)
            obs, reward, terminated, truncated, step_info = env.step(action)
            rope_idx, hole_idx = env.decode_action(action)
            row, col = hole_idx // env.grid_size, hole_idx % env.grid_size
            print(
                f"step={step_idx:02d} action={action} "
                f"(rope={rope_idx} -> ({row},{col})) "
                f"reward={reward:+.3f} "
                f"illegal={step_info.get('illegal', False)} "
                f"crossings={step_info.get('crossings')} "
                f"done={terminated} trunc={truncated}"
            )
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
