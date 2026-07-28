from __future__ import annotations

import argparse
import random

from env import EnclosureChatNoirEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Example usage for the Enclosure Chat Noir environment.")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    parser.add_argument("--animate", action="store_true", help="Enable frontend animation.")
    parser.add_argument("--radius", type=int, default=3, choices=[3, 4], help="Board radius.")
    parser.add_argument("--blocked", type=int, default=5, help="Initial blocked cell count.")
    parser.add_argument("--cat-policy", default="medium", choices=["static", "easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    env = EnclosureChatNoirEnv(
        board_radius=args.radius,
        initial_block_count=args.blocked,
        cat_policy=args.cat_policy,
        headless=args.headless,
        animate=args.animate,
    )

    try:
        obs, info = env.reset(seed=args.seed)
        state = info["state"]
        print(
            f"reset rgb_shape={obs.shape} cat={state['catIndex']} blocked={state['blockedIndices']} "
            f"policy={state['catPolicy']['name']} escape_distance={state['currentEscapeDistance']}"
        )

        rng = random.Random(args.seed)
        for step_index in range(args.steps):
            legal_actions = state["legalActionIndices"]
            if not legal_actions:
                break
            action = rng.choice(legal_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            state = info["state"]
            print(
                f"step={step_index:02d} action={action} reward={reward:.2f} cat={state['catIndex']} "
                f"illegal={info.get('illegal')} reason={info.get('reason')} done={terminated}"
            )
            if terminated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
