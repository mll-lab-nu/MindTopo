from __future__ import annotations

import argparse
import random

from env import SeparationOneStrokeEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Example usage for the Separation One Stroke planning environment.")
    parser.add_argument("--level-index", type=int, default=0, help="0-based built-in level index.")
    parser.add_argument("--steps", type=int, default=12, help="Maximum random actions to take.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = SeparationOneStrokeEnv(level_index=args.level_index, headless=args.headless)
    try:
        obs, info = env.reset(seed=args.seed, options={"frontend_config": {"levelIndex": args.level_index}})
        state = info["state"]
        print(
            f"reset rgb_shape={obs.shape} level={info['levelNumber']} "
            f"pos=({state['currentPos']['x']},{state['currentPos']['y']}) "
            f"legal={info['legalDirections']} max_steps={info['maxSteps']}"
        )

        for step_index in range(args.steps):
            legal_actions = info["legalActionIndices"]
            if not legal_actions:
                break
            action = rng.choice(legal_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            direction = env.decode_action(action)
            state = info["state"]
            print(
                f"step={step_index:02d} action={direction} reward={reward:.2f} "
                f"illegal={info.get('illegal')} reason={info.get('reason')} "
                f"pos=({state['currentPos']['x']},{state['currentPos']['y']}) "
                f"done={terminated} trunc={truncated}"
            )
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
