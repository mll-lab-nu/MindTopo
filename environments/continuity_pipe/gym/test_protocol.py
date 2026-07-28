from __future__ import annotations

import argparse

from env import ContinuityPipeEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    env = ContinuityPipeEnv(grid_size=3, headless=args.headless)
    try:
        _, info = env.reset(seed=11)
        state = info["state"]
        for _ in range(info["max_steps"]):
            ticks = state["targetRotationTicks"]
            action = next((index for index, remaining in enumerate(ticks) if int(remaining) > 0), None)
            if action is None:
                break
            _, _, terminated, truncated, info = env.step(action)
            state = info["state"]
            if terminated or truncated:
                break
        assert info["success"], state
        print("continuity_pipe protocol test passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
