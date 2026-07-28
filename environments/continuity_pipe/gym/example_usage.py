from __future__ import annotations

import argparse

from env import ContinuityPipeEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    env = ContinuityPipeEnv(grid_size=args.grid_size, headless=args.headless)
    try:
        _, info = env.reset(seed=args.seed)
        state = info["state"]
        print(
            f"grid={state['gridSize']} source={state['source']} "
            f"connected={state['connectedCount']}/{state['totalPipes']}"
        )
        legal = state["legalActionIndices"]
        if legal:
            _, reward, terminated, truncated, next_info = env.step(legal[0])
            next_state = next_info["state"]
            print(
                f"step action={legal[0]} reward={reward} terminated={terminated} "
                f"truncated={truncated} connected={next_state['connectedCount']}/{next_state['totalPipes']}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
