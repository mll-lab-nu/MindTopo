from __future__ import annotations

import argparse

from env import ContinuityPipeEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    env = ContinuityPipeEnv(grid_size=3, headless=args.headless)
    try:
        obs, info = env.reset(seed=7)
        assert obs.ndim == 3
        state = info["state"]
        assert state["gridSize"] == 3
        assert 1 < state["totalPipes"] <= 9
        assert state["solutionTicks"] >= 5
        assert state["difficulty"] in {"easy", "medium", "hard"}
        assert state["connectedCount"] < state["totalPipes"]
        legal = state["legalActionIndices"]
        assert len(legal) == state["totalPipes"]
        _, _, _, _, next_info = env.step(legal[0])
        assert "state" in next_info
        print("continuity_pipe env smoke test passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
