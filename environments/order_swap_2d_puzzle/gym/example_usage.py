from __future__ import annotations

from env import OrderSwap2DPuzzleEnv, shortest_action_sequence


def main() -> None:
    env = OrderSwap2DPuzzleEnv(grid_rows=2, grid_cols=2, headless=True, animate=False)
    try:
        _obs, info = env.reset(
            options={
                "frontend_config": {
                    "gridRows": 2,
                    "gridCols": 2,
                    "selectedBlockIds": ["G", "R", "P"],
                    "initialArrangement": ["G", "_", "R", "P"],
                    "goalArrangement": ["G", "R", "P", "_"],
                }
            }
        )
        state = info["state"]
        print("current:", state["currentGrid"])
        print("goal:", state["goalGrid"])
        for action in shortest_action_sequence(state["currentArrangement"], state["goalArrangement"]):
            _obs, reward, terminated, truncated, info = env.step(action)
            print("action", action, "reward", reward, "grid", info["state"]["currentGrid"])
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
