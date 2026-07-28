from __future__ import annotations

import argparse

from env import OrderSwap2DPuzzleEnv


def assert_state_response(payload: dict) -> None:
    required = {"observation", "reward", "done", "success", "step_count", "info"}
    missing = required.difference(payload)
    assert not missing, f"Missing protocol fields: {sorted(missing)}"
    assert isinstance(payload["observation"], dict)
    assert isinstance(payload["done"], bool)
    assert isinstance(payload["success"], bool)
    assert isinstance(payload["step_count"], int)
    assert isinstance(payload["info"], dict)
    assert "state" in payload["info"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol conformance smoke test for Order Swap 2D Puzzle.")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    args = parser.parse_args()

    env = OrderSwap2DPuzzleEnv(grid_rows=2, grid_cols=2, headless=args.headless, animate=False, max_steps=None)
    try:
        obs, info = env.reset(
            options={
                "frontend_config": {
                    "gridRows": 2,
                    "gridCols": 2,
                    "selectedBlockIds": ["G", "R", "P"],
                    "initialArrangement": [["G", "_"], ["R", "P"]],
                    "goalArrangement": [["G", "R"], ["P", "_"]],
                    "theoreticalMinSteps": 999,
                }
            }
        )
        assert obs.ndim == 3 and obs.shape[-1] == 3
        assert "state" in info

        reset_payload = env.get_frontend_state()
        assert_state_response(reset_payload)

        step_payload = env.evaluate_frontend_step(2)
        assert_state_response(step_payload)
        assert step_payload["step_count"] == 1

        debug_state = env.get_state()
        assert debug_state["currentArrangement"] == ["G", "R", "_", "P"]
        assert debug_state["blankCellIndex"] == 2
        assert debug_state["blankRow"] == 1
        assert debug_state["blankCol"] == 0
        assert debug_state["theoreticalMinSteps"] == 2
        print("protocol_ok=True")
    finally:
        env.close()


if __name__ == "__main__":
    main()
