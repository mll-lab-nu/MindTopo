from __future__ import annotations

import argparse

from env import OrderSwap2DPuzzleEnv, shortest_action_sequence


def reset_to_reference_case(env: OrderSwap2DPuzzleEnv):
    return env.reset(
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Order Swap 2D Puzzle environment smoke tests.")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    parser.add_argument("--animate", action="store_true", help="Enable frontend animation.")
    args = parser.parse_args()

    env = OrderSwap2DPuzzleEnv(
        grid_rows=2,
        grid_cols=2,
        animate=args.animate,
        headless=args.headless,
        max_steps=None,
    )
    try:
        obs, info = reset_to_reference_case(env)
        state = info["state"]
        assert obs.ndim == 3 and obs.shape[-1] == 3
        assert info["theoretical_min_steps"] == 2
        assert info["step_budget"] == 3
        print(
            f"[reset] rgb_shape={obs.shape} current={state['currentGrid']} "
            f"goal={state['goalGrid']} theoretical={info['theoretical_min_steps']} "
            f"budget={info['step_budget']}"
        )

        obs, reward, terminated, truncated, info = env.step(1)
        print(
            f"[illegal] action=1 reward={reward:.2f} illegal={info.get('illegal')} "
            f"reason={info.get('reason')} current={info['state']['currentGrid']}"
        )
        assert obs.ndim == 3 and obs.shape[-1] == 3
        assert info.get("illegal") is True
        assert info.get("reason") == "selected_blank"
        assert not terminated and not truncated

        reset_to_reference_case(env)
        plan = shortest_action_sequence(["G", "_", "R", "P"], ["G", "R", "P", "_"])
        print(f"[solver] optimal_plan={plan}")
        assert plan == [2, 3]
        for step_idx, action in enumerate(plan):
            _obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"[solver] step={step_idx:02d} action={action} reward={reward:.2f} "
                f"illegal={info.get('illegal')} current={info['state']['currentGrid']} "
                f"done={terminated} trunc={truncated}"
            )
        assert terminated is True
        assert truncated is False
        assert info["state"]["currentArrangement"] == ["G", "R", "P", "_"]

        reset_to_reference_case(env)
        truncation_actions = [0, 1, 0]
        for step_idx, action in enumerate(truncation_actions):
            _obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"[budget] step={step_idx:02d} action={action} reward={reward:.2f} "
                f"illegal={info.get('illegal')} current={info['state']['currentGrid']} "
                f"done={terminated} trunc={truncated}"
            )
        assert terminated is False
        assert truncated is True
        assert info.get("truncated_reason") == "max_steps"
    finally:
        env.close()


if __name__ == "__main__":
    main()
