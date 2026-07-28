from __future__ import annotations

import argparse

from env import SeparationOneStrokeEnv


SIMPLE_LEVEL = {
    "W": 2,
    "H": 2,
    "cells": [["red"]],
    "solution": ["R", "U"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Separation One Stroke environment smoke tests.")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    args = parser.parse_args()

    env = SeparationOneStrokeEnv(level_index=0, headless=args.headless, max_steps=8)
    try:
        obs, info = env.reset(options={"frontend_config": {"levelIndex": 0}})
        state = info["state"]
        assert obs.ndim == 3 and obs.shape[-1] == 3
        assert state["currentPos"] == {"x": 0, "y": 0}
        print(
            f"[reset] rgb_shape={obs.shape} level={info['levelNumber']} "
            f"pos=({state['currentPos']['x']},{state['currentPos']['y']}) legal={info['legalDirections']}"
        )

        obs, reward, terminated, truncated, info = env.step(env.encode_direction("D"))
        print(
            f"[illegal] action=D reward={reward:.2f} illegal={info.get('illegal')} "
            f"reason={info.get('reason')} pos={info['state']['currentPos']}"
        )
        assert info.get("illegal") is True
        assert info.get("reason") == "OUT_OF_BOUNDS"
        assert terminated is False and truncated is False

        env.reset(options={"frontend_config": {"levelJson": SIMPLE_LEVEL}})
        for action in ["R", "U"]:
            obs, reward, terminated, truncated, info = env.step(env.encode_direction(action))
            print(
                f"[solve] action={action} reward={reward:.2f} illegal={info.get('illegal')} "
                f"reason={info.get('reason')} pos={info['state']['currentPos']} "
                f"done={terminated} trunc={truncated}"
            )
        assert terminated is True
        assert truncated is False
        assert info["success"] is True

        env.reset(options={"frontend_config": {"levelJson": SIMPLE_LEVEL}, "max_steps": 1})
        obs, reward, terminated, truncated, info = env.step(env.encode_direction("R"))
        print(
            f"[truncate] reward={reward:.2f} illegal={info.get('illegal')} "
            f"reason={info.get('reason')} step_count={info['episode_steps']} done={terminated} trunc={truncated}"
        )
        assert terminated is False
        assert truncated is True
        assert info.get("truncated_reason") == "step_budget"
    finally:
        env.close()


if __name__ == "__main__":
    main()
