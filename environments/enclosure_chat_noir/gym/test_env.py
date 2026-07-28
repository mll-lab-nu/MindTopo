from __future__ import annotations

import argparse

from env import EnclosureChatNoirEnv


def trapped_reference_case(env: EnclosureChatNoirEnv):
    return env.reset(
        options={
            "frontend_config": {
                "boardRadius": 3,
                "catIndex": 18,
                "initialBlockedIndices": [11, 12, 17, 19, 24],
                "catPolicy": "static",
            }
        }
    )


def escaping_reference_case(env: EnclosureChatNoirEnv):
    return env.reset(
        options={
            "frontend_config": {
                "boardRadius": 3,
                "catIndex": 10,
                "initialBlockedIndices": [],
                "catPolicy": "hard",
            }
        }
    )


def static_reference_case(env: EnclosureChatNoirEnv):
    return env.reset(
        options={
            "frontend_config": {
                "boardRadius": 3,
                "catIndex": 18,
                "initialBlockedIndices": [20, 27, 32],
                "catPolicy": "static",
            }
        }
    )


def easy_reference_case(env: EnclosureChatNoirEnv):
    return env.reset(
        options={
            "frontend_config": {
                "boardRadius": 3,
                "catIndex": 18,
                "initialBlockedIndices": [20, 27, 32],
                "catPolicy": "easy",
                "rngSeed": 1,
            }
        }
    )


def medium_reference_case(env: EnclosureChatNoirEnv):
    return env.reset(
        options={
            "frontend_config": {
                "boardRadius": 3,
                "catIndex": 18,
                "initialBlockedIndices": [20, 27, 32],
                "catPolicy": "medium",
                "rngSeed": 1,
            }
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Enclosure Chat Noir environment smoke tests.")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    parser.add_argument("--animate", action="store_true", help="Enable frontend animation.")
    args = parser.parse_args()

    env = EnclosureChatNoirEnv(
        board_radius=3,
        initial_block_count=2,
        cat_policy="medium",
        animate=args.animate,
        headless=args.headless,
    )
    try:
        obs, info = trapped_reference_case(env)
        state = info["state"]
        assert obs.ndim == 3 and obs.shape[-1] == 3
        assert state["catIndex"] == 18
        assert state["currentEscapeDistance"] == 3
        print(
            f"[reset] rgb_shape={obs.shape} cat={state['catIndex']} blocked={state['blockedIndices']} "
            f"escape_distance={state['currentEscapeDistance']}"
        )

        obs, reward, terminated, truncated, info = env.step(18)
        print(
            f"[illegal] action=18 reward={reward:.2f} illegal={info.get('illegal')} "
            f"reason={info.get('reason')} cat={info['state']['catIndex']}"
        )
        assert obs.ndim == 3 and obs.shape[-1] == 3
        assert info.get("illegal") is True
        assert info.get("reason") == "cat_cell"
        assert not terminated and not truncated

        static_reference_case(env)
        obs, reward, terminated, truncated, info = env.step(1)
        print(
            f"[static] action=1 reward={reward:.2f} reason={info.get('reason')} "
            f"cat={info['state']['catIndex']} done={terminated}"
        )
        assert obs.ndim == 3 and obs.shape[-1] == 3
        assert info["state"]["catIndex"] == 18
        assert terminated is False
        assert truncated is False

        easy_reference_case(env)
        obs, reward, terminated, truncated, info = env.step(1)
        print(
            f"[easy] action=1 reward={reward:.2f} reason={info.get('reason')} "
            f"cat={info['state']['catIndex']} done={terminated}"
        )
        assert info["state"]["catIndex"] == 11
        assert terminated is False
        assert truncated is False

        medium_reference_case(env)
        obs, reward, terminated, truncated, info = env.step(1)
        print(
            f"[medium] action=1 reward={reward:.2f} reason={info.get('reason')} "
            f"cat={info['state']['catIndex']} done={terminated}"
        )
        assert info["state"]["catIndex"] == 11
        assert terminated is False
        assert truncated is False

        trapped_reference_case(env)
        obs, reward, terminated, truncated, info = env.step(25)
        print(
            f"[capture] action=25 reward={reward:.2f} illegal={info.get('illegal')} "
            f"reason={info.get('reason')} cat={info['state']['catIndex']} "
            f"done={terminated} trunc={truncated}"
        )
        assert terminated is True
        assert truncated is False
        assert info.get("reason") == "cat_trapped"
        assert info["success"] is True

        escaping_reference_case(env)
        obs, reward, terminated, truncated, info = env.step(3)
        print(
            f"[escape] action=3 reward={reward:.2f} illegal={info.get('illegal')} "
            f"reason={info.get('reason')} cat={info['state']['catIndex']} "
            f"done={terminated} trunc={truncated}"
        )
        assert terminated is True
        assert truncated is False
        assert info.get("reason") == "cat_escaped"
        assert info["success"] is False
    finally:
        env.close()


if __name__ == "__main__":
    main()
