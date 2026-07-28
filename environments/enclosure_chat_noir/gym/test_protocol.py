from __future__ import annotations

import argparse

from env import EnclosureChatNoirEnv


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
    parser = argparse.ArgumentParser(description="Protocol conformance smoke test for Enclosure Chat Noir.")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    args = parser.parse_args()

    env = EnclosureChatNoirEnv(board_radius=3, initial_block_count=2, headless=args.headless, animate=False)
    try:
        obs, info = env.reset(
            options={
                "frontend_config": {
                    "boardRadius": 3,
                    "catIndex": 18,
                    "initialBlockedIndices": [11, 12, 17, 19, 24],
                    "catPolicy": "hard",
                }
            }
        )
        assert obs.ndim == 3 and obs.shape[-1] == 3
        assert "state" in info

        reset_payload = env.get_frontend_state()
        assert_state_response(reset_payload)

        step_payload = env.evaluate_frontend_step(25)
        assert_state_response(step_payload)
        assert step_payload["step_count"] == 1
        assert step_payload["success"] is True

        debug_state = env.get_state()
        assert debug_state["terminal"]["reason"] == "cat_trapped"
        assert debug_state["catPolicy"]["name"] == "hard"
        assert debug_state["difficulty"]["cat_intelligence"] == 3
        print("protocol_ok=True")
    finally:
        env.close()


if __name__ == "__main__":
    main()
