from __future__ import annotations

import argparse

from env import SeparationOneStrokeEnv


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
    assert "legalDirections" in payload["info"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol conformance smoke test for Separation One Stroke.")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    args = parser.parse_args()

    env = SeparationOneStrokeEnv(level_index=0, headless=args.headless, max_steps=8)
    try:
        obs, info = env.reset(options={"frontend_config": {"levelIndex": 0}})
        assert obs.ndim == 3 and obs.shape[-1] == 3
        assert info["levelNumber"] == 1

        reset_payload = env.get_frontend_state()
        assert_state_response(reset_payload)
        assert reset_payload["step_count"] == 0

        obs, reward, terminated, truncated, info = env.step(env.encode_direction("R"))
        assert obs.ndim == 3 and obs.shape[-1] == 3
        assert reward < 0
        assert terminated is False
        assert truncated is False

        payload = env.get_frontend_state()
        assert_state_response(payload)
        assert payload["step_count"] == 1
        assert payload["info"]["state"]["currentPos"] == {"x": 1, "y": 0}
        print("protocol_ok=True")
    finally:
        env.close()


if __name__ == "__main__":
    main()
