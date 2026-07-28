from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    try:
        import gym
        from gym import spaces
    except ImportError:  # pragma: no cover
        class _BaseEnv:
            metadata: Dict[str, Any] = {}

            def reset(self, *, seed: Optional[int] = None):
                if seed is not None:
                    random.seed(seed)
                return None

        class _Discrete:
            def __init__(self, n: int) -> None:
                self.n = int(n)

            def sample(self) -> int:
                return random.randrange(self.n)

        class _Box:
            def __init__(self, low: int, high: int, shape: Tuple[int, ...], dtype: Any) -> None:
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class _Spaces:
            Discrete = _Discrete
            Box = _Box

        class _GymFallback:
            Env = _BaseEnv

        gym = _GymFallback()
        spaces = _Spaces()


BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from vite_server import ViteFrontendServer
from level_data import count_level_colors, get_builtin_level, get_builtin_level_count
from solver import shortest_solution_length


ACTION_DIRECTIONS: Tuple[str, ...] = ("U", "D", "L", "R")
INVALID_ACTION = -999


class SeparationOneStrokeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        level_index: int = 0,
        max_steps: Optional[int] = None,
        host: str = "127.0.0.1",
        port: int = 0,
        headless: bool = True,
        move_reward: float = -0.02,
        illegal_reward: float = -0.10,
        success_reward: float = 1.0,
        failure_reward: float = -1.0,
        viewport_width: int = 1200,
        viewport_height: int = 900,
        frontend_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.default_level_index = int(level_index)
        self.explicit_max_steps = max_steps if max_steps is None else int(max_steps)
        self.host = host
        self.port = int(port)
        self.headless = bool(headless)
        self.move_reward = float(move_reward)
        self.illegal_reward = float(illegal_reward)
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self.viewport_width = int(viewport_width)
        self.viewport_height = int(viewport_height)

        default_frontend = Path(__file__).resolve().parent.parent / "frontend"
        self.frontend_dir = Path(frontend_dir).resolve() if frontend_dir else default_frontend.resolve()
        self.index_file = self.frontend_dir / "index.html"
        if not self.index_file.exists():
            raise FileNotFoundError(f"Frontend index not found: {self.index_file}")

        self.action_space = spaces.Discrete(len(ACTION_DIRECTIONS))
        self._episode_steps = 0
        self.current_max_steps = 1
        self.current_level_index: Optional[int] = None
        self.current_level_json: Optional[Dict[str, Any]] = None
        self.current_reference_solution_length: Optional[int] = None
        self.current_max_steps_source = "fallback_heuristic"
        self._reference_solution_cache: Dict[str, Optional[int]] = {}

        # The Vite dev server can take longer than the default 30s to come up
        # under heavy concurrent planning runs on this machine.
        self._vite = ViteFrontendServer(
            frontend_dir=self.frontend_dir,
            host=self.host,
            port=self.port,
            startup_timeout_seconds=90.0,
        )
        self._entry_url = self._vite.start()

        self._loop = asyncio.new_event_loop()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

        try:
            self._run(self._start_browser())
            initial_obs = self.render()
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=initial_obs.shape,
                dtype=np.uint8,
            )
        except Exception:
            self.close()
            raise

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    async def _start_browser(self) -> None:
        self._playwright = await async_playwright().start()
        launch_kwargs: Dict[str, Any] = {
            "headless": self.headless,
            "args": [
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
            ],
            "timeout": 120000,
        }
        self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        self._context = await self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            device_scale_factor=1.0,
        )
        self._page = await self._context.new_page()
        await self._page.goto(self._entry_url, wait_until="load")
        await self._page.wait_for_function(
            "() => window.topoBench && typeof window.topoBench.stepDir === 'function'",
            timeout=120000,
        )
        await self._page.locator("#game-container").wait_for(state="visible", timeout=120000)

    async def _evaluate(self, expression: str, arg: Any = None) -> Any:
        if arg is None:
            return await self._page.evaluate(expression)
        return await self._page.evaluate(expression, arg)

    def _data_url_to_png_bytes(self, data_url: str) -> bytes:
        if not isinstance(data_url, str) or "," not in data_url:
            raise ValueError("Invalid snapshot data URL returned by frontend.")
        _, encoded = data_url.split(",", 1)
        return base64.b64decode(encoded)

    def _png_bytes_to_rgb(self, png_bytes: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(png_bytes)) as image:
            rgb = image.convert("RGB")
            return np.asarray(rgb, dtype=np.uint8)

    async def _capture_png_bytes_async(self) -> bytes:
        data_url = await self._evaluate("() => window.topoBench.snapshot()")
        return self._data_url_to_png_bytes(data_url)

    def capture_png_bytes(self) -> bytes:
        return self._run(self._capture_png_bytes_async())

    def render(self) -> np.ndarray:
        return self._png_bytes_to_rgb(self.capture_png_bytes())

    def encode_direction(self, direction: str) -> int:
        if direction not in ACTION_DIRECTIONS:
            raise ValueError(f"Invalid direction: {direction}")
        return ACTION_DIRECTIONS.index(direction)

    def decode_action(self, action: Any) -> Optional[str]:
        if isinstance(action, str):
            candidate = action.strip().upper()
            if candidate in ACTION_DIRECTIONS:
                return candidate
            return None
        try:
            index = int(action)
        except (TypeError, ValueError):
            return None
        if 0 <= index < len(ACTION_DIRECTIONS):
            return ACTION_DIRECTIONS[index]
        return None

    def _level_cache_key(self, level_json: Dict[str, Any]) -> str:
        return json.dumps(level_json, sort_keys=True, separators=(",", ":"))

    def _fallback_max_steps(self, level_json: Dict[str, Any]) -> int:
        solution = level_json.get("solution")
        solution_length = len(solution) if isinstance(solution, list) else 0
        W = int(level_json["W"])
        H = int(level_json["H"])
        return max(W * H * 2, solution_length * 2 if solution_length else 0, (W + H) * 2)

    def _build_initial_state(self, level_json: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "W": int(level_json["W"]),
            "H": int(level_json["H"]),
            "cells": [[cell for cell in row] for row in level_json["cells"]],
            "stroke": [{"x": 0, "y": 0}],
            "currentPos": {"x": 0, "y": 0},
            "done": False,
            "success": False,
            "failureReason": None,
        }

    def _reference_solution_length(self, level_json: Dict[str, Any]) -> Optional[int]:
        cache_key = self._level_cache_key(level_json)
        if cache_key in self._reference_solution_cache:
            return self._reference_solution_cache[cache_key]
        solution = level_json.get("solution")
        if isinstance(solution, list) and solution:
            reference_length = len(solution)
            self._reference_solution_cache[cache_key] = reference_length
            return reference_length
        reference_length = shortest_solution_length(
            self._build_initial_state(level_json),
            timeout_seconds=8.0,
        )
        self._reference_solution_cache[cache_key] = reference_length
        return reference_length

    def _infer_max_steps(self, level_json: Dict[str, Any]) -> int:
        if self.explicit_max_steps is not None:
            self.current_reference_solution_length = self._reference_solution_length(level_json)
            self.current_max_steps_source = "explicit"
            return max(1, int(self.explicit_max_steps))
        reference_length = self._reference_solution_length(level_json)
        self.current_reference_solution_length = reference_length
        if reference_length is not None:
            self.current_max_steps_source = "reference_solution_length"
            return max(1, int(math.ceil(reference_length * 1.5)))
        self.current_max_steps_source = "fallback_heuristic"
        return self._fallback_max_steps(level_json)

    def _resolve_level_config(
        self,
        *,
        seed: Optional[int],
        options: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        options = options or {}
        frontend_config = dict(options.get("frontend_config") or {})

        level_json = frontend_config.get("levelJson") or options.get("level_json")
        level_index_raw = frontend_config.get("levelIndex", options.get("level_index"))

        if level_json is not None:
            level_payload = {
                "W": int(level_json["W"]),
                "H": int(level_json["H"]),
                "cells": [[cell for cell in row] for row in level_json["cells"]],
            }
            if "solution" in level_json:
                level_payload["solution"] = list(level_json["solution"])
            selected_level_index = None
        else:
            if level_index_raw is None:
                rng = random.Random(seed)
                selected_level_index = rng.randrange(get_builtin_level_count()) if seed is not None else self.default_level_index
            else:
                selected_level_index = int(level_index_raw)
            level_payload = get_builtin_level(selected_level_index)
            if level_payload is None:
                raise ValueError(f"Unknown built-in level index: {selected_level_index}")

        explicit_max_steps = options.get("max_steps", frontend_config.get("maxSteps"))
        self.current_level_index = selected_level_index
        self.current_level_json = level_payload
        if explicit_max_steps is not None:
            self.current_reference_solution_length = self._reference_solution_length(level_payload)
            self.current_max_steps_source = "explicit"
            self.current_max_steps = max(1, int(explicit_max_steps))
        else:
            self.current_max_steps = self._infer_max_steps(level_payload)
        return level_payload

    def _load_level(self, level_json: Dict[str, Any], level_index: Optional[int]) -> None:
        if level_index is not None:
            self._run(self._evaluate("(idx) => window.topoBench.loadLevelByIndex(idx)", int(level_index)))
        else:
            self._run(self._evaluate("(level) => window.topoBench.loadLevel(level)", level_json))
        self._run(self._evaluate("() => window.topoBench.reset()"))

    def _get_raw_state(self) -> Dict[str, Any]:
        return self._run(self._evaluate("() => window.topoBench.getState()"))

    def _get_evaluation(self) -> Dict[str, Any]:
        return self._run(self._evaluate("() => window.topoBench.evaluate()"))

    def _get_available_moves(self) -> List[str]:
        return list(self._run(self._evaluate("() => window.topoBench.getAvailableMoves()")))

    def _build_info(
        self,
        *,
        raw_state: Dict[str, Any],
        evaluation: Dict[str, Any],
        legal_directions: Sequence[str],
        reward: float,
        done: bool,
        success: bool,
        illegal: bool,
        reason: Optional[str],
        raw_step_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        level_json = self.current_level_json or {}
        solution = level_json.get("solution")
        info: Dict[str, Any] = {
            "state": raw_state,
            "evaluation": evaluation,
            "levelIndex": self.current_level_index,
            "levelNumber": (self.current_level_index + 1) if self.current_level_index is not None else None,
            "levelCount": get_builtin_level_count(),
            "legalDirections": list(legal_directions),
            "legalActionIndices": [self.encode_direction(direction) for direction in legal_directions],
            "gridSize": {"W": int(raw_state["W"]), "H": int(raw_state["H"])},
            "numColors": count_level_colors(level_json) if level_json else 0,
            "maxSteps": self.current_max_steps,
            "maxStepsSource": self.current_max_steps_source,
            "episode_steps": self._episode_steps,
            "success": bool(success),
            "illegal": bool(illegal),
            "reason": reason,
        }
        if self.current_reference_solution_length is not None:
            info["referenceSolutionLength"] = int(self.current_reference_solution_length)
        if isinstance(solution, list):
            info["solutionLength"] = len(solution)
        if raw_step_result is not None:
            info["rawStepResult"] = raw_step_result
        return info

    def _build_state_response(
        self,
        *,
        raw_state: Dict[str, Any],
        evaluation: Dict[str, Any],
        legal_directions: Sequence[str],
        reward: float,
        done: bool,
        success: bool,
        illegal: bool,
        reason: Optional[str],
        raw_step_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        info = self._build_info(
            raw_state=raw_state,
            evaluation=evaluation,
            legal_directions=legal_directions,
            reward=reward,
            done=done,
            success=success,
            illegal=illegal,
            reason=reason,
            raw_step_result=raw_step_result,
        )
        return {
            "observation": {
                "type": "rgb_array",
                "source": "window.topoBench.snapshot",
            },
            "reward": float(reward),
            "done": bool(done),
            "success": bool(success),
            "step_count": int(self._episode_steps),
            "info": info,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        try:
            super().reset(seed=seed)
        except TypeError:  # pragma: no cover
            super().reset()

        self._episode_steps = 0
        level_json = self._resolve_level_config(seed=seed, options=options)
        self._load_level(level_json, self.current_level_index)
        raw_state = self._get_raw_state()
        evaluation = self._get_evaluation()
        legal_directions = self._get_available_moves()
        payload = self._build_state_response(
            raw_state=raw_state,
            evaluation=evaluation,
            legal_directions=legal_directions,
            reward=0.0,
            done=False,
            success=False,
            illegal=False,
            reason=None,
        )
        obs = self.render()
        return obs, payload["info"]

    def step(self, action: Any):
        self._episode_steps += 1
        direction = self.decode_action(action)
        illegal = False
        raw_step_result: Optional[Dict[str, Any]]
        if direction is None:
            illegal = True
            raw_step_result = {"ok": False, "reason": "INVALID_ACTION"}
        else:
            raw_step_result = self._run(self._evaluate("(dir) => window.topoBench.stepDir(dir)", direction))
            illegal = not bool(raw_step_result.get("ok", False))

        raw_state = self._get_raw_state()
        evaluation = self._get_evaluation()
        legal_directions = self._get_available_moves()

        terminated = bool(raw_state.get("done", False))
        success = bool(raw_state.get("success", False))
        truncated = False
        reason = None

        if illegal:
            reward = self.illegal_reward
            reason = raw_step_result.get("reason", "INVALID_ACTION") if raw_step_result else "INVALID_ACTION"
        elif terminated and success:
            reward = self.success_reward
            reason = evaluation.get("reason") or "SOLVED"
        elif terminated and not success:
            reward = self.failure_reward
            reason = raw_state.get("failureReason") or evaluation.get("reason") or "DONE_BUT_CONSTRAINT_FAIL"
        else:
            reward = self.move_reward

        if self._episode_steps >= self.current_max_steps and not terminated:
            truncated = True
            reason = "step_budget"

        payload = self._build_state_response(
            raw_state=raw_state,
            evaluation=evaluation,
            legal_directions=legal_directions,
            reward=reward,
            done=terminated or truncated,
            success=success,
            illegal=illegal,
            reason=reason,
            raw_step_result=raw_step_result,
        )
        if truncated:
            payload["info"]["truncated_reason"] = "step_budget"

        obs = self.render()
        return obs, reward, terminated, truncated, payload["info"]

    def get_frontend_state(self) -> Dict[str, Any]:
        raw_state = self._get_raw_state()
        evaluation = self._get_evaluation()
        legal_directions = self._get_available_moves()
        return self._build_state_response(
            raw_state=raw_state,
            evaluation=evaluation,
            legal_directions=legal_directions,
            reward=0.0,
            done=bool(raw_state.get("done", False)),
            success=bool(raw_state.get("success", False)),
            illegal=False,
            reason=raw_state.get("failureReason"),
        )

    def get_state(self) -> Dict[str, Any]:
        return self._get_raw_state()

    def close(self) -> None:
        try:
            if self._context is not None:
                self._run(self._context.close())
                self._context = None
            if self._browser is not None:
                self._run(self._browser.close())
                self._browser = None
            if self._playwright is not None:
                self._run(self._playwright.stop())
                self._playwright = None
        finally:
            self._vite.close()
            if not self._loop.is_closed():
                self._loop.close()
