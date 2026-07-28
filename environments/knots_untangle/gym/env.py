from __future__ import annotations

import asyncio
import base64
import io
import random
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    from playwright.async_api import Playwright as _Playwright  # noqa: F401

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


DIFFICULTY_PRESETS: Dict[str, Dict[str, Any]] = {
    "easy": {"grid_size": 5, "rope_count": 4},
    "medium": {"grid_size": 6, "rope_count": 5},
    "hard": {"grid_size": 6, "rope_count": 6},
}

# Screenshot capture is retried on timeout: under parallel load a single
# capture can transiently stall past its timeout but succeed on a fresh attempt.
_SCREENSHOT_TIMEOUT_MS = 30000
_SCREENSHOT_RETRIES = 3
_SCREENSHOT_RETRY_BACKOFF_S = 0.5


def _require_playwright():
    try:
        from playwright.async_api import async_playwright
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing optional dependency 'playwright'. Install it with:\n"
            "  pip install playwright\n"
            "and then install the Chromium runtime with:\n"
            "  python -m playwright install chromium\n"
            "If you're using uv: `uv sync` (or `uv pip install playwright`)."
        ) from exc
    return async_playwright


class _FrontendHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".ts": "text/javascript",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
    }

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class KnotsUntangleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        difficulty: str = "medium",
        seed: Optional[int] = None,
        frontend_dir: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 0,
        headless: bool = True,
        animate: bool = False,
        illegal_reward: float = -1.0,
        move_reward: float = 0.1,
        step_penalty: float = -0.01,
        win_reward: float = 1.0,
        physics_frames_per_step: int = 60,
        max_steps: Optional[int] = None,
        viewport_width: int = 1440,
        viewport_height: int = 960,
    ) -> None:
        super().__init__()

        difficulty = str(difficulty).lower()
        if difficulty not in DIFFICULTY_PRESETS:
            raise ValueError(
                f"difficulty must be one of {list(DIFFICULTY_PRESETS)}, got {difficulty!r}"
            )
        self.difficulty = difficulty
        preset = DIFFICULTY_PRESETS[difficulty]
        self.grid_size = int(preset["grid_size"])
        self.rope_count = int(preset["rope_count"])
        self.num_holes = self.grid_size * self.grid_size
        self.init_seed = int(seed) if seed is not None else 0

        self.host = host
        self.port = int(port)
        self.headless = bool(headless)
        self.animate = bool(animate)
        self.illegal_reward = float(illegal_reward)
        self.move_reward = float(move_reward)
        self.step_penalty = float(step_penalty)
        self.win_reward = float(win_reward)
        self.physics_frames_per_step = int(physics_frames_per_step)
        self.max_steps = max_steps if max_steps is None else int(max_steps)
        self.viewport_width = int(viewport_width)
        self.viewport_height = int(viewport_height)

        default_frontend = Path(__file__).resolve().parent.parent / "frontend"
        self.frontend_dir = Path(frontend_dir).resolve() if frontend_dir else default_frontend.resolve()
        self.index_file = self.frontend_dir / "index.html"
        if not self.index_file.exists():
            raise FileNotFoundError(f"Frontend index not found: {self.index_file}")

        self.action_map: List[Tuple[int, int]] = [
            (rope_idx, hole_idx)
            for rope_idx in range(self.rope_count)
            for hole_idx in range(self.num_holes)
        ]
        self.action_space = spaces.Discrete(len(self.action_map))

        self._episode_steps = 0
        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._base_url = ""
        self._entry_url = self.index_file.as_uri()

        self._loop = asyncio.new_event_loop()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

        self._start_server()
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

    def _start_server(self) -> None:
        handler = partial(_FrontendHandler, directory=str(self.frontend_dir))
        try:
            self._server = ThreadingHTTPServer((self.host, self.port), handler)
        except PermissionError:
            self._server = None
            self._server_thread = None
            self._base_url = ""
            self._entry_url = self.index_file.as_uri()
            return

        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        bound_host, bound_port = self._server.server_address[:2]
        self._base_url = f"http://{bound_host}:{bound_port}"
        self._entry_url = f"{self._base_url}/index.html"

    async def _start_browser(self) -> None:
        async_playwright = _require_playwright()
        self._playwright = await async_playwright().start()
        frontend_errors: List[str] = []
        frontend_console: List[str] = []
        launch_kwargs: Dict[str, Any] = {
            "headless": self.headless,
            "args": [
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--use-gl=swiftshader",
            ],
            "timeout": 120000,
        }
        self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        self._context = await self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            device_scale_factor=1.0,
        )
        self._page = await self._context.new_page()
        self._page.on("pageerror", lambda exc: frontend_errors.append(str(exc)))
        self._page.on(
            "console",
            lambda msg: (
                frontend_console.append(f"{msg.type}: {msg.text}")
                if msg.type in {"error", "warning"}
                else None
            ),
        )
        last_error: Optional[BaseException] = None
        for attempt in range(3):
            try:
                await self._page.goto(self._entry_url, wait_until="load", timeout=120000)
                await self._page.wait_for_function(
                    "() => window.topoBench && typeof window.topoBench.step === 'function'",
                    timeout=120000,
                )
                await self._page.locator("#scene-container").wait_for(
                    state="visible",
                    timeout=120000,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(1.0)
                    continue
                try:
                    ready_state = await self._page.evaluate("() => document.readyState")
                except Exception:
                    ready_state = "unknown"
                raise TimeoutError(
                    "Knots Untangle frontend did not become ready after 3 attempts; "
                    f"readyState={ready_state!r}; "
                    f"page_errors={frontend_errors[-5:]!r}; "
                    f"console={frontend_console[-10:]!r}"
                ) from last_error
        await self._evaluate(
            "(cfg) => window.topoBench.reset(cfg)",
            self._build_reset_config(seed=self.init_seed),
        )

    def _build_reset_config(
        self,
        extra: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "difficulty": self.difficulty,
            "seed": int(seed) if seed is not None else self.init_seed,
            "animate": self.animate,
            "illegalReward": self.illegal_reward,
            "moveReward": self.move_reward,
            "stepPenalty": self.step_penalty,
            "winReward": self.win_reward,
            "physicsFramesPerStep": self.physics_frames_per_step,
        }
        if extra:
            cfg.update(extra)
        return cfg

    def _png_bytes_to_rgb(self, png_bytes: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(png_bytes)) as image:
            rgb = image.convert("RGB")
            return np.asarray(rgb, dtype=np.uint8)

    async def _evaluate(self, expression: str, arg: Any = None) -> Any:
        if arg is None:
            return await self._page.evaluate(expression)
        return await self._page.evaluate(expression, arg)

    async def _screenshot(self, **kwargs: Any) -> bytes:
        # Under parallel load (many browser sessions all repainting Three.js
        # canvases) a single screenshot can stall past its default timeout and
        # would otherwise kill the whole episode. Retry a few times with a
        # bounded per-attempt timeout and a short backoff so the event loop and
        # compositor get room to settle; only the final attempt's failure
        # propagates.
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        kwargs.setdefault("timeout", _SCREENSHOT_TIMEOUT_MS)
        for attempt in range(_SCREENSHOT_RETRIES):
            try:
                return await self._page.screenshot(**kwargs)
            except PlaywrightTimeoutError:
                if attempt == _SCREENSHOT_RETRIES - 1:
                    raise
                await asyncio.sleep(_SCREENSHOT_RETRY_BACKOFF_S)
        raise AssertionError("unreachable")  # loop always returns or raises

    async def _capture_scene_png(self, scene_only: bool = True, path: Optional[str] = None) -> bytes:
        if scene_only:
            # Element screenshots wait for the target to become "stable". The
            # Three.js canvas is continuously repainted, so use a clipped page
            # screenshot instead; it captures the same pixels without that wait.
            clip = await self._evaluate(
                """() => {
                    const element =
                        document.querySelector("#scene-container canvas") ||
                        document.querySelector("#scene-container");
                    if (!element) return null;
                    const rect = element.getBoundingClientRect();
                    const viewportWidth = window.innerWidth || document.documentElement.clientWidth;
                    const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
                    const x = Math.max(0, rect.left);
                    const y = Math.max(0, rect.top);
                    const right = Math.min(viewportWidth, rect.right);
                    const bottom = Math.min(viewportHeight, rect.bottom);
                    return {
                        x,
                        y,
                        width: Math.max(1, right - x),
                        height: Math.max(1, bottom - y),
                    };
                }"""
            )
            if clip is not None:
                return await self._screenshot(path=path, type="png", clip=clip)
        return await self._screenshot(path=path, type="png", full_page=False)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        try:
            super().reset(seed=seed)
        except TypeError:  # pragma: no cover
            super().reset()

        self._episode_steps = 0
        options = options or {}

        frontend_cfg = self._build_reset_config(options.get("frontend_config"), seed=seed)
        frontend_result = self._run(
            self._evaluate("(cfg) => window.topoBench.reset(cfg)", frontend_cfg)
        )
        obs = self.render()

        info = dict(frontend_result.get("info", {}) or {})
        info["frontend_result"] = frontend_result
        info["symbolic_observation"] = frontend_result.get("observation")
        info["state"] = info.get("state", self.get_state())
        info["success"] = bool(frontend_result.get("success", False))
        return obs, info

    def step(self, action: Any):
        self._episode_steps += 1
        frontend_result = self.evaluate_frontend_step(action)
        obs = self.render()

        reward = float(frontend_result.get("reward", 0.0))
        terminated = bool(frontend_result.get("done", False))
        truncated = False
        if self.max_steps is not None and self._episode_steps >= self.max_steps and not terminated:
            truncated = True

        info = dict(frontend_result.get("info", {}) or {})
        info["episode_steps"] = self._episode_steps
        info["symbolic_observation"] = frontend_result.get("observation")
        info["success"] = bool(frontend_result.get("success", False))
        info["state"] = info.get("state", self.get_state())
        if truncated:
            info["truncated_reason"] = "max_steps"
        return obs, reward, terminated, truncated, info

    def evaluate_frontend_step(self, action: Any) -> Dict[str, Any]:
        return self._run(self._evaluate("(a) => window.topoBench.step(a)", action))

    def get_frontend_state(self) -> Dict[str, Any]:
        return self._run(self._evaluate("() => window.topoBench.getState()"))

    def get_state(self) -> Dict[str, Any]:
        return self._run(self._evaluate("() => window.topoBench.getDebugState()"))

    def screenshot(self, *, scene_only: bool = True, path: Optional[str] = None) -> bytes:
        return self._run(self._capture_scene_png(scene_only=scene_only, path=path))

    def screenshot_base64(self, *, scene_only: bool = True) -> str:
        return base64.b64encode(self.screenshot(scene_only=scene_only)).decode("ascii")

    def render(self):
        return self._png_bytes_to_rgb(self.screenshot(scene_only=True))

    def decode_action(self, action: int) -> Tuple[int, int]:
        action_int = int(action)
        if action_int < 0 or action_int >= len(self.action_map):
            raise ValueError(f"Invalid discrete action index: {action_int}")
        return self.action_map[action_int]

    async def _close_browser(self) -> None:
        if self._page is not None:
            await self._page.close()
            self._page = None
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    def close(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._run(self._close_browser())
            except Exception:
                pass

        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.0)
            self._server_thread = None

        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None


# Backwards-compat alias (old gym code referenced UntangleEnv)
UntangleEnv = KnotsUntangleEnv
