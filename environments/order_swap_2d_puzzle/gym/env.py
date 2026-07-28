from __future__ import annotations

import asyncio
import base64
import io
import math
import random
import threading
from collections.abc import Sequence as SequenceABC
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - fallback when gymnasium is unavailable.
    try:
        import gym
        from gym import spaces
    except ImportError:  # pragma: no cover - local lightweight fallback for this repo.
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


BLANK_TOKEN = "_"
MIN_GRID_ROWS = 2
MIN_GRID_COLS = 2
MAX_GRID_ROWS = 4
MAX_GRID_COLS = 4
MAX_CELLS = 16
DEFAULT_BUDGET_MULTIPLIER = 1.2
INVALID_ACTION = -999
STABLE_RNG_INCREMENT = 0x6D2B79F5
UINT32_MASK = 0xFFFFFFFF
DIFFICULTY_GRID_SHAPES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "easy": ((2, 2), (2, 3), (3, 2), (3, 3)),
    "medium": ((3, 4), (4, 3)),
    "hard": ((4, 4),),
}

BLOCK_LIBRARY: Tuple[Dict[str, str], ...] = (
    {"id": "G", "label": "G", "name": "Green", "color": "#55a612"},
    {"id": "R", "label": "R", "name": "Red", "color": "#fa4045"},
    {"id": "P", "label": "P", "name": "Purple", "color": "#7f75dc"},
    {"id": "B", "label": "B", "name": "Blue", "color": "#178adb"},
    {"id": "Y", "label": "Y", "name": "Yellow", "color": "#fda31d"},
    {"id": "O", "label": "O", "name": "Orange", "color": "#f05a2a"},
    {"id": "T", "label": "T", "name": "Teal", "color": "#1fa6a2"},
    {"id": "M", "label": "M", "name": "Magenta", "color": "#c64191"},
    {"id": "C", "label": "C", "name": "Cyan", "color": "#00a6d6"},
    {"id": "L", "label": "L", "name": "Lime", "color": "#8cc63e"},
    {"id": "N", "label": "N", "name": "Navy", "color": "#2c4f8f"},
    {"id": "S", "label": "S", "name": "Silver", "color": "#9aa7b2"},
    {"id": "V", "label": "V", "name": "Violet", "color": "#9a4bd7"},
    {"id": "I", "label": "I", "name": "Indigo", "color": "#4b5bdc"},
    {"id": "A", "label": "A", "name": "Amber", "color": "#d79a00"},
)
BLOCK_BY_ID = {spec["id"]: spec for spec in BLOCK_LIBRARY}


class StableRng:
    """Small deterministic RNG mirrored by the frontend for seed-based setup sampling."""

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & UINT32_MASK

    def _next_uint32(self) -> int:
        self.state = (self.state + STABLE_RNG_INCREMENT) & UINT32_MASK
        value = self.state
        value = ((value ^ (value >> 15)) * (value | 1)) & UINT32_MASK
        value = (value ^ ((value + (((value ^ (value >> 7)) * (value | 61)) & UINT32_MASK)) & UINT32_MASK)) & UINT32_MASK
        return (value ^ (value >> 14)) & UINT32_MASK

    def randrange(self, stop: int) -> int:
        if stop <= 0:
            raise ValueError("stop must be positive")
        return self._next_uint32() % int(stop)


def normalize_difficulty(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().casefold()
    if not text or text in {"custom", "grid", "manual", "none", "null"} or text.startswith("grid_"):
        return None
    aliases = {
        "easy": "easy",
        "e": "easy",
        "medium": "medium",
        "med": "medium",
        "m": "medium",
        "hard": "hard",
        "h": "hard",
    }
    if text not in aliases:
        raise ValueError("difficulty must be one of: easy, medium, hard.")
    return aliases[text]


def grid_shape_for_difficulty(difficulty: Any, seed: Optional[int] = 1) -> Tuple[int, int]:
    normalized = normalize_difficulty(difficulty)
    if normalized is None:
        raise ValueError("difficulty must be one of: easy, medium, hard.")
    shapes = DIFFICULTY_GRID_SHAPES[normalized]
    if len(shapes) == 1:
        return shapes[0]
    rng = StableRng(1 if seed is None else int(seed))
    return shapes[rng.randrange(len(shapes))]


def difficulty_label_for_grid(grid_rows: int, grid_cols: int) -> str:
    shape = (int(grid_rows), int(grid_cols))
    for difficulty, shapes in DIFFICULTY_GRID_SHAPES.items():
        if shape in shapes:
            return difficulty
    return f"grid_{shape[0]}x{shape[1]}"


class _FrontendHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".ts": "text/javascript",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
    }

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def _normalize_token(value: Any) -> str:
    if value is None:
        return BLANK_TOKEN
    text = str(value).strip().upper()
    return BLANK_TOKEN if not text or text == BLANK_TOKEN else text


def validate_grid_shape(grid_rows: int, grid_cols: int) -> Tuple[int, int]:
    rows = int(grid_rows)
    cols = int(grid_cols)
    if rows < MIN_GRID_ROWS or rows > MAX_GRID_ROWS:
        raise ValueError(f"grid_rows must be between {MIN_GRID_ROWS} and {MAX_GRID_ROWS}.")
    if cols < MIN_GRID_COLS or cols > MAX_GRID_COLS:
        raise ValueError(f"grid_cols must be between {MIN_GRID_COLS} and {MAX_GRID_COLS}.")
    if rows * cols > MAX_CELLS:
        raise ValueError(f"2D swap puzzle supports at most {MAX_CELLS} cells.")
    return rows, cols


def cell_index_to_row_col(cell_index: int, grid_cols: int) -> Dict[str, int]:
    index = int(cell_index)
    cols = int(grid_cols)
    return {"row": index // cols, "col": index % cols}


def row_col_to_cell_index(row: int, col: int, grid_cols: int) -> int:
    return int(row) * int(grid_cols) + int(col)


def serialize_arrangement(arrangement: Sequence[str]) -> Tuple[str, ...]:
    return tuple(arrangement)


def arrangement_to_grid(arrangement: Sequence[str], grid_rows: int, grid_cols: int) -> List[List[str]]:
    rows, cols = validate_grid_shape(grid_rows, grid_cols)
    values = list(arrangement)
    if len(values) != rows * cols:
        raise ValueError(f"Arrangement length must be {rows * cols}, got {len(values)}.")
    return [values[row * cols:(row + 1) * cols] for row in range(rows)]


def flatten_arrangement(arrangement: Sequence[Any]) -> List[Any]:
    if (
        isinstance(arrangement, SequenceABC)
        and not isinstance(arrangement, (str, bytes))
        and arrangement
        and all(isinstance(row, SequenceABC) and not isinstance(row, (str, bytes)) for row in arrangement)
    ):
        flattened: List[Any] = []
        for row in arrangement:
            flattened.extend(list(row))
        return flattened
    return list(arrangement)


def choose_block_ids(num_blocks: int, requested_ids: Optional[Sequence[str]] = None) -> List[str]:
    count = int(num_blocks)
    if count < 3 or count > len(BLOCK_LIBRARY):
        raise ValueError(f"num_blocks must be between 3 and {len(BLOCK_LIBRARY)}.")

    if requested_ids:
        unique: List[str] = []
        seen = set()
        for raw in requested_ids:
            token = _normalize_token(raw)
            if token == BLANK_TOKEN or token in seen or token not in BLOCK_BY_ID:
                continue
            unique.append(token)
            seen.add(token)
        if len(unique) >= count:
            return unique[:count]

    return [spec["id"] for spec in BLOCK_LIBRARY[:count]]


def normalize_arrangement(
    arrangement: Sequence[Any],
    *,
    selected_block_ids: Sequence[str],
    grid_rows: int,
    grid_cols: int,
) -> List[str]:
    rows, cols = validate_grid_shape(grid_rows, grid_cols)
    expected_length = rows * cols
    flat_values = flatten_arrangement(arrangement)
    if len(flat_values) != expected_length:
        raise ValueError(f"Arrangement length must be {expected_length}, got {len(flat_values)}.")

    tokens = [_normalize_token(value) for value in flat_values]
    if tokens.count(BLANK_TOKEN) != 1:
        raise ValueError("Arrangement must contain exactly one blank cell.")

    remaining = {block_id: 1 for block_id in selected_block_ids}
    for token in tokens:
        if token == BLANK_TOKEN:
            continue
        if token not in remaining:
            raise ValueError(f"Unexpected block token: {token}")
        remaining[token] -= 1
        if remaining[token] < 0:
            raise ValueError(f"Duplicate block token: {token}")

    if any(count != 0 for count in remaining.values()):
        raise ValueError("Arrangement does not contain the expected block set.")
    return tokens


def shuffle_in_place(items: List[str], rng: Any) -> None:
    for index in range(len(items) - 1, 0, -1):
        swap_index = int(rng.randrange(index + 1))
        items[index], items[swap_index] = items[swap_index], items[index]


def shortest_swap_distance(initial_arrangement: Sequence[str], goal_arrangement: Sequence[str]) -> int:
    return len(shortest_action_sequence(initial_arrangement, goal_arrangement))


def shortest_action_sequence(initial_arrangement: Sequence[str], goal_arrangement: Sequence[str]) -> List[int]:
    start = serialize_arrangement(initial_arrangement)
    goal = serialize_arrangement(goal_arrangement)
    if start == goal:
        return []
    if len(start) != len(goal) or len(set(start)) != len(start) or len(set(goal)) != len(goal) or set(start) != set(goal):
        raise ValueError("Initial and goal arrangements must contain the same unique tokens.")
    if start.count(BLANK_TOKEN) != 1 or goal.count(BLANK_TOKEN) != 1:
        raise ValueError("Initial and goal arrangements must each contain exactly one blank token.")

    current = list(start)
    target = list(goal)
    actions: List[int] = []
    guard_limit = len(current) * 3
    while tuple(current) != goal:
        if len(actions) > guard_limit:
            raise RuntimeError("2D swap puzzle shortest path reconstruction failed.")
        blank_index = current.index(BLANK_TOKEN)
        desired_at_blank = target[blank_index]
        if desired_at_blank != BLANK_TOKEN:
            action_index = current.index(desired_at_blank)
        else:
            action_index = next(
                index
                for index, (current_token, goal_token) in enumerate(zip(current, target))
                if current_token != goal_token and current_token != BLANK_TOKEN
            )
        current[blank_index], current[action_index] = current[action_index], current[blank_index]
        actions.append(int(action_index))

    return actions


def sample_swap_2d_puzzle_instance(
    grid_rows: int,
    grid_cols: int,
    *,
    seed: Optional[int] = None,
    selected_block_ids: Optional[Sequence[str]] = None,
    min_theoretical_steps: int = 1,
) -> Dict[str, Any]:
    rows, cols = validate_grid_shape(grid_rows, grid_cols)
    rng = StableRng(seed) if seed is not None else random.Random()
    chosen_ids = choose_block_ids(rows * cols - 1, selected_block_ids)
    goal_arrangement = chosen_ids[:]
    shuffle_in_place(goal_arrangement, rng)
    goal_arrangement.append(BLANK_TOKEN)
    cell_tokens = goal_arrangement[:]

    for _attempt in range(1024):
        initial_arrangement = cell_tokens[:]
        shuffle_in_place(initial_arrangement, rng)
        theoretical_steps = shortest_swap_distance(initial_arrangement, goal_arrangement)
        if theoretical_steps >= max(1, int(min_theoretical_steps)):
            return {
                "grid_rows": rows,
                "grid_cols": cols,
                "selected_block_ids": chosen_ids,
                "initial_arrangement": initial_arrangement,
                "goal_arrangement": goal_arrangement,
                "theoretical_min_steps": theoretical_steps,
            }

    raise RuntimeError("Unable to sample a 2D swap puzzle instance with the requested difficulty.")


class OrderSwap2DPuzzleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        grid_rows: int = 3,
        grid_cols: int = 3,
        selected_block_ids: Optional[Sequence[str]] = None,
        frontend_dir: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 0,
        headless: bool = True,
        animate: bool = False,
        illegal_reward: float = -1.0,
        move_reward: float = -0.01,
        win_reward: float = 1.0,
        max_steps: Optional[int] = None,
        budget_multiplier: float = DEFAULT_BUDGET_MULTIPLIER,
        min_theoretical_steps: int = 1,
        viewport_width: int = 1440,
        viewport_height: int = 980,
    ) -> None:
        super().__init__()

        self.grid_rows, self.grid_cols = validate_grid_shape(grid_rows, grid_cols)
        self.num_blocks = self.grid_rows * self.grid_cols - 1
        self.selected_block_ids = choose_block_ids(self.num_blocks, selected_block_ids)
        self.host = host
        self.port = int(port)
        self.headless = bool(headless)
        self.animate = bool(animate)
        self.illegal_reward = float(illegal_reward)
        self.move_reward = float(move_reward)
        self.win_reward = float(win_reward)
        self.explicit_max_steps = max_steps if max_steps is None else int(max_steps)
        self.budget_multiplier = float(budget_multiplier)
        self.min_theoretical_steps = int(min_theoretical_steps)
        self.viewport_width = int(viewport_width)
        self.viewport_height = int(viewport_height)

        default_frontend = Path(__file__).resolve().parent.parent / "frontend"
        self.frontend_dir = Path(frontend_dir).resolve() if frontend_dir else default_frontend.resolve()
        self.index_file = self.frontend_dir / "index.html"
        if not self.index_file.exists():
            raise FileNotFoundError(f"Frontend index not found: {self.index_file}")

        self.action_map = list(range(self.grid_rows * self.grid_cols))
        self.action_space = spaces.Discrete(len(self.action_map))

        self._episode_steps = 0
        self.current_step_budget = max(1, self.explicit_max_steps or 1)
        self.current_theoretical_min_steps = 0
        self.current_difficulty = difficulty_label_for_grid(self.grid_rows, self.grid_cols)
        self.current_initial_arrangement: List[str] = []
        self.current_goal_arrangement: List[str] = []

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
        chrome_path = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        if chrome_path.exists():
            try:
                self._browser = await self._playwright.chromium.launch(
                    **{**launch_kwargs, "channel": "chrome"}
                )
            except Exception:
                self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        else:
            self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        self._context = await self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            device_scale_factor=1.0,
        )
        self._context.set_default_timeout(120000)
        self._context.set_default_navigation_timeout(120000)
        self._page = await self._context.new_page()
        await self._page.goto(self._entry_url, wait_until="load", timeout=120000)
        await self._page.wait_for_function(
            "() => window.topoBench && typeof window.topoBench.step === 'function'",
            timeout=120000,
        )
        await self._page.locator("#scene-container").wait_for(state="visible", timeout=120000)
        await self._page.locator("#current-card").wait_for(state="visible", timeout=120000)
        await self._page.locator("#goal-card").wait_for(state="visible", timeout=120000)

    async def _evaluate(self, expression: str, arg: Any = None) -> Any:
        if arg is None:
            return await self._page.evaluate(expression)
        return await self._page.evaluate(expression, arg)

    async def _capture_locator_png(self, selector: str, path: Optional[str] = None) -> bytes:
        locator = self._page.locator(selector)
        await locator.wait_for(state="visible", timeout=120000)
        return await locator.screenshot(path=path, type="png", timeout=120000)

    def _png_bytes_to_rgb(self, png_bytes: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(png_bytes)) as image:
            rgb = image.convert("RGB")
            return np.asarray(rgb, dtype=np.uint8)

    def _build_reset_config(
        self,
        *,
        seed: Optional[int],
        options: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        options = options or {}
        frontend_config = dict(options.get("frontend_config") or {})
        requested_seed = frontend_config.get("seed", options.get("seed", seed))
        sample_seed = None if requested_seed is None else int(requested_seed)
        has_explicit_grid = (
            any(key in options for key in ("grid_rows", "rows", "grid_cols", "cols"))
            or any(key in frontend_config for key in ("gridRows", "rows", "gridCols", "cols"))
        )
        difficulty_source = options.get("difficulty", None)
        if difficulty_source is None:
            difficulty_source = options.get("target_difficulty", options.get("targetDifficulty", None))
        if difficulty_source is None:
            difficulty_source = frontend_config.get("targetDifficulty", None)
        if difficulty_source is None and not has_explicit_grid:
            difficulty_source = frontend_config.get("difficulty", None)
        requested_difficulty = normalize_difficulty(difficulty_source)
        if requested_difficulty is not None:
            requested_rows, requested_cols = grid_shape_for_difficulty(requested_difficulty, sample_seed)
        else:
            requested_rows = int(
                options.get(
                    "grid_rows",
                    options.get("rows", frontend_config.get("gridRows", frontend_config.get("rows", self.grid_rows))),
                )
            )
            requested_cols = int(
                options.get(
                    "grid_cols",
                    options.get("cols", frontend_config.get("gridCols", frontend_config.get("cols", self.grid_cols))),
                )
            )
            requested_rows, requested_cols = validate_grid_shape(requested_rows, requested_cols)
        resolved_difficulty = requested_difficulty or difficulty_label_for_grid(requested_rows, requested_cols)
        requested_block_ids = options.get("selected_block_ids", frontend_config.get("selectedBlockIds"))
        chosen_block_ids = choose_block_ids(requested_rows * requested_cols - 1, requested_block_ids)

        if "initialArrangement" in frontend_config and "goalArrangement" in frontend_config:
            initial_arrangement = normalize_arrangement(
                frontend_config["initialArrangement"],
                selected_block_ids=chosen_block_ids,
                grid_rows=requested_rows,
                grid_cols=requested_cols,
            )
            goal_arrangement = normalize_arrangement(
                frontend_config["goalArrangement"],
                selected_block_ids=chosen_block_ids,
                grid_rows=requested_rows,
                grid_cols=requested_cols,
            )
            theoretical_min_steps = shortest_swap_distance(initial_arrangement, goal_arrangement)
        else:
            sampled = sample_swap_2d_puzzle_instance(
                requested_rows,
                requested_cols,
                seed=sample_seed,
                selected_block_ids=chosen_block_ids,
                min_theoretical_steps=options.get(
                    "min_theoretical_steps",
                    frontend_config.get("minTheoreticalSteps", self.min_theoretical_steps),
                ),
            )
            chosen_block_ids = sampled["selected_block_ids"]
            initial_arrangement = sampled["initial_arrangement"]
            goal_arrangement = sampled["goal_arrangement"]
            theoretical_min_steps = sampled["theoretical_min_steps"]

        step_budget = frontend_config.get("stepBudget", options.get("step_budget"))
        if step_budget is None:
            if self.explicit_max_steps is not None:
                step_budget = self.explicit_max_steps
            else:
                step_budget = max(1, math.ceil(theoretical_min_steps * self.budget_multiplier))

        self.grid_rows = requested_rows
        self.grid_cols = requested_cols
        self.num_blocks = self.grid_rows * self.grid_cols - 1
        self.selected_block_ids = list(chosen_block_ids)
        self.action_map = list(range(self.grid_rows * self.grid_cols))
        self.action_space = spaces.Discrete(len(self.action_map))
        self.current_initial_arrangement = list(initial_arrangement)
        self.current_goal_arrangement = list(goal_arrangement)
        self.current_theoretical_min_steps = int(theoretical_min_steps)
        self.current_step_budget = int(step_budget)
        self.current_difficulty = resolved_difficulty

        frontend_payload = {
            "gridRows": self.grid_rows,
            "gridCols": self.grid_cols,
            "difficulty": self.current_difficulty,
            "targetDifficulty": requested_difficulty,
            "seed": 1 if sample_seed is None else sample_seed,
            "selectedBlockIds": self.selected_block_ids,
            "initialArrangement": self.current_initial_arrangement,
            "goalArrangement": self.current_goal_arrangement,
            "theoreticalMinSteps": self.current_theoretical_min_steps,
            "stepBudget": self.current_step_budget,
            "budgetMultiplier": self.budget_multiplier,
            "animate": self.animate,
            "illegalReward": self.illegal_reward,
            "moveReward": self.move_reward,
            "winReward": self.win_reward,
        }
        passthrough_keys = {
            "animate": options.get("animate", frontend_config.get("animate", self.animate)),
            "illegalReward": options.get("illegal_reward", frontend_config.get("illegalReward", self.illegal_reward)),
            "moveReward": options.get("move_reward", frontend_config.get("moveReward", self.move_reward)),
            "winReward": options.get("win_reward", frontend_config.get("winReward", self.win_reward)),
            "budgetMultiplier": options.get(
                "budget_multiplier",
                frontend_config.get("budgetMultiplier", self.budget_multiplier),
            ),
            "stepBudget": self.current_step_budget,
        }
        frontend_payload.update(passthrough_keys)
        return frontend_payload

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        try:
            super().reset(seed=seed)
        except TypeError:  # pragma: no cover
            super().reset()

        self._episode_steps = 0
        frontend_cfg = self._build_reset_config(seed=seed, options=options)
        frontend_result = self._run(self._evaluate("(async (cfg) => await window.topoBench.reset(cfg))", frontend_cfg))
        obs = self.render()

        info = dict(frontend_result.get("info", {}) or {})
        info["frontend_result"] = frontend_result
        info["symbolic_observation"] = frontend_result.get("observation")
        info["state"] = info.get("state", self.get_state())
        info["success"] = bool(frontend_result.get("success", False))
        info["theoretical_min_steps"] = self.current_theoretical_min_steps
        info["step_budget"] = self.current_step_budget
        info["grid_rows"] = self.grid_rows
        info["grid_cols"] = self.grid_cols
        info["difficulty"] = self.current_difficulty
        return obs, info

    def step(self, action: int):
        self._episode_steps += 1
        action_int = int(action)
        frontend_result = self.evaluate_frontend_step(action_int)
        obs = self.render()

        reward = float(frontend_result.get("reward", 0.0))
        terminated = bool(frontend_result.get("done", False))
        truncated = False
        if self._episode_steps >= self.current_step_budget and not terminated:
            truncated = True

        info = dict(frontend_result.get("info", {}) or {})
        info["episode_steps"] = self._episode_steps
        info["symbolic_observation"] = frontend_result.get("observation")
        info["success"] = bool(frontend_result.get("success", False))
        info["state"] = info.get("state", self.get_state())
        info["theoretical_min_steps"] = self.current_theoretical_min_steps
        info["step_budget"] = self.current_step_budget
        info["grid_rows"] = self.grid_rows
        info["grid_cols"] = self.grid_cols
        info["difficulty"] = self.current_difficulty
        if truncated:
            info["truncated_reason"] = "max_steps"
        return obs, reward, terminated, truncated, info

    def evaluate_frontend_step(self, action: int) -> Dict[str, Any]:
        return self._run(self._evaluate("(async (a) => await window.topoBench.step(a))", int(action)))

    def get_frontend_state(self) -> Dict[str, Any]:
        return self._run(self._evaluate("() => window.topoBench.getState()"))

    def get_state(self) -> Dict[str, Any]:
        return self._run(self._evaluate("() => window.topoBench.getDebugState()"))

    def decode_action(self, action: int) -> int:
        action_int = int(action)
        if action_int < 0 or action_int >= len(self.action_map):
            raise ValueError(f"Invalid cell action index: {action_int}")
        return action_int

    def screenshot(self, *, scene_only: bool = True, path: Optional[str] = None) -> bytes:
        selector = "#scene-container" if scene_only else "body"
        return self._run(self._capture_locator_png(selector, path=path))

    def screenshot_current_card(self, *, path: Optional[str] = None) -> bytes:
        return self._run(self._capture_locator_png("#current-board", path=path))

    def screenshot_goal_card(self, *, path: Optional[str] = None) -> bytes:
        return self._run(self._capture_locator_png("#goal-board", path=path))

    def screenshot_base64(self, *, scene_only: bool = True) -> str:
        return base64.b64encode(self.screenshot(scene_only=scene_only)).decode("ascii")

    def render(self):
        return self._png_bytes_to_rgb(self.screenshot(scene_only=True))

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
