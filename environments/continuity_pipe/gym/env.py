from __future__ import annotations

import asyncio
import base64
import io
import random
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


DIRS: Tuple[Tuple[int, int, int, int], ...] = (
    (0, -1, 1, 4),  # N
    (1, 0, 2, 8),   # E
    (0, 1, 4, 1),   # S
    (-1, 0, 8, 2),  # W
)
MIN_GRID_SIZE = 3
MAX_GRID_SIZE = 6
MIN_BENCHMARK_SOLUTION_TICKS = 13
EASY_MAX_SOLUTION_TICKS = 17
MEDIUM_MAX_SOLUTION_TICKS = 23
INVALID_ACTION = -999
DIFFICULTY_SETUPS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "easy": {
        "grid_size": (4, 4),
        "active_pipe_count": (10, 13),
        "junction_count": (2, 3),
        "solution_ticks": (13, 17),
    },
    "medium": {
        "grid_size": (5, 5),
        "active_pipe_count": (13, 17),
        "junction_count": (3, 5),
        "solution_ticks": (17, 23),
    },
    "hard": {
        "grid_size": (5, 5),
        "active_pipe_count": (17, 23),
        "junction_count": (5, 7),
        "solution_ticks": (23, 27),
    },
}


class _FrontendHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".ts": "text/javascript",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
    }

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def clamp_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, parsed))


def difficulty_label_for_solution_ticks(solution_ticks: int) -> str:
    ticks = int(solution_ticks)
    if ticks < MIN_BENCHMARK_SOLUTION_TICKS:
        return "warmup"
    if ticks <= EASY_MAX_SOLUTION_TICKS:
        return "easy"
    if ticks <= MEDIUM_MAX_SOLUTION_TICKS:
        return "medium"
    return "hard"


def normalize_difficulty(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"easy", "medium", "hard"}:
        return normalized
    return None


def solution_tick_bounds_for_difficulty(difficulty: str) -> Tuple[int, Optional[int]]:
    normalized = normalize_difficulty(difficulty)
    if normalized is not None:
        minimum, maximum = DIFFICULTY_SETUPS[normalized]["solution_ticks"]
        return minimum, maximum
    raise ValueError(f"Unsupported continuity_pipe difficulty: {difficulty}")


def grid_size_for_difficulty(difficulty: str) -> int:
    normalized = normalize_difficulty(difficulty)
    if normalized is None:
        raise ValueError(f"Unsupported continuity_pipe difficulty: {difficulty}")
    minimum, maximum = DIFFICULTY_SETUPS[normalized]["grid_size"]
    if minimum != maximum:
        raise ValueError(f"Difficulty {normalized!r} does not map to one fixed grid size.")
    return int(minimum)


def action_to_xy(action_index: int, grid_size: int) -> Dict[str, int]:
    action = int(action_index)
    return {"x": action % int(grid_size), "y": action // int(grid_size)}


def xy_to_action(x: int, y: int, grid_size: int) -> int:
    return int(y) * int(grid_size) + int(x)


def rotate_mask(mask: int, ticks: int) -> int:
    current = int(mask) & 15
    steps = int(ticks) % 4
    for _ in range(steps):
        next_mask = 0
        if current & 1:
            next_mask |= 2
        if current & 2:
            next_mask |= 4
        if current & 4:
            next_mask |= 8
        if current & 8:
            next_mask |= 1
        current = next_mask
    return current


def target_ticks_for_mask_rotation(mask: int, rotation: int) -> int:
    solved_mask = int(mask) & 15
    if solved_mask == 0:
        return 0
    current_mask = rotate_mask(solved_mask, rotation)
    for ticks in range(4):
        if rotate_mask(current_mask, ticks) == solved_mask:
            return ticks
    return 0


def neighbor_index(x: int, y: int, grid_size: int, dx: int, dy: int) -> Optional[int]:
    nx = int(x) + int(dx)
    ny = int(y) + int(dy)
    if nx < 0 or nx >= grid_size or ny < 0 or ny >= grid_size:
        return None
    return xy_to_action(nx, ny, grid_size)


def grid_neighbors(index: int, grid_size: int) -> List[Tuple[int, int, int]]:
    x = int(index) % int(grid_size)
    y = int(index) // int(grid_size)
    neighbors: List[Tuple[int, int, int]] = []
    for dx, dy, bit, opposite_bit in DIRS:
        other = neighbor_index(x, y, grid_size, dx, dy)
        if other is not None:
            neighbors.append((other, bit, opposite_bit))
    return neighbors


def default_active_pipe_count(grid_size: int, *, rng: random.Random) -> int:
    total = int(grid_size) * int(grid_size)
    min_active = min(total - 1, max(3, (total + 1) // 2))
    max_active = max(min_active, total - 1)
    return rng.randint(min_active, max_active)


def active_pipe_indices_from_masks(masks: Sequence[int]) -> List[int]:
    return [index for index, mask in enumerate(masks) if int(mask) & 15]


def junction_count_from_masks(masks: Sequence[int]) -> int:
    return sum(1 for mask in masks if (int(mask) & 15).bit_count() == 3)


def has_four_way_junction(masks: Sequence[int]) -> bool:
    return any((int(mask) & 15).bit_count() >= 4 for mask in masks)


def _int_or_none(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def _sample_count_range(
    *,
    rng: random.Random,
    exact: Optional[int],
    minimum: Optional[int],
    maximum: Optional[int],
    fallback_minimum: int,
    fallback_maximum: int,
) -> int:
    if exact is not None:
        return int(exact)
    low = fallback_minimum if minimum is None else int(minimum)
    high = fallback_maximum if maximum is None else int(maximum)
    if high < low:
        raise ValueError(f"Invalid count range: {low}-{high}")
    return rng.randint(low, high)


def generate_solved_masks(
    grid_size: int,
    *,
    source_index: int,
    rng: random.Random,
    active_pipe_count: Optional[int] = None,
) -> List[int]:
    total = int(grid_size) * int(grid_size)
    fallback_active_count = (
        default_active_pipe_count(grid_size, rng=rng)
        if active_pipe_count is None
        else int(active_pipe_count)
    )
    target_active_count = clamp_int(
        active_pipe_count if active_pipe_count is not None else fallback_active_count,
        2,
        total,
        min(total, max(2, fallback_active_count)),
    )
    masks = [0 for _ in range(total)]
    active = {int(source_index)}

    while len(active) < target_active_count:
        candidates: List[Tuple[int, int, int, int]] = []
        for current in sorted(active):
            candidates.extend(
                (current, other, bit, opposite_bit)
                for other, bit, opposite_bit in grid_neighbors(current, grid_size)
                if other not in active
            )
        if not candidates:
            break
        current, other, bit, opposite_bit = rng.choice(candidates)
        masks[current] |= bit
        masks[other] |= opposite_bit
        active.add(other)

    if len(active) != target_active_count:
        raise RuntimeError("Failed to generate a connected sparse pipe board.")
    return masks


def current_masks(solved_masks: Sequence[int], rotations: Sequence[int]) -> List[int]:
    return [rotate_mask(mask, rotation) for mask, rotation in zip(solved_masks, rotations, strict=True)]


def connected_indices(*, grid_size: int, source_index: int, masks: Sequence[int]) -> List[int]:
    connected = {int(source_index)}
    queue = [int(source_index)]
    while queue:
        current = queue.pop(0)
        xy = action_to_xy(current, grid_size)
        current_mask = int(masks[current])
        for dx, dy, bit, opposite_bit in DIRS:
            if not current_mask & bit:
                continue
            other = neighbor_index(xy["x"], xy["y"], grid_size, dx, dy)
            if other is None or other in connected:
                continue
            if int(masks[other]) & opposite_bit:
                connected.add(other)
                queue.append(other)
    return sorted(connected)


def solution_ticks_for_rotations(rotations: Sequence[int], solved_masks: Optional[Sequence[int]] = None) -> int:
    total = 0
    for index, rotation in enumerate(rotations):
        if solved_masks is not None and not (int(solved_masks[index]) & 15):
            continue
        mask = int(solved_masks[index]) if solved_masks is not None else 1
        total += target_ticks_for_mask_rotation(mask, int(rotation))
    return total


def max_steps_for_solution_ticks(solution_ticks: int) -> int:
    ticks = int(solution_ticks)
    return max(1, (ticks * 13 + 9) // 10)


def _rotation_weight(target_ticks: int, difficulty: Optional[str]) -> int:
    _ = target_ticks, difficulty
    return 1


def sample_rotations(
    solved_masks: Sequence[int],
    *,
    rng: random.Random,
    difficulty: Optional[str],
) -> List[int]:
    rotations: List[int] = []
    for mask in solved_masks:
        if not (int(mask) & 15):
            rotations.append(0)
            continue
        choices = list(range(4))
        weights = [
            _rotation_weight(target_ticks_for_mask_rotation(int(mask), rotation), difficulty)
            for rotation in choices
        ]
        rotations.append(rng.choices(choices, weights=weights, k=1)[0])
    return rotations


def sample_pipe_instance(
    grid_size: int,
    *,
    seed: Optional[int] = None,
    difficulty: Optional[str] = None,
    min_solution_ticks: Optional[int] = None,
    max_solution_ticks: Optional[int] = None,
    active_pipe_count: Optional[int] = None,
    min_active_pipe_count: Optional[int] = None,
    max_active_pipe_count: Optional[int] = None,
    min_junction_count: Optional[int] = None,
    max_junction_count: Optional[int] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    total = int(grid_size) * int(grid_size)
    target_difficulty = normalize_difficulty(difficulty)
    if target_difficulty is not None:
        setup = DIFFICULTY_SETUPS[target_difficulty]
        setup_grid_min, setup_grid_max = setup["grid_size"]
        if not setup_grid_min <= int(grid_size) <= setup_grid_max:
            raise ValueError(
                f"Difficulty {target_difficulty!r} requires grid size "
                f"{setup_grid_min} through {setup_grid_max}; got {grid_size}."
            )
        setup_active_min, setup_active_max = setup["active_pipe_count"]
        setup_junction_min, setup_junction_max = setup["junction_count"]
        min_solution_ticks = setup["solution_ticks"][0] if min_solution_ticks is None else min_solution_ticks
        max_solution_ticks = setup["solution_ticks"][1] if max_solution_ticks is None else max_solution_ticks
        min_active_pipe_count = setup_active_min if min_active_pipe_count is None else min_active_pipe_count
        max_active_pipe_count = setup_active_max if max_active_pipe_count is None else max_active_pipe_count
        min_junction_count = setup_junction_min if min_junction_count is None else min_junction_count
        max_junction_count = setup_junction_max if max_junction_count is None else max_junction_count
    min_ticks = MIN_BENCHMARK_SOLUTION_TICKS if min_solution_ticks is None else int(min_solution_ticks)
    max_ticks = None if max_solution_ticks is None else int(max_solution_ticks)
    fallback_active_min = min(total, max(2, (total + 1) // 2))
    fallback_active_max = max(fallback_active_min, total - 1)

    rotations: List[int] = []
    for _attempt in range(20000):
        source_index = rng.randrange(total)
        sampled_active_count = _sample_count_range(
            rng=rng,
            exact=active_pipe_count,
            minimum=_int_or_none(min_active_pipe_count),
            maximum=_int_or_none(max_active_pipe_count),
            fallback_minimum=fallback_active_min,
            fallback_maximum=fallback_active_max,
        )
        solved_masks = generate_solved_masks(
            grid_size,
            source_index=source_index,
            rng=rng,
            active_pipe_count=sampled_active_count,
        )
        if has_four_way_junction(solved_masks):
            continue
        total_pipes = len(active_pipe_indices_from_masks(solved_masks))
        junction_count = junction_count_from_masks(solved_masks)
        if min_junction_count is not None and junction_count < int(min_junction_count):
            continue
        if max_junction_count is not None and junction_count > int(max_junction_count):
            continue

        for _rotation_attempt in range(40):
            rotations = sample_rotations(solved_masks, rng=rng, difficulty=target_difficulty)
            solution_ticks = solution_ticks_for_rotations(rotations, solved_masks)
            if solution_ticks < min_ticks:
                continue
            if max_ticks is not None and solution_ticks > max_ticks:
                continue
            masks = current_masks(solved_masks, rotations)
            connected = connected_indices(grid_size=grid_size, source_index=source_index, masks=masks)
            if len(connected) < total_pipes and solution_ticks > 0:
                break
        else:
            continue
        break
    else:
        raise RuntimeError("Unable to sample a scrambled continuity_pipe board.")

    return {
        "grid_size": int(grid_size),
        "source_index": int(source_index),
        "solved_masks": solved_masks,
        "rotations": rotations,
        "solution_ticks": int(solution_ticks),
        "difficulty": target_difficulty or difficulty_label_for_solution_ticks(solution_ticks),
        "total_pipes": len(active_pipe_indices_from_masks(solved_masks)),
        "junction_count": junction_count_from_masks(solved_masks),
        "max_steps": max_steps_for_solution_ticks(solution_ticks),
    }


def normalize_int_list(values: Any, expected_length: int, *, field_name: str) -> List[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a list of integers.")
    normalized = [int(value) for value in values]
    if len(normalized) != int(expected_length):
        raise ValueError(f"{field_name} must have length {expected_length}, got {len(normalized)}.")
    return normalized


class ContinuityPipeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        grid_size: int = 3,
        frontend_dir: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 0,
        headless: bool = True,
        illegal_reward: float = -1.0,
        move_reward: float = -0.01,
        success_reward: float = 1.0,
        timeout_penalty: float = -1.0,
        viewport_width: int = 1440,
        viewport_height: int = 1080,
    ) -> None:
        super().__init__()
        self.grid_size = clamp_int(grid_size, MIN_GRID_SIZE, MAX_GRID_SIZE, 3)
        self.host = host
        self.port = int(port)
        self.headless = bool(headless)
        self.illegal_reward = float(illegal_reward)
        self.move_reward = float(move_reward)
        self.success_reward = float(success_reward)
        self.timeout_penalty = float(timeout_penalty)
        self.viewport_width = int(viewport_width)
        self.viewport_height = int(viewport_height)

        default_frontend = Path(__file__).resolve().parent.parent / "frontend"
        self.frontend_dir = Path(frontend_dir).resolve() if frontend_dir else default_frontend.resolve()
        self.index_file = self.frontend_dir / "index.html"
        if not self.index_file.exists():
            raise FileNotFoundError(f"Frontend index not found: {self.index_file}")

        self.action_map = list(range(self.grid_size * self.grid_size))
        self.action_space = spaces.Discrete(len(self.action_map))
        self._episode_steps = 0
        self.current_source_index = 0
        self.current_solved_masks: List[int] = []
        self.current_rotations: List[int] = []
        self.current_solution_ticks = 0
        self.current_difficulty = "warmup"
        self.current_junction_count = 0
        self.current_max_steps = self.grid_size * self.grid_size * 4

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
            self.observation_space = spaces.Box(low=0, high=255, shape=initial_obs.shape, dtype=np.uint8)
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
                self._browser = await self._playwright.chromium.launch(**{**launch_kwargs, "channel": "chrome"})
            except Exception:
                self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        else:
            self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        self._context = await self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            device_scale_factor=1.0,
        )
        self._page = await self._context.new_page()
        await self._page.goto(self._entry_url, wait_until="load")
        await self._page.wait_for_function(
            "() => window.topoBench && typeof window.topoBench.step === 'function'",
            timeout=30000,
        )
        await self._page.locator("#scene-container").wait_for(state="visible", timeout=30000)
        await self._page.locator("#board-card").wait_for(state="visible", timeout=30000)

    async def _evaluate(self, expression: str, arg: Any = None) -> Any:
        if arg is None:
            return await self._page.evaluate(expression)
        return await self._page.evaluate(expression, arg)

    async def _capture_locator_png(self, selector: str, path: Optional[str] = None) -> bytes:
        locator = self._page.locator(selector)
        await locator.wait_for(state="visible", timeout=30000)
        return await locator.screenshot(path=path, type="png")

    async def _capture_scene_png(self, path: Optional[str] = None) -> bytes:
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
            return await self._page.screenshot(path=path, type="png", clip=clip)
        return await self._page.screenshot(path=path, type="png", full_page=False)

    def _png_bytes_to_rgb(self, png_bytes: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(png_bytes)) as image:
            rgb = image.convert("RGB")
            return np.asarray(rgb, dtype=np.uint8)

    def _build_reset_config(self, *, seed: Optional[int], options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        options = options or {}
        frontend_config = dict(options.get("frontend_config") or {})
        target_difficulty = normalize_difficulty(options.get("difficulty", frontend_config.get("difficulty", frontend_config.get("targetDifficulty"))))
        difficulty_grid_size = grid_size_for_difficulty(target_difficulty) if target_difficulty is not None else self.grid_size
        has_explicit_masks = "solvedMasks" in frontend_config or "solved_masks" in options
        requested_grid_size = options.get("grid_size", frontend_config.get("gridSize", difficulty_grid_size))
        if target_difficulty is not None and not has_explicit_masks:
            requested_grid_size = difficulty_grid_size
        grid_size = clamp_int(
            requested_grid_size,
            MIN_GRID_SIZE,
            MAX_GRID_SIZE,
            difficulty_grid_size,
        )
        total = grid_size * grid_size

        if has_explicit_masks:
            solved_masks = normalize_int_list(
                options.get("solved_masks", frontend_config.get("solvedMasks")),
                total,
                field_name="solvedMasks",
            )
            rotations = normalize_int_list(
                options.get("rotations", frontend_config.get("rotations")),
                total,
                field_name="rotations",
            )
            source_index = clamp_int(
                options.get("source_index", frontend_config.get("sourceIndex", 0)),
                0,
                total - 1,
                0,
            )
            rotations = [rotation if int(solved_masks[index]) & 15 else 0 for index, rotation in enumerate(rotations)]
            solution_ticks = solution_ticks_for_rotations(rotations, solved_masks)
            difficulty_label = target_difficulty or difficulty_label_for_solution_ticks(solution_ticks)
            junction_count = junction_count_from_masks(solved_masks)
            max_steps = int(
                options.get(
                    "max_steps",
                    frontend_config.get("maxSteps", max_steps_for_solution_ticks(solution_ticks)),
                )
            )
        else:
            sampled = sample_pipe_instance(
                grid_size,
                seed=seed,
                difficulty=target_difficulty,
                min_solution_ticks=options.get(
                    "min_solution_ticks",
                    frontend_config.get("minSolutionTicks", frontend_config.get("solutionStepsMin")),
                ),
                max_solution_ticks=options.get(
                    "max_solution_ticks",
                    frontend_config.get("maxSolutionTicks", frontend_config.get("solutionStepsMax")),
                ),
                active_pipe_count=options.get("active_pipe_count", frontend_config.get("activePipeCount")),
                min_active_pipe_count=options.get("min_active_pipe_count", frontend_config.get("minActivePipeCount")),
                max_active_pipe_count=options.get("max_active_pipe_count", frontend_config.get("maxActivePipeCount")),
                min_junction_count=options.get("min_junction_count", frontend_config.get("minJunctionCount")),
                max_junction_count=options.get("max_junction_count", frontend_config.get("maxJunctionCount")),
            )
            source_index = int(sampled["source_index"])
            solved_masks = list(sampled["solved_masks"])
            rotations = list(sampled["rotations"])
            solution_ticks = int(sampled["solution_ticks"])
            difficulty_label = str(sampled["difficulty"])
            junction_count = int(sampled["junction_count"])
            max_steps = int(options.get("max_steps", frontend_config.get("maxSteps", sampled["max_steps"])))

        self.grid_size = grid_size
        self.action_map = list(range(total))
        self.action_space = spaces.Discrete(len(self.action_map))
        self.current_source_index = source_index
        self.current_solved_masks = [int(value) for value in solved_masks]
        self.current_rotations = [int(value) % 4 for value in rotations]
        self.current_solution_ticks = int(solution_ticks)
        self.current_difficulty = difficulty_label
        self.current_junction_count = int(junction_count)
        self.current_max_steps = max(1, int(max_steps))

        return {
            "gridSize": grid_size,
            "seed": seed,
            "sourceIndex": source_index,
            "solvedMasks": self.current_solved_masks,
            "rotations": self.current_rotations,
            "difficulty": self.current_difficulty,
            "junctionCount": self.current_junction_count,
            "maxSteps": self.current_max_steps,
            "illegalReward": self.illegal_reward,
            "moveReward": self.move_reward,
            "successReward": self.success_reward,
            "timeoutPenalty": self.timeout_penalty,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        try:
            super().reset(seed=seed)
        except TypeError:  # pragma: no cover
            super().reset()

        self._episode_steps = 0
        frontend_cfg = self._build_reset_config(seed=seed, options=options)
        frontend_result = self._run(self._evaluate("(cfg) => window.topoBench.reset(cfg)", frontend_cfg))
        obs = self.render()

        info = dict(frontend_result.get("info", {}) or {})
        info["frontend_result"] = frontend_result
        info["symbolic_observation"] = frontend_result.get("observation")
        info["state"] = info.get("state", self.get_state())
        info["success"] = bool(frontend_result.get("success", False))
        info["solution_ticks"] = self.current_solution_ticks
        info["max_steps"] = self.current_max_steps
        info["difficulty"] = self.current_difficulty
        info["junction_count"] = self.current_junction_count
        return obs, info

    def step(self, action: Any):
        self._episode_steps += 1
        action_payload: Any
        if isinstance(action, dict):
            action_payload = dict(action)
        else:
            action_payload = int(action)
        frontend_result = self.evaluate_frontend_step(action_payload)
        obs = self.render()

        reward = float(frontend_result.get("reward", 0.0))
        terminated = bool(frontend_result.get("done", False))
        truncated = False
        if self._episode_steps >= self.current_max_steps and not terminated:
            truncated = True

        info = dict(frontend_result.get("info", {}) or {})
        info["episode_steps"] = self._episode_steps
        info["symbolic_observation"] = frontend_result.get("observation")
        info["success"] = bool(frontend_result.get("success", False))
        info["state"] = info.get("state", self.get_state())
        info["solution_ticks"] = self.current_solution_ticks
        info["max_steps"] = self.current_max_steps
        info["difficulty"] = self.current_difficulty
        info["junction_count"] = self.current_junction_count
        if truncated and not info.get("reason"):
            info["reason"] = "step_budget_exhausted"
        return obs, reward, terminated, truncated, info

    def evaluate_frontend_step(self, action: Any) -> Dict[str, Any]:
        return self._run(self._evaluate("(a) => window.topoBench.step(a)", action))

    def get_frontend_state(self) -> Dict[str, Any]:
        return self._run(self._evaluate("() => window.topoBench.getState()"))

    def get_state(self) -> Dict[str, Any]:
        return self._run(self._evaluate("() => window.topoBench.getDebugState()"))

    def decode_action(self, action: int) -> int:
        action_int = int(action)
        if action_int < 0 or action_int >= len(self.action_map):
            raise ValueError(f"Invalid continuity_pipe action index: {action_int}")
        return action_int

    def screenshot(self, *, scene_only: bool = True, path: Optional[str] = None) -> bytes:
        if scene_only:
            return self._run(self._capture_scene_png(path=path))
        return self._run(self._capture_locator_png("body", path=path))

    def screenshot_board_card(self, *, path: Optional[str] = None) -> bytes:
        return self._run(self._capture_locator_png("#board-card", path=path))

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
