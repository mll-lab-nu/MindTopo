from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_QUESTION_JSONL = DEFAULT_OUTPUT_ROOT / "question.jsonl"
DEFAULT_FRONTEND_DIR = PROJECT_ROOT / "frontend"
TASK_NAME = "knots_untangle"
DEFAULT_DIFFICULTIES = "easy,medium,hard"
DEFAULT_REPEATS = 200
DEFAULT_SEED_START = 1
DEFAULT_MAX_STEPS_BY_DIFFICULTY = {"easy": 15, "medium": 15, "hard": 15}
REQUIRED_INITIAL_CROSSINGS_BY_DIFFICULTY = {
    "easy": (2, 3),
    "medium": (4, 6),
    "hard": (5, 7),
}

if str(REPO_ROOT / "environments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "environments"))

from jsonl_export import to_relative_path, write_jsonl  # noqa: E402


class _FrontendHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".ts": "text/javascript",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
    }

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class FrontendSession:
    def __init__(self, *, frontend_dir: Path, host: str, port: int, headless: bool, timeout_ms: int) -> None:
        self.frontend_dir = frontend_dir.resolve()
        self.host = host
        self.port = int(port)
        self.headless = bool(headless)
        self.timeout_ms = int(timeout_ms)
        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._playwright = None
        self._browser = None
        self._page = None

    async def __aenter__(self) -> "FrontendSession":
        index_file = self.frontend_dir / "index.html"
        if not index_file.exists():
            raise FileNotFoundError(f"Frontend index not found: {index_file}")
        self._start_server()
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-dev-shm-usage", "--no-first-run", "--no-default-browser-check"],
            timeout=self.timeout_ms,
        )
        self._page = await self._browser.new_page(viewport={"width": 1440, "height": 960}, device_scale_factor=1.0)
        await self._page.goto(self.entry_url, wait_until="load", timeout=self.timeout_ms)
        await self._page.wait_for_function(
            "() => window.topoBench && typeof window.topoBench.reset === 'function'",
            timeout=self.timeout_ms,
        )
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        if self._page is not None:
            await self._page.close()
            self._page = None
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.0)
            self._server_thread = None

    def _start_server(self) -> None:
        handler = partial(_FrontendHandler, directory=str(self.frontend_dir))
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        bound_host, bound_port = self._server.server_address[:2]
        self.entry_url = f"http://{bound_host}:{bound_port}/index.html"

    async def reset(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if self._page is None:
            raise RuntimeError("FrontendSession is not started.")
        result = await self._page.evaluate("(cfg) => window.topoBench.reset(cfg)", config)
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected reset result: {result!r}")
        return result


def parse_difficulty_list(spec: str) -> List[str]:
    values = [part.strip().lower() for part in str(spec).split(",") if part.strip()]
    if not values:
        raise ValueError("--difficulties must contain at least one difficulty.")
    invalid = [v for v in values if v not in DEFAULT_MAX_STEPS_BY_DIFFICULTY]
    if invalid:
        raise ValueError(f"Unsupported difficulties: {invalid}")
    return values


def build_episode_id(*, difficulty: str, seed: int) -> str:
    return f"{TASK_NAME}_{difficulty}_seed_{seed}"


def _build_reset_config(*, difficulty: str, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "difficulty": difficulty,
        "seed": int(seed),
        "animate": bool(args.animate),
        "illegalReward": float(args.illegal_reward),
        "moveReward": float(args.move_reward),
        "stepPenalty": float(args.step_penalty),
        "winReward": float(args.win_reward),
        "physicsFramesPerStep": int(args.physics_frames_per_step),
    }


def _build_question_row(
    *,
    output_dir: Path,
    difficulty: str,
    seed: int,
    repeat_index: int,
    reset_result: Dict[str, Any],
    reset_config: Dict[str, Any],
    max_steps: int,
) -> Dict[str, Any]:
    observation = reset_result.get("observation") or {}
    info = reset_result.get("info") or {}
    initial_state = dict(observation if observation else info.get("state", {}))
    initial_state["reset_config"] = dict(reset_config)
    return {
        "id": build_episode_id(difficulty=difficulty, seed=seed),
        "category": ["knots", TASK_NAME, "interactive"],
        "type": "episode_initialization",
        "meta_info": {
            "task_name": TASK_NAME,
            "config": to_relative_path(PROJECT_ROOT / "metadata.json", base_dir=output_dir),
            "level": difficulty,
            "seed": int(seed),
            "repeat_index": int(repeat_index),
            "difficulty": difficulty,
            "max_actions_per_traj": int(max_steps),
            "initial_state": initial_state,
            "legal_action_format": '{"answer":{"src_row":0,"src_col":0,"tgt_row":0,"tgt_col":0}}',
        },
        "images": [],
    }


def _coerce_crossing_count(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _initial_crossings(row: Dict[str, Any]) -> Optional[int]:
    meta = row.get("meta_info")
    if not isinstance(meta, dict):
        return None
    state = meta.get("initial_state")
    if not isinstance(state, dict):
        return None
    for key in ("visualCrossings", "crossings"):
        count = _coerce_crossing_count(state.get(key))
        if count is not None:
            return count
    return None


def _matches_required_initial_crossings(*, difficulty: str, row: Dict[str, Any]) -> bool:
    required_range = REQUIRED_INITIAL_CROSSINGS_BY_DIFFICULTY.get(difficulty)
    if required_range is None:
        return True
    count = _initial_crossings(row)
    if count is None:
        return False
    return required_range[0] <= count <= required_range[1]


async def generate_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    difficulties = parse_difficulty_list(args.difficulties)
    repeats = int(args.repeats)
    output_dir = Path(args.output_json).resolve().parent
    rows: List[Dict[str, Any]] = []
    next_seed = int(args.seed_start)
    total_rows = len(difficulties) * repeats
    async with FrontendSession(
        frontend_dir=Path(args.frontend_dir).resolve(),
        host=str(args.host),
        port=int(args.port),
        headless=bool(args.headless),
        timeout_ms=int(args.timeout_ms),
    ) as frontend:
        progress = tqdm(total=total_rows, desc="Knots question JSONL", unit="episode", disable=bool(args.no_progress))
        try:
            for difficulty in difficulties:
                max_steps = (
                    int(args.max_steps)
                    if args.max_steps is not None
                    else DEFAULT_MAX_STEPS_BY_DIFFICULTY[difficulty]
                )
                repeat_index = 0
                attempts = 0
                max_attempts = repeats * 100
                while repeat_index < repeats:
                    if attempts >= max_attempts:
                        raise RuntimeError(
                            f"Unable to generate {repeats} accepted {difficulty} rows "
                            f"after {attempts} attempts."
                        )
                    attempts += 1
                    reset_config = _build_reset_config(difficulty=difficulty, seed=next_seed, args=args)
                    reset_result = await frontend.reset(reset_config)
                    row = _build_question_row(
                        output_dir=output_dir,
                        difficulty=difficulty,
                        seed=next_seed,
                        repeat_index=repeat_index,
                        reset_result=reset_result,
                        reset_config=reset_config,
                        max_steps=max_steps,
                    )
                    next_seed += 1
                    if not _matches_required_initial_crossings(difficulty=difficulty, row=row):
                        progress.set_postfix(difficulty=difficulty, seed=next_seed - 1, accepted=repeat_index)
                        continue
                    rows.append(row)
                    repeat_index += 1
                    progress.set_postfix(difficulty=difficulty, seed=next_seed - 1)
                    progress.update(1)
        finally:
            progress.close()
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate knots_untangle question.jsonl without screenshots or rollouts.")
    parser.add_argument("--difficulties", default=DEFAULT_DIFFICULTIES, help="Comma-separated difficulties, e.g. easy,medium,hard.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Rows per difficulty.")
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START, help="First deterministic seed; seeds increment monotonically across difficulties.")
    parser.add_argument("--frontend-dir", default=str(DEFAULT_FRONTEND_DIR))
    parser.add_argument("--output-json", default=str(DEFAULT_QUESTION_JSONL))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--animate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--physics-frames-per-step", type=int, default=60)
    parser.add_argument("--illegal-reward", type=float, default=-1.0)
    parser.add_argument("--move-reward", type=float, default=0.1)
    parser.add_argument("--step-penalty", type=float, default=-0.01)
    parser.add_argument("--win-reward", type=float, default=1.0)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    rows = asyncio.run(generate_rows(args))
    write_jsonl(output_json, rows)
    print(f"Wrote {len(rows)} question rows to {output_json}")


if __name__ == "__main__":
    main()
