from __future__ import annotations

import argparse
import asyncio
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlencode

from playwright.async_api import async_playwright


ROOT = Path(__file__).resolve().parents[2]
ENTRY_PATH = "/separation_objects/frontend/index.html"
DEFAULT_SCREENSHOT = ROOT / "separation_objects" / "output" / "smoke.png"


class _RepoHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".ts": "text/javascript",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".obj": "text/plain",
    }

    def log_message(self, _format: str, *_args) -> None:
        return


def start_server(host: str, port: int) -> tuple[ThreadingHTTPServer, Thread, str]:
    handler = partial(_RepoHandler, directory=str(ROOT))
    server = ThreadingHTTPServer((host, port), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://{host}:{server.server_address[1]}"
    return server, thread, base_url


async def run_smoke(args) -> None:
    server, thread, base_url = start_server(args.host, args.port)
    screenshot_path = Path(args.screenshot).resolve()
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)

    params = urlencode(
        {
            "category": args.category,
            "object": args.object_name,
            "seed": args.seed,
        }
    )
    page_url = f"{base_url}{ENTRY_PATH}?{params}"

    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=args.headless,
                args=["--disable-dev-shm-usage", "--no-first-run", "--no-default-browser-check"],
            )
            page = await browser.new_page(viewport={"width": 1600, "height": 1280}, device_scale_factor=1.0)
            await page.goto(page_url, wait_until="load")
            await page.wait_for_function(
                "() => window.topoBench && typeof window.topoBench.generate === 'function'",
                timeout=120000,
            )
            await page.wait_for_function(
                "() => window.__sceneReady && window.__sceneReady.optionCount > 0",
                timeout=120000,
            )
            await page.wait_for_function(
                "(count) => document.querySelectorAll('.view-image').length === count + 1",
                arg=await page.evaluate("() => window.__sceneReady.optionCount"),
                timeout=120000,
            )
            await page.locator("#interactive-viewer img").wait_for(state="visible", timeout=120000)
            await page.screenshot(path=str(screenshot_path), full_page=True)
            ready = await page.evaluate("() => window.__sceneReady")
            print(
                f"Loaded: {ready['objectCategory']}/{ready['objectName']} "
                f"parts={ready['partCount']} options={ready['optionCount']}"
            )
            print(f"Screenshot: {screenshot_path}")
            await browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Playwright smoke test for the separation_objects reasoning renderer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--category", default="Bench")
    parser.add_argument("--object-name", default="applaro")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--screenshot", default=str(DEFAULT_SCREENSHOT))
    parser.add_argument("--headless", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(run_smoke(args))


if __name__ == "__main__":
    main()
