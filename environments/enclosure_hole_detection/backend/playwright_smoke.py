from __future__ import annotations

import argparse
import asyncio
import base64
from pathlib import Path

from generate_samples import label_state_image, scene_config
from vite_server import ViteFrontendServer

try:
    from playwright.async_api import async_playwright
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "playwright_smoke.py requires the `playwright` package. Install it with: pip install playwright"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
OUTPUT_DIR = PROJECT_ROOT / "output"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the enclosure_hole_detection frontend.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with ViteFrontendServer(frontend_dir=FRONTEND_DIR, host=args.host, port=args.port) as server:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=args.headless)
            page = await browser.new_page(viewport={"width": 1400, "height": 900})
            await page.goto(server.base_url, wait_until="networkidle")
            await page.wait_for_function("() => !!window.topoBench && typeof window.topoBench.generate === 'function'")

            metadata = await page.evaluate(
                """async (config) => {
                    return await window.topoBench.generate(config);
                }""",
                scene_config(args.seed, args.difficulty),
            )
            screenshot_data_url = await page.evaluate("() => window.topoBench.screenshot()")
            png_bytes = label_state_image(base64.b64decode(screenshot_data_url.split(",", 1)[1]))
            output_path = OUTPUT_DIR / f"smoke_seed_{args.seed}_difficulty_{args.difficulty}.png"
            output_path.write_bytes(png_bytes)

            print(
                f"seed={metadata['seed']} difficulty={metadata['difficulty']} "
                f"through={metadata['visible_through_hole_count']} pits={metadata['visible_pit_count']} "
                f"features={metadata['visible_feature_count']} output={output_path}"
            )
            await browser.close()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
