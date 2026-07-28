from __future__ import annotations

import argparse
import time
from pathlib import Path

from vite_server import ViteFrontendServer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the separation_one_stroke frontend with Vite.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--frontend-dir",
        default=str(Path(__file__).resolve().parent.parent / "frontend"),
        help="Directory containing the frontend package.json.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    server = ViteFrontendServer(frontend_dir=Path(args.frontend_dir), host=args.host, port=args.port)
    try:
        server.start()
        print(f"Serving frontend from {server.frontend_dir}")
        print(f"Open: {server.base_url}/")
        print("Press Ctrl+C to stop.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
