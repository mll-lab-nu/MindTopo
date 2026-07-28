from __future__ import annotations

import argparse
import time
from pathlib import Path

from vite_server import ViteFrontendServer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the enclosure_hole_detection frontend through a Vite dev server."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--frontend-dir",
        default=str(Path(__file__).resolve().parent.parent / "frontend"),
        help="Directory containing the Vite frontend.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    frontend_dir = Path(args.frontend_dir).resolve()
    server = ViteFrontendServer(frontend_dir=frontend_dir, host=args.host, port=args.port)
    base_url = server.start()

    print(f"Serving frontend from {frontend_dir}")
    print(f"Open: {base_url}/")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
