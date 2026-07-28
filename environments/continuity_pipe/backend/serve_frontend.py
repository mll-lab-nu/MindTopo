from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _FrontendHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".ts": "text/javascript",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
    }

    def log_message(self, _format: str, *_args) -> None:
        return


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the Continuity Pipe frontend with correct MIME types for .ts modules."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--frontend-dir",
        default=str(Path(__file__).resolve().parent.parent / "frontend"),
        help="Directory containing the frontend assets.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    frontend_dir = Path(args.frontend_dir).resolve()
    if not frontend_dir.exists():
        raise FileNotFoundError(f"Frontend directory not found: {frontend_dir}")

    handler = partial(_FrontendHandler, directory=str(frontend_dir))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    base_url = f"http://{args.host}:{server.server_address[1]}"

    print(f"Serving frontend from {frontend_dir}")
    print(f"Open: {base_url}/")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
