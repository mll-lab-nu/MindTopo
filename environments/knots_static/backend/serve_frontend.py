from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the repository root so the knots_static frontend can load assets."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    handler = partial(_RepoHandler, directory=str(repo_root))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    base_url = f"http://{args.host}:{server.server_address[1]}"
    viewer_url = f"{base_url}/knots_static/frontend/index.html"

    print(f"Serving repo root from {repo_root}")
    print(f"Open: {viewer_url}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
