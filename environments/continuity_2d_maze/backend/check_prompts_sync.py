"""Guard that REASONING_TEMPLATE / BAR_REMOVAL_TEMPLATE stay byte-equal
between backend/prompts.py and frontend/src/maze/prompts.ts.

Strategy: the TS file writes each template as a chain of single-quoted
string literals concatenated with `+`. We grep the `export const NAME =
(...);` block, extract every `'...'` in order (honoring `\\'` / `\\\\`
escapes and common `\\n` / `\\t` escapes used in the templates), concat
them, then md5-compare against the Python value.

Run manually: `python backend/check_prompts_sync.py`
Exit 0 on match, 1 on mismatch with a unified-diff-style preview.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import prompts  # backend/prompts.py

BACKEND_DIR = Path(__file__).resolve().parent
REPO_DIR = BACKEND_DIR.parent
TS_PATH = REPO_DIR / "frontend" / "src" / "maze" / "prompts.ts"

TEMPLATE_NAMES = ("REACHABILITY_SET_TEMPLATE", "BAR_REMOVAL_TEMPLATE")


# Single-quoted TS string literal; handles escapes via non-greedy match with
# a lookbehind that rejects `\'` (an escaped quote inside the literal). We
# evaluate common backslash escapes after extraction.
_LITERAL_RE = re.compile(r"'((?:\\.|[^'\\])*)'", re.DOTALL)

_SIMPLE_ESCAPES = {
    "\\": "\\",
    "'": "'",
    '"': '"',
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "0": "\0",
}


def unescape_ts_literal(raw: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(raw):
        c = raw[i]
        if c == "\\" and i + 1 < len(raw):
            nxt = raw[i + 1]
            if nxt in _SIMPLE_ESCAPES:
                out.append(_SIMPLE_ESCAPES[nxt])
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


def extract_ts_template(ts_source: str, name: str) -> str:
    # Locate `export const NAME =` and then read forward until the first
    # semicolon that is OUTSIDE any string literal (template strings contain
    # semicolons as prose, e.g. "walls; they block passage", which would
    # fool a naive non-greedy regex).
    header = re.search(
        rf"export const {re.escape(name)}\s*(?::[^=]+)?=",
        ts_source,
    )
    if not header:
        raise RuntimeError(f"TS file missing `export const {name}`")
    i = header.end()
    n = len(ts_source)
    in_string = False
    end = None
    while i < n:
        c = ts_source[i]
        if in_string:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == "'":
                in_string = False
                i += 1
                continue
            i += 1
            continue
        if c == "'":
            in_string = True
            i += 1
            continue
        if c == ";":
            end = i
            break
        i += 1
    if end is None:
        raise RuntimeError(f"TS {name} never terminates with a semicolon")
    rhs = ts_source[header.end() : end]
    parts = _LITERAL_RE.findall(rhs)
    if not parts:
        raise RuntimeError(f"No string literals found in TS {name}")
    return "".join(unescape_ts_literal(p) for p in parts)


def _diff_preview(a: str, b: str, limit: int = 400) -> str:
    # Show the first diverging character plus short context; good enough
    # to spot whitespace / typo drift in reviews.
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            lo = max(0, i - 40)
            hi = i + 40
            return (
                f"first diff at char {i}\n"
                f"  backend ... {a[lo:hi]!r}\n"
                f"  frontend... {b[lo:hi]!r}"
            )
    if len(a) != len(b):
        shorter, longer = (a, b) if len(a) < len(b) else (b, a)
        tail = longer[len(shorter):][:limit]
        return (
            f"length differs (py={len(a)}, ts={len(b)}); tail-only side:\n"
            f"  {tail!r}"
        )
    return "(no diff detected; check the md5 values above)"


def main() -> int:
    if not TS_PATH.exists():
        print(f"FATAL: {TS_PATH} not found", file=sys.stderr)
        return 1
    ts_source = TS_PATH.read_text(encoding="utf-8")

    failed = 0
    for name in TEMPLATE_NAMES:
        py_value = getattr(prompts, name, None)
        if not isinstance(py_value, str):
            print(f"FAIL  backend prompts.py missing {name}")
            failed += 1
            continue
        try:
            ts_value = extract_ts_template(ts_source, name)
        except RuntimeError as exc:
            print(f"FAIL  {name}: {exc}")
            failed += 1
            continue
        # Python templates use `.format()` so literal braces are written
        # doubled (`{{ans}}` → `{ans}` post-format); the TS side writes the
        # post-format form directly because it uses `.replace()`. Normalize
        # Python to the post-format form before comparing; the single-braced
        # `{question}` / `{images}` placeholders survive either way.
        py_normalized = py_value.replace("{{", "{").replace("}}", "}")
        py_md5 = hashlib.md5(py_normalized.encode("utf-8")).hexdigest()
        ts_md5 = hashlib.md5(ts_value.encode("utf-8")).hexdigest()
        if py_md5 == ts_md5:
            print(f"OK    {name}  md5={py_md5}")
        else:
            print(f"FAIL  {name}")
            print(f"        py  md5={py_md5}  len={len(py_normalized)}")
            print(f"        ts  md5={ts_md5}  len={len(ts_value)}")
            print("      " + _diff_preview(py_normalized, ts_value).replace("\n", "\n      "))
            failed += 1

    if failed:
        print(f"\n{failed} template(s) out of sync.")
        return 1
    print("\nall templates in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
