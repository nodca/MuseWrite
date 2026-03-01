#!/usr/bin/env python3
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path


def _set_or_append(lines: list[str], key: str, value: str) -> list[str]:
    prefix = f"{key}="
    replaced = False
    next_lines: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            next_lines.append(f"{prefix}{value}")
            replaced = True
        else:
            next_lines.append(line)
    if not replaced:
        next_lines.append(f"{prefix}{value}")
    return next_lines


def _render_env(example_path: Path, token: str) -> str:
    raw = example_path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    lines = _set_or_append(lines, "AUTH_TOKENS", f"local-user:{token}")
    lines = _set_or_append(lines, "VITE_API_TOKEN", token)
    lines = _set_or_append(lines, "AUTH_ENABLED", "true")
    lines = _set_or_append(lines, "CONFIG_PROFILE", "local-dev")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize local .env from .env.example and generate a secure auth token.",
    )
    parser.add_argument(
        "--example",
        default=".env.example",
        help="Path to .env.example template (default: .env.example)",
    )
    parser.add_argument(
        "--output",
        default=".env",
        help="Path to output .env file (default: .env)",
    )
    parser.add_argument(
        "--token",
        default="",
        help="Optional token override. If empty, a random token is generated.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    args = parser.parse_args()

    example_path = Path(args.example).resolve()
    output_path = Path(args.output).resolve()

    if not example_path.exists():
        print(f"[init-local-env] missing template: {example_path}")
        return 2

    if output_path.exists() and not args.force:
        print(f"[init-local-env] output exists: {output_path}")
        print("[init-local-env] rerun with --force to overwrite")
        return 1

    token = args.token.strip() or secrets.token_urlsafe(32)
    rendered = _render_env(example_path, token)
    output_path.write_text(rendered, encoding="utf-8", newline="\n")

    masked = f"{token[:4]}...{token[-4:]}" if len(token) >= 8 else token
    print(f"[init-local-env] wrote {output_path}")
    print(f"[init-local-env] AUTH_TOKENS=local-user:{masked}")
    print(f"[init-local-env] VITE_API_TOKEN={masked}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
