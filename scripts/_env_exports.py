#!/usr/bin/env python3
"""Emit shell-quoted `export` lines for API keys found in ./.env (never echoed to the terminal;
eval'd by eval_frontier.sh). Handles `export X=..`, spaces around '=', and quotes."""
import shlex
from pathlib import Path
p = Path(".env")
if p.exists():
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k.isidentifier() and v:
            print(f"export {k}={shlex.quote(v)}")
