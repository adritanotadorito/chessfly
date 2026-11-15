#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$DIR/../../.venv/bin/python" ]; then
    "$DIR/../../.venv/bin/python" "$DIR/oracle_policy.py"
else
    python3 "$DIR/oracle_policy.py"
fi
