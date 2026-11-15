#!/usr/bin/env bash
set -e

echo "[Verifier] Running ChessFly Counterplay Verification..."

mkdir -p /logs/verifier

# Run the authoritative verifier
if [ -f "/referee/verifier.py" ]; then
    python3 /referee/verifier.py
else
    DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -x "$DIR/../../.venv/bin/python" ]; then
        "$DIR/../../.venv/bin/python" "$DIR/verifier.py"
    else
        python3 "$DIR/verifier.py"
    fi
fi

if [ -f "/logs/verifier/reward.txt" ]; then
    echo "[Verifier] Successfully generated reward:"
    cat /logs/verifier/reward.txt
else
    echo "0.0000" > /logs/verifier/reward.txt
    echo "[Verifier] Fallback reward written: 0.0000"
fi
