#!/usr/bin/env bash
# Runs as root in the shared container. Replays the referee-owned move log
# (/referee/logs/session.json, writable only by UID 1001) with Stockfish and
# writes exactly one number in [0,1] to /logs/verifier/reward.txt.
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.txt
python3 /referee/verifier.py || echo "0.0000" > /logs/verifier/reward.txt
[ -s /logs/verifier/reward.txt ] || echo "0.0000" > /logs/verifier/reward.txt
echo "[Verifier] reward: $(cat /logs/verifier/reward.txt)"
