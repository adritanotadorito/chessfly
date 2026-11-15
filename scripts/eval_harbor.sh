#!/usr/bin/env bash
# One-command reproduction of the ChessFly: Counterplay Harbor baselines.
set -e
cd "$(dirname "$0")/.."
harbor run -p ./harbor_task -a oracle -o harbor_runs --job-name oracle-$(date +%s)
harbor run -p ./harbor_task -a nop    -o harbor_runs --job-name nop-$(date +%s)
harbor run -p ./harbor_baselines/stop-early-1 -a oracle -o harbor_runs --job-name stop-early-1-$(date +%s)
harbor run -p ./harbor_baselines/stop-early-2 -a oracle -o harbor_runs --job-name stop-early-2-$(date +%s)
grep -h "TOTAL VERIFIED REWARD" harbor_runs/*/*/verifier/test-stdout.txt
