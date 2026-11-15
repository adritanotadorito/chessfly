#!/usr/bin/env bash
# Baseline: real oracle moves, but quits after 1 of 4 scenarios.
STOP_AFTER=1 python3 "$(dirname "$0")/oracle_policy.py"
