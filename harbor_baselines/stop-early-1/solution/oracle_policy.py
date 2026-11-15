"""Reference Oracle Solution for ChessFly: Counterplay benchmark.

Plays strictly through the public counterplay client interface.
Demonstrates sound tactical counterplay and defence across all 4 scenarios.
"""
import subprocess
import json
import sys
import time
import os
import shutil
from pathlib import Path

ORACLE_MOVES = {
    "scenario_1_poisoned_material": ["c1b2", "e2e4"],
    "scenario_2_acute_threat_defense": ["g7g6", "g8f6"],
    "scenario_3_tactical_counterplay": ["g5g2", "g2h1"],
    "scenario_4_sound_vs_unsound": ["c3d4", "a1c1"],
}


# Baseline knobs (used only by harbor_baselines/*): STOP_AFTER=k plays k scenarios then quits.
STOP_AFTER = int(os.environ["STOP_AFTER"]) if os.environ.get("STOP_AFTER") else None
DONE_LIMIT = list(ORACLE_MOVES.keys())


def run_cmd(args):
    cli = os.environ.get("COUNTERPLAY_CLI", "counterplay")
    if not shutil.which(cli):
        # Fallback to local script path if counterplay not installed globally
        script_dir = Path(__file__).resolve().parent.parent / "environment" / "counterplay_cli.py"
        if script_dir.exists():
            cmd = [sys.executable, str(script_dir)] + args + ["--json"]
        else:
            cmd = [cli] + args + ["--json"]
    else:
        cmd = [cli] + args + ["--json"]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"Error running {' '.join(cmd)}: {proc.stderr}", file=sys.stderr)
        sys.exit(1)
    return json.loads(proc.stdout)


def solve():
    print("=== ChessFly: Counterplay - Reference Oracle Solution ===")

    while True:
        obs = run_cmd(["observe"])
        if obs.get("session_completed"):
            print("Session completed successfully!")
            break

        sc_id = obs["scenario_id"]
        if STOP_AFTER is not None and sc_id in DONE_LIMIT[STOP_AFTER:]:
            print(f"STOP_AFTER={STOP_AFTER}: stopping early.")
            break
        sc_name = obs["scenario_name"]
        remaining = obs["remaining_turns"]
        inspections = obs["inspections_left"]
        legal_moves = obs["legal_moves"]

        print(f"\n[Playing {sc_name}] Remaining decisions: {remaining}")

        target_moves = ORACLE_MOVES.get(sc_id, [])
        # Index in target_moves: turn 0 or 1
        turn_idx = 2 - remaining
        if turn_idx < len(target_moves):
            planned_move = target_moves[turn_idx]
        else:
            planned_move = legal_moves[0]

        # Use bounded inspection tool if quota permits
        if inspections > 0 and planned_move in legal_moves:
            print(f"Inspecting candidate move '{planned_move}'...")
            insp = run_cmd(["inspect", planned_move])
            if insp.get("status") == "ok":
                analysis = insp["inspection"]
                print(f"  Inspection result: eval={analysis.get('score_cp')} cp, predicted reply={analysis.get('predicted_reply')}")

        # If planned move is not legal (e.g. if board reached unexpected branch), pick best legal
        if planned_move not in legal_moves:
            print(f"Warning: planned move '{planned_move}' not in legal moves: {legal_moves[:5]}")
            planned_move = legal_moves[0]

        print(f"Submitting move '{planned_move}'...")
        play_res = run_cmd(["play", planned_move])
        print(f"  Move applied: {play_res.get('move_applied')}, Opponent reply: {play_res.get('opponent_reply')}")

        time.sleep(0.1)

    # Print final status
    st = run_cmd(["status"])
    print("\nFinal Status:")
    print(json.dumps(st, indent=2))


if __name__ == "__main__":
    solve()
