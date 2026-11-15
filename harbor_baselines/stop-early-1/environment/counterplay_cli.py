#!/usr/bin/env python3
"""Command-line client for ChessFly: Counterplay benchmark.

Usage:
  counterplay observe [--json]
  counterplay inspect <uci_move> [--json]
  counterplay play <uci_move> [--json]
  counterplay status [--json]
"""
import os
import sys
import json
import socket
import argparse

SOCKET_PATH = os.environ.get("COUNTERPLAY_SOCKET", "/run/counterplay/referee.sock")
IPC_DIR = os.environ.get("COUNTERPLAY_IPC_DIR", "/tmp/counterplay_ipc")


def send_command(req: dict) -> dict:
    # 1. Try UNIX domain socket if available
    if os.path.exists(SOCKET_PATH):
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(SOCKET_PATH)
            s.sendall(json.dumps(req).encode("utf-8"))
            data = b""
            while True:
                chunk = s.recv(16384)
                if not chunk:
                    break
                data += chunk
            s.close()
            return json.loads(data.decode("utf-8"))
        except Exception:
            pass

    # 2. Fallback to File IPC channel
    inbox = os.path.join(IPC_DIR, "inbox")
    outbox = os.path.join(IPC_DIR, "outbox")
    if not (os.path.exists(inbox) and os.path.exists(outbox)):
        print(f"Error: Counterplay daemon is not running (checked socket {SOCKET_PATH} and IPC {IPC_DIR}).", file=sys.stderr)
        sys.exit(1)

    import uuid, time
    req_id = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.json"
    req_file = os.path.join(inbox, req_id)
    resp_file = os.path.join(outbox, req_id)

    # Write request atomically
    tmp_req = f"{req_file}.tmp"
    with open(tmp_req, "w") as f:
        json.dump(req, f)
    try:
        os.chmod(tmp_req, 0o666)
    except Exception:
        pass
    os.replace(tmp_req, req_file)

    # Poll for response (timeout 15s)
    t0 = time.time()
    while time.time() - t0 < 15.0:
        if os.path.exists(resp_file):
            try:
                with open(resp_file, "r") as f:
                    res = json.load(f)
                os.remove(resp_file)
                return res
            except Exception:
                pass
        time.sleep(0.02)

    print("Error: Counterplay referee response timed out.", file=sys.stderr)
    sys.exit(1)


def print_observe(res: dict, as_json: bool):
    if as_json:
        print(json.dumps(res, indent=2))
        return

    if res.get("session_completed"):
        print("\n=== SESSION COMPLETED ===")
        print(res.get("message"))
        return

    print("\n" + "=" * 60)
    print(f"[{res['scenario_index'] + 1}/{res['total_scenarios']}] {res['scenario_name']}")
    print("=" * 60)
    print(f"Goal: {res['scenario_description']}")
    print(f"Side to move: {res['side_to_move'].upper()}")
    print(f"FEN: {res['fen']}")
    print(f"Remaining Agent Decisions in this scenario: {res['remaining_turns']}")
    print(f"Inspection Quota remaining: {res['inspections_left']}")
    if res.get("history"):
        print(f"Move History: {' -> '.join(res['history'])}")
    print("\nLegal Moves (UCI):")
    legals = res.get("legal_moves", [])
    # Format nicely in rows
    chunks = [legals[i:i + 8] for i in range(0, len(legals), 8)]
    for chunk in chunks:
        print("  " + "  ".join(f"{m:5s}" for m in chunk))
    print()


def print_inspect(res: dict, as_json: bool):
    if as_json:
        print(json.dumps(res, indent=2))
        return

    if res.get("status") != "ok":
        print(f"\n[INSPECT ERROR] {res.get('message')}")
        return

    insp = res["inspection"]
    print("\n--- Bounded Candidate Line Inspection ---")
    print(f"Candidate Move: {insp['candidate']}")
    print(f"Inspections Left: {insp['inspections_left']}")
    print(f"Evaluation (cp, agent perspective): {insp.get('score_cp', 'N/A')}")
    print(f"Predicted Opponent Reply: {insp.get('predicted_reply', 'None')}")
    if insp.get("pv"):
        print(f"Principal Line: {' -> '.join(insp['pv'])}")
    print()


def print_play(res: dict, as_json: bool):
    if as_json:
        print(json.dumps(res, indent=2))
        return

    if res.get("status") != "ok":
        print(f"\n[PLAY ERROR] {res.get('message')}")
        return

    print("\n>>> Move Applied Successfully <<<")
    print(f"Your Move: {res.get('move_applied')}")
    if res.get("opponent_reply"):
        print(f"Opponent Reply: {res.get('opponent_reply')}")
    else:
        print("Opponent Reply: None (Terminal position)")

    if res.get("session_completed"):
        print("\n" + "=" * 60)
        print("ALL SCENARIOS COMPLETED!")
        print("Session is now closed. Verifier will score your completed run.")
        print("=" * 60)
    elif res.get("scenario_ended"):
        print("\n" + "-" * 60)
        print("Scenario complete! Advancing to next scenario...")
        next_sc = res.get("next_scenario", {})
        print(f"Next: [{next_sc.get('index', 0) + 1}] {next_sc.get('name')}")
        print(f"FEN: {next_sc.get('fen')}")
        print("-" * 60)
    else:
        print(f"Remaining decisions in scenario: {res.get('remaining_turns')}")
        print(f"Current FEN: {res.get('fen')}")
        print(f"Legal moves: {' '.join(res.get('legal_moves', []))}")


def print_status(res: dict, as_json: bool):
    if as_json:
        print(json.dumps(res, indent=2))
        return

    print("\n=== Counterplay Session Status ===")
    print(f"Overall Completed: {res.get('completed')}")
    print(f"Current Scenario: {res.get('current_scenario_idx', 0) + 1}/{res.get('total_scenarios', 4)}")
    for sc in res.get("scenarios", []):
        st = "DONE" if sc["completed"] else "IN PROGRESS"
        print(f"  - {sc['name']:36s}: [{st}] (moves: {sc['moves_count']}, inspections left: {sc['inspections_left']})")
    print()


def main():
    p = argparse.ArgumentParser(description="ChessFly: Counterplay CLI")
    p.add_argument("--json", action="store_true", help="Output raw JSON")
    sub = p.add_subparsers(dest="cmd", required=True)

    obs_p = sub.add_parser("observe", help="Get current board, legal moves, remaining turns, and quota")
    obs_p.add_argument("--json", action="store_true", help="Output raw JSON")

    insp_p = sub.add_parser("inspect", help="Inspect candidate line (bounded quota)")
    insp_p.add_argument("move", help="Legal UCI move to inspect (e.g. a1b1)")
    insp_p.add_argument("--json", action="store_true", help="Output raw JSON")

    play_p = sub.add_parser("play", help="Submit a legal UCI move")
    play_p.add_argument("move", help="Legal UCI move to play (e.g. a1b1)")
    play_p.add_argument("--json", action="store_true", help="Output raw JSON")

    st_p = sub.add_parser("status", help="Get overall scenario completion status")
    st_p.add_argument("--json", action="store_true", help="Output raw JSON")

    args = p.parse_args()
    as_json = args.json

    if args.cmd == "observe":
        res = send_command({"command": "observe"})
        print_observe(res, as_json)
    elif args.cmd == "inspect":
        res = send_command({"command": "inspect", "move": args.move})
        print_inspect(res, as_json)
    elif args.cmd == "play":
        res = send_command({"command": "play", "move": args.move})
        print_play(res, as_json)
    elif args.cmd == "status":
        res = send_command({"command": "status"})
        print_status(res, as_json)


if __name__ == "__main__":
    main()
