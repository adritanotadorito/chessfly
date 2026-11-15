"""Independent authoritative verifier for ChessFly: Counterplay benchmark.

Reads the protected referee session log, replays all moves using python-chess,
evaluates position quality with Stockfish, and outputs exactly one number in [0, 1]
to /logs/verifier/reward.txt.
"""
import os
import sys
import json
import shutil
from pathlib import Path
import chess
import chess.engine

LOG_PATH = os.environ.get("COUNTERPLAY_LOG", "/referee/logs/session.json")
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "/referee/bin/stockfish")
REWARD_OUTPUT = os.environ.get("VERIFIER_REWARD_PATH", "/logs/verifier/reward.txt")

if not os.path.exists(STOCKFISH_PATH):
    sys_sf = shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"
    if os.path.exists(sys_sf):
        STOCKFISH_PATH = sys_sf


def evaluate_scenario_performance(sc_data: dict, engine: chess.engine.SimpleEngine) -> dict:
    sc_id = sc_data["id"]
    side = chess.WHITE if sc_data["side"] == "white" else chess.BLACK
    initial_fen = sc_data["initial_fen"]
    moves = sc_data.get("moves", [])

    # If scenario was not completed or has no moves -> 0.0
    if not sc_data.get("completed", False) or len(moves) < 2:
        return {"scenario_id": sc_id, "score": 0.0, "reason": "incomplete_or_stopped_early"}

    # Replay moves independently on a fresh board
    b = chess.Board(initial_fen)
    for m_record in moves:
        agent_uci = m_record.get("agent_move")
        if not agent_uci:
            return {"scenario_id": sc_id, "score": 0.0, "reason": "missing_agent_move"}
        try:
            m = chess.Move.from_uci(agent_uci)
            if m not in b.legal_moves:
                return {"scenario_id": sc_id, "score": 0.0, "reason": f"illegal_move_{agent_uci}"}
            b.push(m)
        except Exception as e:
            return {"scenario_id": sc_id, "score": 0.0, "reason": f"invalid_move_syntax_{e}"}

        opp_uci = m_record.get("opponent_reply")
        if opp_uci:
            try:
                opp_m = chess.Move.from_uci(opp_uci)
                if opp_m in b.legal_moves:
                    b.push(opp_m)
            except Exception:
                pass

    # Check terminal status
    if b.is_checkmate():
        # Did the agent win or get checkmated?
        if b.turn != side:
            # Agent delivered checkmate!
            return {"scenario_id": sc_id, "score": 0.25, "reason": "checkmate_delivered", "cp": 10000}
        else:
            # Agent got checkmated! Outright blunder
            return {"scenario_id": sc_id, "score": 0.0, "reason": "checkmated_by_opponent", "cp": -10000}

    # Evaluate final board with Stockfish (15,000 nodes budget)
    try:
        info = engine.analyse(b, chess.engine.Limit(nodes=15000))
        score_obj = info.get("score")
        if score_obj:
            pov_score = score_obj.pov(side)
            cp = pov_score.score(mate_score=10000)
        else:
            cp = 0
    except Exception as e:
        cp = 0

    # Specific tactical criteria per scenario
    if sc_id == "scenario_1_poisoned_material":
        # White traps/punishes Black's Queen on b2
        # If White played Rb1 and won the Queen or forced concessions -> cp >= +200
        if cp >= 250:
            score = 0.25
        elif cp >= 100:
            score = 0.20
        elif cp >= 0:
            score = 0.10
        else:
            score = 0.0  # Fell for poisoned trap / lost rook on a1
    elif sc_id == "scenario_2_acute_threat_defense":
        # Black must defend f7 against Qxf7#
        # If defended accurately, Black has equalized or has better position (cp >= 0)
        if cp >= 0:
            score = 0.25
        elif cp >= -100:
            score = 0.15
        elif cp >= -200:
            score = 0.05
        else:
            score = 0.0  # Scholar's mate conceded or massive blunder
    elif sc_id == "scenario_3_tactical_counterplay":
        # Black refutes 5. Nxf7? with 5... Qxg2!
        # If counterplay succeeded -> Black is winning (cp >= +150)
        if cp >= 250:
            score = 0.25
        elif cp >= 100:
            score = 0.20
        elif cp >= 0:
            score = 0.08
        else:
            score = 0.0  # Passive or blundered f7
    elif sc_id == "scenario_4_sound_vs_unsound":
        # White chooses sound central consolidation over unsound sacrifice
        if cp >= 130:
            score = 0.25
        elif cp >= 50:
            score = 0.20
        elif cp >= 0:
            score = 0.10
        else:
            score = 0.0  # Sacrificed unsoundly or dropped piece
    else:
        # Default scaling
        if cp >= 200:
            score = 0.25
        elif cp >= 0:
            score = 0.125
        else:
            score = 0.0

    return {"scenario_id": sc_id, "score": score, "cp": cp, "reason": "evaluated"}


def verify():
    print("=" * 60)
    print("ChessFly: Counterplay - Independent Verifier")
    print("=" * 60)

    if not os.path.exists(LOG_PATH):
        print(f"Error: Authoritative log not found at {LOG_PATH}.")
        final_score = 0.0
        write_reward(final_score)
        return

    try:
        with open(LOG_PATH, "r") as f:
            session_data = json.load(f)
    except Exception as e:
        print(f"Error reading session log: {e}")
        write_reward(0.0)
        return

    scenarios = session_data.get("scenarios", [])
    if not scenarios:
        print("No scenarios recorded in session.")
        write_reward(0.0)
        return

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 1, "Hash": 16})
    engine_name = engine.id.get("name")
    print(f"Engine: {engine.id.get('name')} at {STOCKFISH_PATH} (Threads=1, Hash=16, nodes=15000)")
    total_score = 0.0
    details = []

    try:
        for sc in scenarios:
            res = evaluate_scenario_performance(sc, engine)
            total_score += res["score"]
            details.append(res)
            print(f"[{sc['id']:32s}] Score: {res['score']:.4f} | Reason: {res.get('reason')} | CP: {res.get('cp', 'N/A')}")
    finally:
        engine.quit()

    # Completion gate: stopping early is penalised quadratically.
    # 1/4 scenarios completed keeps 25% of its earned credit, 2/4 keeps 50%, etc.
    n_done = sum(1 for d in details if d.get("reason") != "incomplete_or_stopped_early")
    gate = n_done / 4.0
    print(f"Completed scenarios: {n_done}/4 -> completion gate x{gate:.2f}")
    final_score = max(0.0, min(1.0, round(total_score * gate, 4)))
    try:
        Path(os.path.dirname(REWARD_OUTPUT) or ".").mkdir(parents=True, exist_ok=True)
        with open(os.path.join(os.path.dirname(REWARD_OUTPUT), "details.json"), "w") as f:
            json.dump({"engine": engine_name, "completed": n_done, "gate": gate,
                       "raw_sum": total_score, "reward": final_score, "scenarios": details}, f, indent=2)
    except Exception as e:
        print(f"details.json not written: {e}")
    print("-" * 60)
    print(f"TOTAL VERIFIED REWARD: {final_score:.4f} / 1.0000")
    print("-" * 60)

    write_reward(final_score)


def write_reward(score: float):
    out_dir = os.path.dirname(REWARD_OUTPUT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(REWARD_OUTPUT, "w") as f:
        f.write(f"{score:.4f}\n")
    print(f"Written reward {score:.4f} to {REWARD_OUTPUT}")


if __name__ == "__main__":
    verify()
