"""Authoritative referee daemon and isolated referee service for ChessFly: Counterplay.

Enforces:
1. Strict isolation: Stockfish binary is accessible ONLY to this daemon.
2. Hard quota: max 3 candidate inspections per scenario with shallow 5k node budget.
3. Move legality: illegal moves return deterministic error and NEVER advance board.
4. Authoritative state & action log stored in protected referee path.
"""
import os
import sys
import json
import socket
import threading
from pathlib import Path
import chess
import chess.engine

from scenarios import SCENARIOS

SOCKET_PATH = os.environ.get("COUNTERPLAY_SOCKET", "/run/counterplay/referee.sock")
LOG_PATH = os.environ.get("COUNTERPLAY_LOG", "/referee/logs/session.json")
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "/referee/bin/stockfish")
IPC_DIR = os.environ.get("COUNTERPLAY_IPC_DIR", "/tmp/counterplay_ipc")

# Development fallback if stockfish is in homebrew or system path
if not os.path.exists(STOCKFISH_PATH):
    import shutil
    sys_sf = shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"
    if os.path.exists(sys_sf):
        STOCKFISH_PATH = sys_sf


class RefereeSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.scenarios = SCENARIOS
        self.current_idx = 0
        self.completed = False
        self.scenario_states = []

        for sc in self.scenarios:
            self.scenario_states.append({
                "id": sc["id"],
                "name": sc["name"],
                "side": sc["side"],
                "initial_fen": sc["fen"],
                "board": chess.Board(sc["fen"]),
                "remaining_turns": sc["agent_turns"],
                "inspections_left": sc["max_inspections"],
                "moves": [],
                "completed": False,
                "history": [],
            })

        self.save_log()

    def _get_engine(self):
        eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        eng.configure({"Threads": 1, "Hash": 16})  # deterministic with fixed node limits
        return eng

    def save_log(self):
        log_dir = os.path.dirname(LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        data = {
            "completed": self.completed,
            "current_scenario_idx": self.current_idx,
            "scenarios": [
                {
                    "id": s["id"],
                    "name": s["name"],
                    "side": s["side"],
                    "initial_fen": s["initial_fen"],
                    "current_fen": s["board"].fen(),
                    "remaining_turns": s["remaining_turns"],
                    "inspections_left": s["inspections_left"],
                    "moves": s["moves"],
                    "completed": s["completed"],
                    "history": s["history"],
                }
                for s in self.scenario_states
            ]
        }
        tmp_path = f"{LOG_PATH}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, LOG_PATH)

    def observe(self):
        with self.lock:
            if self.completed or self.current_idx >= len(self.scenarios):
                return {
                    "status": "ok",
                    "session_completed": True,
                    "message": "All scenarios completed. You may now conclude your run.",
                    "total_scenarios": len(self.scenarios),
                }

            s = self.scenario_states[self.current_idx]
            b = s["board"]
            return {
                "status": "ok",
                "session_completed": False,
                "scenario_index": self.current_idx,
                "total_scenarios": len(self.scenarios),
                "scenario_id": s["id"],
                "scenario_name": s["name"],
                "scenario_description": self.scenarios[self.current_idx]["description"],
                "side_to_move": "white" if b.turn == chess.WHITE else "black",
                "fen": b.fen(),
                "remaining_turns": s["remaining_turns"],
                "inspections_left": s["inspections_left"],
                "legal_moves": [m.uci() for m in b.legal_moves],
                "history": s["history"],
            }

    def inspect(self, uci_move: str):
        with self.lock:
            if self.completed or self.current_idx >= len(self.scenarios):
                return {"status": "error", "message": "All scenarios completed."}

            s = self.scenario_states[self.current_idx]
            if s["inspections_left"] <= 0:
                return {
                    "status": "error",
                    "message": f"Inspection quota exceeded for {s['name']}. Max {self.scenarios[self.current_idx]['max_inspections']} allowed."
                }

            b = s["board"]
            try:
                m = chess.Move.from_uci(uci_move)
            except Exception:
                return {"status": "error", "message": f"Invalid UCI move syntax: '{uci_move}'."}

            if m not in b.legal_moves:
                return {"status": "error", "message": f"Move '{uci_move}' is not legal in current position."}

            # Deduct quota
            s["inspections_left"] -= 1

            # Run shallow bounded inspection (5,000 nodes budget)
            b_next = b.copy()
            b_next.push(m)
            analysis = {"candidate": uci_move, "inspections_left": s["inspections_left"]}

            if b_next.is_game_over():
                analysis["predicted_reply"] = None
                analysis["score_cp"] = 10000 if b_next.is_checkmate() else 0
                analysis["pv"] = [uci_move]
            else:
                try:
                    engine = self._get_engine()
                    try:
                        info = engine.analyse(b_next, chess.engine.Limit(nodes=5000))
                        pv = info.get("pv", [])
                        score = info.get("score")
                        if score:
                            # From agent's perspective
                            agent_color = chess.WHITE if s["side"] == "white" else chess.BLACK
                            pov_score = score.pov(agent_color)
                            score_val = pov_score.score(mate_score=10000)
                        else:
                            score_val = 0
                        analysis["predicted_reply"] = pv[0].uci() if pv else None
                        analysis["score_cp"] = score_val
                        analysis["pv"] = [uci_move] + [p.uci() for p in pv[:3]]
                    finally:
                        engine.quit()
                except Exception as e:
                    analysis["error"] = str(e)

            self.save_log()
            return {"status": "ok", "inspection": analysis}

    def play(self, uci_move: str):
        with self.lock:
            if self.completed or self.current_idx >= len(self.scenarios):
                return {"status": "error", "message": "All scenarios completed."}

            s = self.scenario_states[self.current_idx]
            b = s["board"]

            try:
                m = chess.Move.from_uci(uci_move)
            except Exception:
                return {
                    "status": "error",
                    "message": f"Invalid UCI syntax: '{uci_move}'. Board was NOT advanced."
                }

            if m not in b.legal_moves:
                return {
                    "status": "error",
                    "message": f"Illegal move: '{uci_move}' is not in legal moves. Board was NOT advanced."
                }

            # Apply legal agent move
            b.push(m)
            s["history"].append(f"agent:{uci_move}")
            s["remaining_turns"] -= 1

            turn_record = {"agent_move": uci_move, "opponent_reply": None}

            # Opponent reply if game not over and remaining_turns > 0 (or terminal check)
            if not b.is_game_over():
                # Opponent replies using pinned Stockfish (10,000 nodes)
                engine = self._get_engine()
                try:
                    res = engine.play(b, chess.engine.Limit(nodes=10000))
                    opp_m = res.move
                    if opp_m:
                        b.push(opp_m)
                        s["history"].append(f"opponent:{opp_m.uci()}")
                        turn_record["opponent_reply"] = opp_m.uci()
                finally:
                    engine.quit()

            turn_record["fen_after"] = b.fen()
            s["moves"].append(turn_record)

            scenario_ended = (s["remaining_turns"] <= 0) or b.is_game_over()
            if scenario_ended:
                s["completed"] = True
                self.current_idx += 1
                if self.current_idx >= len(self.scenarios):
                    self.completed = True

            self.save_log()

            if self.completed:
                return {
                    "status": "ok",
                    "move_applied": uci_move,
                    "opponent_reply": turn_record["opponent_reply"],
                    "scenario_ended": True,
                    "session_completed": True,
                    "message": "Scenario completed. All scenarios finished!",
                }
            elif scenario_ended:
                next_s = self.scenario_states[self.current_idx]
                return {
                    "status": "ok",
                    "move_applied": uci_move,
                    "opponent_reply": turn_record["opponent_reply"],
                    "scenario_ended": True,
                    "session_completed": False,
                    "next_scenario": {
                        "index": self.current_idx,
                        "id": next_s["id"],
                        "name": next_s["name"],
                        "side": next_s["side"],
                        "fen": next_s["board"].fen(),
                        "remaining_turns": next_s["remaining_turns"],
                    }
                }
            else:
                return {
                    "status": "ok",
                    "move_applied": uci_move,
                    "opponent_reply": turn_record["opponent_reply"],
                    "scenario_ended": False,
                    "remaining_turns": s["remaining_turns"],
                    "fen": b.fen(),
                    "legal_moves": [lm.uci() for lm in b.legal_moves],
                }

    def status(self):
        with self.lock:
            return {
                "status": "ok",
                "completed": self.completed,
                "current_scenario_idx": self.current_idx,
                "total_scenarios": len(self.scenarios),
                "scenarios": [
                    {
                        "id": s["id"],
                        "name": s["name"],
                        "completed": s["completed"],
                        "remaining_turns": s["remaining_turns"],
                        "inspections_left": s["inspections_left"],
                        "moves_count": len(s["moves"]),
                    }
                    for s in self.scenario_states
                ]
            }


def execute_request(req: dict, session: RefereeSession) -> dict:
    cmd = req.get("command")
    if cmd == "observe":
        return session.observe()
    elif cmd == "inspect":
        return session.inspect(req.get("move", ""))
    elif cmd == "play":
        return session.play(req.get("move", ""))
    elif cmd == "status":
        return session.status()
    else:
        return {"status": "error", "message": f"Unknown command: '{cmd}'."}


def handle_client(conn, session: RefereeSession):
    try:
        data = conn.recv(16384).decode("utf-8")
        if not data:
            return
        req = json.loads(data)
        res = execute_request(req, session)
        conn.sendall(json.dumps(res).encode("utf-8"))
    except Exception as e:
        err = {"status": "error", "message": str(e)}
        try:
            conn.sendall(json.dumps(err).encode("utf-8"))
        except Exception:
            pass
    finally:
        conn.close()


def run_file_ipc_worker(session: RefereeSession, stop_event: threading.Event):
    inbox = os.path.join(IPC_DIR, "inbox")
    outbox = os.path.join(IPC_DIR, "outbox")
    os.makedirs(inbox, exist_ok=True)
    os.makedirs(outbox, exist_ok=True)
    try:
        os.chmod(IPC_DIR, 0o777)
        os.chmod(inbox, 0o777)
        os.chmod(outbox, 0o777)
    except Exception:
        pass

    import time
    while not stop_event.is_set():
        try:
            entries = sorted(os.listdir(inbox))
            if not entries:
                time.sleep(0.02)
                continue

            for fname in entries:
                if not fname.endswith(".json"):
                    continue
                req_path = os.path.join(inbox, fname)
                try:
                    with open(req_path, "r") as f:
                        req = json.load(f)
                except Exception:
                    continue

                res = execute_request(req, session)
                tmp_out = os.path.join(outbox, f"tmp_{fname}")
                final_out = os.path.join(outbox, fname)
                with open(tmp_out, "w") as f:
                    json.dump(res, f)
                try:
                    os.chmod(tmp_out, 0o666)
                except Exception:
                    pass
                os.replace(tmp_out, final_out)

                try:
                    os.remove(req_path)
                except Exception:
                    pass
        except Exception:
            time.sleep(0.05)


def run_daemon():
    session = RefereeSession()
    stop_event = threading.Event()

    # Always start File IPC worker
    t_ipc = threading.Thread(target=run_file_ipc_worker, args=(session, stop_event))
    t_ipc.daemon = True
    t_ipc.start()
    print(f"[Referee Daemon] File IPC active at {IPC_DIR}")

    # Also bind UNIX socket if permitted by environment
    server = None
    try:
        sock_dir = os.path.dirname(SOCKET_PATH)
        if sock_dir:
            os.makedirs(sock_dir, exist_ok=True)
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        os.chmod(SOCKET_PATH, 0o666)
        server.listen(16)
        print(f"[Referee Daemon] UNIX socket active at {SOCKET_PATH}")
    except Exception as e:
        print(f"[Referee Daemon] Notice: UNIX socket disabled ({e}); relying on File IPC.")

    if server:
        while True:
            try:
                conn, _ = server.accept()
                t = threading.Thread(target=handle_client, args=(conn, session))
                t.daemon = True
                t.start()
            except Exception:
                break
    else:
        # Keep main thread alive for File IPC worker
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop_event.set()


if __name__ == "__main__":
    run_daemon()
