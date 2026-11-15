"""Real-time match runner supporting 1+0 bullet and untimed games with live streaming."""
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import chess
import chess.engine
import chess.pgn
import numpy as np

from .games import save_game
from .model import FlyPolicy
from .players import (BasePlayer, FlyPolicyPlayer, HumanPlayer,
                     StockfishPlayer, find_stockfish)


def format_clock(seconds: Optional[float]) -> str:
    """Format seconds into standard clock format M:SS or MM:SS."""
    if seconds is None:
        return "∞"
    sec = max(0.0, seconds)
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"


# Global thread-safe registry of live matches
_ACTIVE_LOCK = threading.Lock()
_CURRENT_MATCH: Optional["ChessMatch"] = None


def get_current_match() -> Optional["ChessMatch"]:
    with _ACTIVE_LOCK:
        return _CURRENT_MATCH


def set_current_match(match: Optional["ChessMatch"]):
    global _CURRENT_MATCH
    with _ACTIVE_LOCK:
        _CURRENT_MATCH = match


class ChessMatch:
    def __init__(self,
                 white_player: BasePlayer,
                 black_player: BasePlayer,
                 timed: bool = True,
                 base_time: float = 60.0,
                 increment: float = 0.0,
                 game_id: Optional[str] = None):
        self.white_player = white_player
        self.black_player = black_player
        self.timed = timed
        self.base_time = base_time
        self.increment = increment

        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.game_id = game_id or f"match-{timestamp_slug}-{uuid.uuid4().hex[:6]}"

        self.board = chess.Board()
        self.white_clock = base_time if timed else None
        self.black_clock = base_time if timed else None

        self.moves_log: List[Dict[str, Any]] = []
        self.game_over = False
        self.result = "*"
        self.termination_reason = "In progress"
        self.winner: Optional[str] = None  # "white", "black", "draw"

        # Fly mascot visual state: "idle", "thinking", "happy", "upset"
        self.fly_state = "idle"
        self.last_move: Optional[str] = None

        self._pending_human_move: Optional[chess.Move] = None
        self._human_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Stockfish engine instance for bot moves if needed
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def get_state(self) -> Dict[str, Any]:
        """Thread-safe snapshot of live game state."""
        return {
            "game_id": self.game_id,
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "white_player": self.white_player.name,
            "black_player": self.black_player.name,
            "white_clock": round(self.white_clock, 2) if self.white_clock is not None else None,
            "black_clock": round(self.black_clock, 2) if self.black_clock is not None else None,
            "formatted_white_clock": format_clock(self.white_clock),
            "formatted_black_clock": format_clock(self.black_clock),
            "timed": self.timed,
            "is_game_over": self.game_over,
            "result": self.result,
            "winner": self.winner,
            "termination": self.termination_reason,
            "ply_count": len(self.moves_log),
            "last_move": self.last_move,
            "fly_state": self.fly_state,
            "moves": list(self.moves_log),
            "legal_moves": [m.uci() for m in self.board.legal_moves] if not self.game_over else [],
            "is_human_turn": (self.white_player.is_human if self.board.turn == chess.WHITE else self.black_player.is_human) and not self.game_over,
        }

    def start_background(self):
        """Start running the match in a background thread."""
        set_current_match(self)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def submit_human_move(self, uci_str: str) -> bool:
        """Submit a move played by a human in the UI."""
        try:
            move = chess.Move.from_uci(uci_str)
            if move in self.board.legal_moves:
                self._pending_human_move = move
                self._human_event.set()
                return True
        except Exception:
            pass
        return False

    def _run_loop(self):
        """Internal execution loop."""
        # Find if either player is Stockfish
        sf_player = None
        if isinstance(self.white_player, StockfishPlayer):
            sf_player = self.white_player
        elif isinstance(self.black_player, StockfishPlayer):
            sf_player = self.black_player

        if sf_player:
            try:
                self._engine = chess.engine.SimpleEngine.popen_uci(sf_player.stockfish_path)
                config = {k: v for k, v in sf_player.bot_options.items() if k != "Ponder"}
                self._engine.configure(config)
            except Exception as e:
                self.game_over = True
                self.termination_reason = f"Stockfish startup error: {e}"
                return

        pgn_game = chess.pgn.Game()
        pgn_game.headers["Event"] = f"ChessFly {'1+0 Bullet' if self.timed else 'Analysis'} Match"
        pgn_game.headers["Site"] = "Local Machine"
        pgn_game.headers["Date"] = datetime.now(timezone.utc).strftime("%Y.%m.%d")
        pgn_game.headers["White"] = self.white_player.name
        pgn_game.headers["Black"] = self.black_player.name
        pgn_game.headers["TimeControl"] = f"{int(self.base_time)}+0" if self.timed else "-"
        current_node = pgn_game

        try:
            while not self.game_over:
                side_to_move = self.board.turn
                player = self.white_player if side_to_move == chess.WHITE else self.black_player
                fen_before = self.board.fen()

                if not list(self.board.legal_moves):
                    self.game_over = True
                    if self.board.is_checkmate():
                        self.winner = "black" if side_to_move == chess.WHITE else "white"
                        self.result = "0-1" if side_to_move == chess.WHITE else "1-0"
                        self.termination_reason = "Checkmate"
                        self.fly_state = "upset"
                    else:
                        self.winner = "draw"
                        self.result = "1/2-1/2"
                        self.termination_reason = "Stalemate"
                        self.fly_state = "idle"
                    break

                # Fly mascot state indication
                if "Fly" in player.name:
                    self.fly_state = "thinking"
                else:
                    self.fly_state = "idle"

                t0 = time.perf_counter()
                move = None

                if player.is_human:
                    # Wait for human input event
                    self._human_event.clear()
                    self._pending_human_move = None

                    # If timed, loop with timeout to decrement clock
                    if self.timed:
                        while not self._human_event.is_set() and not self.game_over:
                            self._human_event.wait(timeout=0.05)
                            now = time.perf_counter()
                            elapsed = now - t0
                            t0 = now
                            if side_to_move == chess.WHITE:
                                self.white_clock -= elapsed
                                if self.white_clock <= 0:
                                    break
                            else:
                                self.black_clock -= elapsed
                                if self.black_clock <= 0:
                                    break

                        if self.game_over:
                            break

                        if (side_to_move == chess.WHITE and self.white_clock <= 0) or (side_to_move == chess.BLACK and self.black_clock <= 0):
                            self.game_over = True
                            opp_color = not side_to_move
                            if self.board.has_insufficient_material(opp_color):
                                self.winner = "draw"
                                self.result = "1/2-1/2"
                                self.termination_reason = f"{'White' if side_to_move == chess.WHITE else 'Black'} flagged (draw vs insufficient material)"
                            else:
                                self.winner = "black" if side_to_move == chess.WHITE else "white"
                                self.result = "0-1" if side_to_move == chess.WHITE else "1-0"
                                self.termination_reason = f"{'White' if side_to_move == chess.WHITE else 'Black'} lost on time (flagged)"
                            self.fly_state = "upset"
                            break

                        move = self._pending_human_move
                    else:
                        while not self._human_event.is_set() and not self.game_over:
                            self._human_event.wait(timeout=0.1)
                        if self.game_over:
                            break
                        move = self._pending_human_move
                    elapsed = 0.0  # already accounted for
                else:
                    my_c = self.white_clock if side_to_move == chess.WHITE else self.black_clock
                    opp_c = self.black_clock if side_to_move == chess.WHITE else self.white_clock

                    try:
                        if isinstance(player, StockfishPlayer) and self._engine:
                            move = player.get_move(self.board, my_clock=my_c, opp_clock=opp_c, engine_instance=self._engine)
                        else:
                            move = player.get_move(self.board, my_clock=my_c, opp_clock=opp_c)
                    except Exception as exc:
                        import logging
                        logging.error(
                            "Player %s (%s) threw an exception on FEN '%s' with %d legal moves: %s",
                            player.name,
                            "White" if side_to_move == chess.WHITE else "Black",
                            fen_before,
                            len(list(self.board.legal_moves)),
                            exc,
                        )
                        self.game_over = True
                        self.winner = "black" if side_to_move == chess.WHITE else "white"
                        self.result = "0-1" if side_to_move == chess.WHITE else "1-0"
                        self.termination_reason = f"Player error ({player.name}): {exc}"
                        self.fly_state = "upset"
                        break

                    elapsed = time.perf_counter() - t0

                # Charge elapsed time strictly to active player
                if self.timed and not player.is_human:
                    if side_to_move == chess.WHITE:
                        self.white_clock -= elapsed
                        rem = self.white_clock
                    else:
                        self.black_clock -= elapsed
                        rem = self.black_clock

                    if rem <= 0:
                        self.game_over = True
                        opp_color = not side_to_move
                        if self.board.has_insufficient_material(opp_color):
                            self.winner = "draw"
                            self.result = "1/2-1/2"
                            self.termination_reason = f"{'White' if side_to_move == chess.WHITE else 'Black'} flagged (draw vs insufficient material)"
                        else:
                            self.winner = "black" if side_to_move == chess.WHITE else "white"
                            self.result = "0-1" if side_to_move == chess.WHITE else "1-0"
                            self.termination_reason = f"{'White' if side_to_move == chess.WHITE else 'Black'} lost on time (flagged)"
                        self.fly_state = "upset"
                        break

                if move is None:
                    if self.game_over:
                        break
                    import logging
                    logging.error(
                        "Player %s (%s) returned None move on FEN '%s' with %d legal moves",
                        player.name,
                        "White" if side_to_move == chess.WHITE else "Black",
                        fen_before,
                        len(list(self.board.legal_moves)),
                    )
                    self.game_over = True
                    self.winner = "black" if side_to_move == chess.WHITE else "white"
                    self.result = "0-1" if side_to_move == chess.WHITE else "1-0"
                    self.termination_reason = f"Player error: {player.name} returned None on legal position"
                    self.fly_state = "upset"
                    break

                if move not in self.board.legal_moves:
                    import logging
                    logging.error(
                        "Player %s (%s) produced illegal move '%s' on FEN '%s'",
                        player.name,
                        "White" if side_to_move == chess.WHITE else "Black",
                        move.uci() if hasattr(move, 'uci') else str(move),
                        fen_before,
                    )
                    self.game_over = True
                    self.winner = "black" if side_to_move == chess.WHITE else "white"
                    self.result = "0-1" if side_to_move == chess.WHITE else "1-0"
                    self.termination_reason = f"Illegal move: {move.uci() if hasattr(move, 'uci') else move}"
                    self.fly_state = "upset"
                    break

                san_move = self.board.san(move)
                self.board.push(move)
                self.last_move = move.uci()

                if "Fly" in player.name:
                    self.fly_state = "happy"

                # Append to PGN
                current_node = current_node.add_variation(move)
                rem_c = self.white_clock if side_to_move == chess.WHITE else self.black_clock
                if rem_c is not None:
                    current_node.comment = f"[%clk {format_clock(rem_c)}] [%emt {elapsed:.3f}]"

                record = {
                    "ply": len(self.moves_log) + 1,
                    "color": "white" if side_to_move == chess.WHITE else "black",
                    "player": player.name,
                    "uci": move.uci(),
                    "san": san_move,
                    "elapsed_ms": round(elapsed * 1000, 1),
                    "white_clock": round(self.white_clock, 2) if self.white_clock is not None else None,
                    "black_clock": round(self.black_clock, 2) if self.black_clock is not None else None,
                    "fen_before": fen_before,
                    "fen_after": self.board.fen(),
                }
                self.moves_log.append(record)

                # Check game termination conditions
                if self.board.is_checkmate():
                    self.game_over = True
                    self.winner = "white" if side_to_move == chess.WHITE else "black"
                    self.result = "1-0" if side_to_move == chess.WHITE else "0-1"
                    self.termination_reason = "Checkmate"
                    self.fly_state = "happy" if "Fly" in player.name else "upset"
                elif self.board.is_stalemate():
                    self.game_over = True
                    self.winner = "draw"
                    self.result = "1/2-1/2"
                    self.termination_reason = "Stalemate"
                elif self.board.is_insufficient_material():
                    self.game_over = True
                    self.winner = "draw"
                    self.result = "1/2-1/2"
                    self.termination_reason = "Insufficient material"
                elif self.board.can_claim_threefold_repetition():
                    self.game_over = True
                    self.winner = "draw"
                    self.result = "1/2-1/2"
                    self.termination_reason = "Threefold repetition claim"
                elif self.board.can_claim_fifty_moves():
                    self.game_over = True
                    self.winner = "draw"
                    self.result = "1/2-1/2"
                    self.termination_reason = "Fifty-move rule claim"

        finally:
            if self._engine:
                try:
                    self._engine.quit()
                except Exception:
                    pass

            pgn_game.headers["Result"] = self.result
            pgn_game.headers["Termination"] = self.termination_reason
            pgn_str = str(pgn_game)

            # Persist game to disk
            final_record = {
                "id": self.game_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": "bullet" if self.timed else "analysis",
                "white_player": self.white_player.name,
                "black_player": self.black_player.name,
                "initial_fen": chess.STARTING_FEN,
                "result": self.result,
                "winner": self.winner,
                "termination_reason": self.termination_reason,
                "ply_count": len(self.moves_log),
                "white_clock": self.white_clock,
                "black_clock": self.black_clock,
                "moves": self.moves_log,
                "pgn": pgn_str,
            }
            save_game(final_record)


class BulletMatch(ChessMatch):
    """Backward-compatible subclass for existing CLI / tests."""
    def __init__(self,
                 fly_model: FlyPolicy,
                 stockfish_path: Optional[str] = None,
                 fly_color: chess.Color = chess.WHITE,
                 base_time: float = 60.0,
                 increment: float = 0.0,
                 bot_difficulty: str = "weak",
                 bot_options: Optional[dict] = None,
                 sample_moves: bool = False):
        self.fly_model = fly_model
        self.stockfish_path = stockfish_path or find_stockfish()
        self.fly_color = fly_color
        self.bot_difficulty = bot_difficulty
        self.sample_moves = sample_moves

        # Wrap in player instances
        class CustomFlyPlayer(BasePlayer):
            name = "ChessFly-Policy"
            def __init__(self, m, sample):
                self.m = m
                self.sample = sample
            def get_move(self, board, my_clock=None, opp_clock=None):
                return self.m.choose(board, sample=self.sample, my_clock=my_clock, opp_clock=opp_clock)

        fly_p = CustomFlyPlayer(fly_model, sample_moves)
        sf_p = StockfishPlayer(stockfish_path, difficulty=bot_difficulty)
        if bot_options:
            sf_p.bot_options = bot_options

        white_p = fly_p if fly_color == chess.WHITE else sf_p
        black_p = sf_p if fly_color == chess.WHITE else fly_p

        super().__init__(white_player=white_p, black_player=black_p,
                         timed=True, base_time=base_time, increment=increment)

    def run(self, on_move: Optional[Callable[[dict], None]] = None) -> dict:
        self._run_loop()
        final_record = {
            "id": self.game_id,
            "result": self.result,
            "winner": "fly" if (self.winner == "white" and self.fly_color == chess.WHITE) or (self.winner == "black" and self.fly_color == chess.BLACK) else ("stockfish" if self.winner in ("white", "black") else "draw"),
            "termination_reason": self.termination_reason,
            "fly_color": "white" if self.fly_color == chess.WHITE else "black",
            "ply_count": len(self.moves_log),
            "white_clock": round(self.white_clock, 2) if self.white_clock is not None else 0.0,
            "black_clock": round(self.black_clock, 2) if self.black_clock is not None else 0.0,
            "moves": self.moves_log,
            "pgn": "",
        }
        # Re-export PGN
        pgn_game = chess.pgn.Game()
        pgn_game.headers["Event"] = "ChessFly Bullet 1+0 Match"
        pgn_game.headers["White"] = self.white_player.name
        pgn_game.headers["Black"] = self.black_player.name
        pgn_game.headers["Result"] = self.result
        pgn_game.headers["Termination"] = self.termination_reason
        node = pgn_game
        for m in self.moves_log:
            node = node.add_variation(chess.Move.from_uci(m["uci"]))
        final_record["pgn"] = str(pgn_game)
        return final_record
