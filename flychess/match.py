"""Real 1+0 bullet match runner against Stockfish with precise clock management."""
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import chess
import chess.engine
import chess.pgn
import numpy as np

from .model import FlyPolicy


def format_clock(seconds: float) -> str:
    """Format seconds into standard PGN clock format H:MM:SS or M:SS."""
    sec = max(0.0, seconds)
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"


class BulletMatch:
    def __init__(self,
                 fly_model: FlyPolicy,
                 stockfish_path: str = "/opt/homebrew/bin/stockfish",
                 fly_color: chess.Color = chess.WHITE,
                 base_time: float = 60.0,
                 increment: float = 0.0,
                 bot_difficulty: str = "weak",
                 bot_options: Optional[dict] = None,
                 sample_moves: bool = False):
        self.fly_model = fly_model
        self.stockfish_path = stockfish_path
        self.fly_color = fly_color
        self.base_time = base_time
        self.increment = increment
        self.sample_moves = sample_moves
        self.bot_difficulty = bot_difficulty

        # Standard preset bot options
        if bot_options is not None:
            self.bot_options = bot_options
        elif bot_difficulty == "weak":
            self.bot_options = {"Skill Level": 0, "UCI_LimitStrength": True, "UCI_Elo": 1350, "Threads": 1, "Hash": 16}
        elif bot_difficulty == "medium":
            self.bot_options = {"Skill Level": 5, "UCI_LimitStrength": True, "UCI_Elo": 1600, "Threads": 1, "Hash": 16}
        else: # "strong"
            self.bot_options = {"Skill Level": 20, "UCI_LimitStrength": False, "Threads": 1, "Hash": 16}

        self.board = chess.Board()
        self.white_clock = base_time
        self.black_clock = base_time
        self.moves_log = []
        self.game_over = False
        self.result = "*"
        self.termination_reason = "In progress"
        self.winner = None

    def run(self, on_move: Optional[Callable[[dict], None]] = None) -> dict:
        """Run the full 1+0 bullet match. Engine and model are preloaded before clocks start."""
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        try:
            # Configure engine (Ponder is disabled by default in python-chess unless ponder=True is passed)
            config = {k: v for k, v in self.bot_options.items() if k != "Ponder"}
            engine.configure(config)

            engine_id = engine.id.get("name", "Stockfish")

            # Initialize PGN game structure
            pgn_game = chess.pgn.Game()
            pgn_game.headers["Event"] = "ChessFly Bullet 1+0 Match"
            pgn_game.headers["Site"] = "Local Machine"
            pgn_game.headers["Date"] = datetime.now(timezone.utc).strftime("%Y.%m.%d")
            pgn_game.headers["Round"] = "1"
            pgn_game.headers["White"] = "ChessFly-Policy" if self.fly_color == chess.WHITE else f"Stockfish ({self.bot_difficulty})"
            pgn_game.headers["Black"] = f"Stockfish ({self.bot_difficulty})" if self.fly_color == chess.WHITE else "ChessFly-Policy"
            pgn_game.headers["TimeControl"] = f"{int(self.base_time)}+{int(self.increment)}"

            current_node = pgn_game

            while not self.game_over:
                side_to_move = self.board.turn
                is_fly_turn = (side_to_move == self.fly_color)
                fen_before = self.board.fen()
                legal_moves = list(self.board.legal_moves)

                if not legal_moves:
                    # Check terminal state before requesting move
                    if self.board.is_checkmate():
                        self.game_over = True
                        winner_color = not side_to_move
                        self.winner = "fly" if winner_color == self.fly_color else "stockfish"
                        self.result = "1-0" if winner_color == chess.WHITE else "0-1"
                        self.termination_reason = "Checkmate"
                    else:
                        self.game_over = True
                        self.winner = "draw"
                        self.result = "1/2-1/2"
                        self.termination_reason = "Stalemate"
                    break

                # Start clock immediately before move request
                t0 = time.perf_counter()

                move = None
                log_prob = None
                try:
                    if is_fly_turn:
                        my_clock = self.white_clock if self.fly_color == chess.WHITE else self.black_clock
                        opp_clock = self.black_clock if self.fly_color == chess.WHITE else self.white_clock
                        legal, logits, probs, _ = self.fly_model.forward(self.board, my_clock=my_clock, opp_clock=opp_clock)
                        if self.sample_moves:
                            idx = self.fly_model.rng.choice(len(legal), p=probs)
                        else:
                            idx = int(probs.argmax())
                        move = legal[idx]
                        log_prob = float(np.log(max(probs[idx], 1e-12))) if "np" in globals() else 0.0
                    else:
                        limit = chess.engine.Limit(
                            white_clock=self.white_clock,
                            black_clock=self.black_clock,
                            white_inc=self.increment,
                            black_inc=self.increment,
                        )
                        engine_res = engine.play(self.board, limit)
                        move = engine_res.move
                except Exception as exc:
                    self.game_over = True
                    winner_color = not side_to_move
                    self.winner = "fly" if winner_color == self.fly_color else "stockfish"
                    self.result = "1-0" if winner_color == chess.WHITE else "0-1"
                    self.termination_reason = f"Engine/Model exception: {exc}"
                    break

                elapsed = time.perf_counter() - t0

                # Charge elapsed time strictly to the current player
                if side_to_move == chess.WHITE:
                    self.white_clock -= elapsed
                    self.white_clock += self.increment
                    remaining = self.white_clock
                else:
                    self.black_clock -= elapsed
                    self.black_clock += self.increment
                    remaining = self.black_clock

                # Check flagging (time expiration)
                if remaining <= 0:
                    self.game_over = True
                    opp_color = not side_to_move
                    # FIDE rule on timeout: if opponent has insufficient mating material, draw.
                    if self.board.has_insufficient_material(opp_color):
                        self.winner = "draw"
                        self.result = "1/2-1/2"
                        self.termination_reason = f"{'White' if side_to_move == chess.WHITE else 'Black'} flagged, draw due to insufficient material"
                    else:
                        self.winner = "stockfish" if is_fly_turn else "fly"
                        self.result = "0-1" if side_to_move == chess.WHITE else "1-0"
                        self.termination_reason = f"{'White' if side_to_move == chess.WHITE else 'Black'} lost on time (flagged)"

                    # Log the flagged move attempt if any
                    record = {
                        "ply": len(self.moves_log) + 1,
                        "color": "white" if side_to_move == chess.WHITE else "black",
                        "player": "fly" if is_fly_turn else "stockfish",
                        "uci": move.uci() if move else "None",
                        "san": self.board.san(move) if move and move in self.board.legal_moves else "??",
                        "elapsed_ms": round(elapsed * 1000, 1),
                        "white_clock": round(self.white_clock, 2),
                        "black_clock": round(self.black_clock, 2),
                        "fen_before": fen_before,
                        "flagged": True,
                    }
                    self.moves_log.append(record)
                    if on_move:
                        on_move(record)
                    break

                # Verify move legality
                if move is None or move not in self.board.legal_moves:
                    self.game_over = True
                    self.winner = "stockfish" if is_fly_turn else "fly"
                    self.result = "0-1" if side_to_move == chess.WHITE else "1-0"
                    self.termination_reason = f"Illegal move attempted by {'fly' if is_fly_turn else 'stockfish'}: {move}"
                    break

                san_move = self.board.san(move)
                self.board.push(move)

                # Add to PGN
                current_node = current_node.add_variation(move)
                current_node.comment = f"[%clk {format_clock(remaining)}] [%emt {elapsed:.3f}]"

                record = {
                    "ply": len(self.moves_log) + 1,
                    "color": "white" if side_to_move == chess.WHITE else "black",
                    "player": "fly" if is_fly_turn else "stockfish",
                    "uci": move.uci(),
                    "san": san_move,
                    "elapsed_ms": round(elapsed * 1000, 1),
                    "white_clock": round(self.white_clock, 2),
                    "black_clock": round(self.black_clock, 2),
                    "fen_before": fen_before,
                    "fen_after": self.board.fen(),
                }
                self.moves_log.append(record)
                if on_move:
                    on_move(record)

                # Check game termination conditions
                if self.board.is_checkmate():
                    self.game_over = True
                    self.winner = "fly" if is_fly_turn else "stockfish"
                    self.result = "1-0" if side_to_move == chess.WHITE else "0-1"
                    self.termination_reason = "Checkmate"
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

            pgn_game.headers["Result"] = self.result
            pgn_game.headers["Termination"] = self.termination_reason

            return {
                "result": self.result,
                "winner": self.winner,
                "termination_reason": self.termination_reason,
                "fly_color": "white" if self.fly_color == chess.WHITE else "black",
                "ply_count": len(self.moves_log),
                "white_clock": round(self.white_clock, 2),
                "black_clock": round(self.black_clock, 2),
                "moves": self.moves_log,
                "pgn": str(pgn_game),
                "engine": engine_id,
                "bot_options": self.bot_options,
            }
        finally:
            engine.quit()
