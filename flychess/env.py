"""Short chess episodes with legal actions, bounded inspection, and fixed opponents."""
import random
import chess
import chess.engine

from .reward import features, potential, result_value


class ChessEnv:
    def __init__(self, fen=chess.STARTING_FEN, opponent="random", stockfish=None,
                 seed=42, max_agent_moves=8, inspect_budget=4, opponent_nodes=3000):
        self.board = chess.Board(fen)
        self.color = self.board.turn
        self.opponent = opponent
        self.rng = random.Random(seed)
        self.max_agent_moves = max_agent_moves
        self.inspect_budget = inspect_budget
        self.opponent_nodes = opponent_nodes
        self.stockfish = stockfish
        self.engine = None
        if opponent == "stockfish":
            if not stockfish:
                raise ValueError("--stockfish path required for Stockfish opponent")
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish)
            self.engine.configure({"Threads": 1, "Hash": 16})
        elif opponent != "random":
            raise ValueError("Unknown opponent")
        self.moves_played = 0
        self.inspections = 0
        self.invalid = 0
        self.done = False
        self.history = []

    def close(self):
        if self.engine:
            self.engine.quit()
            self.engine = None

    def observe(self):
        return {"fen": self.board.fen(), "ascii_board": str(self.board),
                "legal_moves": [m.uci() for m in self.board.legal_moves] if not self.done else [],
                "agent_moves_left": self.max_agent_moves - self.moves_played,
                "inspections_left": self.inspect_budget - self.inspections,
                "history": list(self.history), "done": self.done}

    def inspect_line(self, uci_moves):
        if self.done or self.inspections >= self.inspect_budget:
            raise ValueError("Episode finished or inspection budget exhausted")
        if len(uci_moves) > 6:
            raise ValueError("At most six plies per inspection")
        copy = self.board.copy(stack=True)
        for uci in uci_moves:
            move = chess.Move.from_uci(uci)
            if move not in copy.legal_moves:
                raise ValueError(f"Illegal proposed continuation: {uci}")
            copy.push(move)
        self.inspections += 1
        return {"fen": copy.fen(), "ascii_board": str(copy),
                "legal_moves": [m.uci() for m in copy.legal_moves],
                "inspections_left": self.inspect_budget - self.inspections}

    def play(self, uci: str):
        if self.done:
            raise ValueError("Episode already finished")
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            move = None
        if move not in self.board.legal_moves:
            self.invalid += 1
            if self.invalid >= 2:
                self.done = True
            return {"error": "illegal move", "done": self.done, "training_reward": 0.0,
                    "observation": self.observe()}
        before = potential(self.board, self.color)
        self.board.push(move)
        self.history.append(move.uci())
        self.moves_played += 1
        response = None
        if not self.board.is_game_over(claim_draw=True):
            if self.engine:
                response = self.engine.play(self.board, chess.engine.Limit(nodes=self.opponent_nodes)).move
            else:
                response = self.rng.choice(list(self.board.legal_moves))
            self.board.push(response)
            self.history.append(response.uci())
        actual_result = result_value(self.board, self.color)
        self.done = actual_result is not None or self.moves_played >= self.max_agent_moves
        after = 0.0 if actual_result is not None else potential(self.board, self.color)
        shaped = 0.20 * (0.99*after - before)
        reward = shaped + (actual_result if actual_result is not None else 0.0)
        return {"opponent_move": response.uci() if response else None,
                "training_reward": round(reward, 6), "actual_result": actual_result,
                "diagnostics": features(self.board, self.color), "observation": self.observe()}
