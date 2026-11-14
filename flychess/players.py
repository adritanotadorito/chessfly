"""Player abstractions for Human, FlyPolicy, Stockfish, and Gemini."""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chess
import chess.engine
import numpy as np

from .model import FlyPolicy


def load_env_file():
    """Load key-value pairs from .env file into os.environ if present, without overwriting existing vars."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("\"'")
                if k and k not in os.environ:
                    os.environ[k] = v


class BasePlayer:
    name: str = "Player"
    is_human: bool = False

    def get_move(self,
                 board: chess.Board,
                 my_clock: Optional[float] = None,
                 opp_clock: Optional[float] = None) -> chess.Move:
        raise NotImplementedError


class HumanPlayer(BasePlayer):
    name: str = "Human"
    is_human: bool = True

    def get_move(self, board: chess.Board, my_clock=None, opp_clock=None) -> chess.Move:
        raise RuntimeError("HumanPlayer move must be submitted directly through the UI")


class FlyPolicyPlayer(BasePlayer):
    def __init__(self,
                 model_path: str,
                 graph_path: str,
                 name: str = "Fly Policy",
                 shuffled: bool = False,
                 sample_moves: bool = False,
                 seed: int = 7):
        self.name = name
        self.model = FlyPolicy(graph_path, seed=seed, shuffled=shuffled)
        self.model.load_weights(model_path)
        self.sample_moves = sample_moves

    def get_move(self,
                 board: chess.Board,
                 my_clock: Optional[float] = None,
                 opp_clock: Optional[float] = None) -> chess.Move:
        return self.model.choose(board, sample=self.sample_moves, my_clock=my_clock, opp_clock=opp_clock)


class StockfishPlayer(BasePlayer):
    def __init__(self,
                 stockfish_path: str = "/opt/homebrew/bin/stockfish",
                 difficulty: str = "weak",
                 name: Optional[str] = None):
        self.stockfish_path = stockfish_path
        self.difficulty = difficulty
        self.name = name or f"Stockfish ({difficulty})"

        if difficulty == "weak":
            self.bot_options = {"Skill Level": 0, "UCI_LimitStrength": True, "UCI_Elo": 1350, "Threads": 1, "Hash": 16}
        elif difficulty == "medium":
            self.bot_options = {"Skill Level": 5, "UCI_LimitStrength": True, "UCI_Elo": 1600, "Threads": 1, "Hash": 16}
        else: # "strong"
            self.bot_options = {"Skill Level": 20, "UCI_LimitStrength": False, "Threads": 1, "Hash": 16}

    def get_move(self,
                 board: chess.Board,
                 my_clock: Optional[float] = None,
                 opp_clock: Optional[float] = None,
                 engine_instance: Optional[chess.engine.SimpleEngine] = None) -> chess.Move:
        close_after = False
        if engine_instance is None:
            engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            config = {k: v for k, v in self.bot_options.items() if k != "Ponder"}
            engine.configure(config)
            close_after = True
        else:
            engine = engine_instance

        try:
            if my_clock is not None and opp_clock is not None:
                # Timed bullet limit
                is_white = (board.turn == chess.WHITE)
                w_c = my_clock if is_white else opp_clock
                b_c = opp_clock if is_white else my_clock
                limit = chess.engine.Limit(white_clock=w_c, black_clock=b_c, white_inc=0, black_inc=0)
            else:
                # Untimed analysis limit: fixed node budget
                limit = chess.engine.Limit(nodes=10000)

            res = engine.play(board, limit)
            return res.move
        finally:
            if close_after:
                engine.quit()


def get_gemini_status() -> Dict[str, Any]:
    """Check Gemini configuration status cleanly without ever exposing or logging key values."""
    load_env_file()
    sdk_installed = False
    try:
        import google.genai  # noqa: F401
        sdk_installed = True
    except ImportError:
        pass

    raw_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    raw_key = raw_key.strip()
    key_present = bool(raw_key and not raw_key.lower().startswith("your_"))

    if not sdk_installed:
        return {
            "sdk_installed": False,
            "key_present": key_present,
            "is_configured": False,
            "status_text": "SDK Missing",
            "message": "The google-genai package is not installed.",
        }

    if not key_present:
        return {
            "sdk_installed": True,
            "key_present": False,
            "is_configured": False,
            "status_text": "Not configured",
            "message": "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set.",
        }

    return {
        "sdk_installed": True,
        "key_present": True,
        "is_configured": True,
        "status_text": "Configured",
        "message": "Gemini API key is detected.",
    }


class GeminiPlayer(BasePlayer):
    def __init__(self, model_name: str = "gemini-2.5-flash", name: Optional[str] = None):
        load_env_file()
        self.model_name = model_name
        self.name = name or f"Gemini ({model_name})"
        status = get_gemini_status()
        self.is_ready = status["is_configured"]
        self.api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "") if self.is_ready else ""

    def is_configured(self) -> bool:
        return get_gemini_status()["is_configured"]

    def get_move(self,
                 board: chess.Board,
                 my_clock: Optional[float] = None,
                 opp_clock: Optional[float] = None) -> chess.Move:
        if not self.is_configured():
            raise ValueError(
                "GEMINI_API_KEY is not configured. Please set the GEMINI_API_KEY environment variable "
                "or place it in a local .env file."
            )

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("The 'google-genai' package is required for GeminiPlayer. Install via pip install google-genai.")

        client = genai.Client(api_key=self.api_key)

        legal_moves_uci = [m.uci() for m in board.legal_moves]
        history_uci = [m.uci() for m in board.move_stack]
        side = "White" if board.turn == chess.WHITE else "Black"

        prompt = f"""You are playing chess as {side}.
Current FEN position: {board.fen()}
Move history (UCI): {history_uci[-10:] if history_uci else 'Game start'}
Remaining clocks: You have {round(my_clock, 1) if my_clock is not None else 'Untimed'}s, opponent has {round(opp_clock, 1) if opp_clock is not None else 'Untimed'}s.

Legal moves available: {legal_moves_uci}

Select the single best move from the legal moves list.
Respond ONLY with a JSON object in this exact format:
{{"move_uci": "<chosen move from legal moves list>"}}
"""

        # Attempt up to 2 times
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                )
                text = response.text.strip()
                parsed = json.loads(text)
                chosen_uci = parsed.get("move_uci", "").strip()
                move = chess.Move.from_uci(chosen_uci)
                if move in board.legal_moves:
                    return move
                else:
                    prompt += f"\nNote: Move '{chosen_uci}' was illegal. Choose STRICTLY from {legal_moves_uci}."
            except Exception as e:
                if attempt == 1:
                    raise RuntimeError(f"Gemini move generation failed: {e}")

        # If both attempts failed, forfeit or raise
        raise ValueError(f"Gemini failed to produce a valid legal move from: {legal_moves_uci}")
