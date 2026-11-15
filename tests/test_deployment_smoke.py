"""Smoke test verifying clean Streamlit Community Cloud deployment prerequisites:
1. Clean startup and Stockfish dynamic discovery (no hardcoded Mac paths)
2. Gemini status unconfigured without key leak
3. Human move against FlyPolicy
4. Fly vs Stockfish one complete turn
5. Game replay reconstruction
"""
import os
import time
import unittest
from pathlib import Path

import chess
from flychess.graph import load_graph
from flychess.match import ChessMatch
from flychess.model import FlyPolicy
from flychess.players import (FlyPolicyPlayer, HumanPlayer, StockfishPlayer,
                             find_stockfish, get_gemini_status,
                             get_stockfish_info)
from flychess.games import save_game, load_game, list_games


class DeploymentSmokeTests(unittest.TestCase):
    def test_01_stockfish_discovery_and_info(self):
        """Verify dynamic Stockfish discovery and non-hardcoded version reporting."""
        sf_path = find_stockfish()
        self.assertIsNotNone(sf_path, "Stockfish must be discovered on Mac/Linux")
        self.assertTrue(os.path.exists(sf_path), f"Discovered path must exist: {sf_path}")
        self.assertTrue(os.access(sf_path, os.X_OK), f"Discovered path must be executable: {sf_path}")

        info = get_stockfish_info(sf_path)
        self.assertTrue(info["available"], f"Stockfish info must be available: {info}")
        self.assertIn("name", info)
        self.assertIn("version", info)
        print(f"\n[Smoke Check] Discovered Engine: {info['name']} at {info['path']}")

    def test_02_gemini_status_clean_when_unconfigured(self):
        """App must start with no Gemini key and report 'Not configured'."""
        orig_key = os.environ.pop("GEMINI_API_KEY", None)
        orig_gkey = os.environ.pop("GOOGLE_API_KEY", None)
        try:
            status = get_gemini_status()
            self.assertFalse(status["is_configured"])
            self.assertEqual(status["status_text"], "Not configured")
            # Verify no secret printed or leaked
            for k, v in status.items():
                self.assertNotIn("AIza", str(v))
            print("[Smoke Check] Gemini status: Not configured (clean, no key leaks)")
        finally:
            if orig_key:
                os.environ["GEMINI_API_KEY"] = orig_key
            if orig_gkey:
                os.environ["GOOGLE_API_KEY"] = orig_gkey

    def test_03_graph_and_fly_checkpoint_loadable(self):
        """Required graph and chosen checkpoints must be loadable from repo."""
        root = Path(__file__).resolve().parent.parent
        graph_path = root / "data" / "flywire-subgraph.npz"
        self.assertTrue(graph_path.is_file(), "data/flywire-subgraph.npz must exist")

        ckpt_path = root / "outputs" / "fly-rl-pilot.npz"
        if not ckpt_path.is_file():
            ckpt_path = root / "outputs" / "fly-200.npz"
        self.assertTrue(ckpt_path.is_file(), "Primary fly checkpoint must exist")

        player = FlyPolicyPlayer(str(ckpt_path), str(graph_path))
        board = chess.Board()
        move = player.get_move(board)
        self.assertIn(move, board.legal_moves, "Fly policy must produce legal move")
        print(f"[Smoke Check] FlyPolicy loaded successfully, chose: {move.uci()}")

    def test_04_human_move_against_fly(self):
        """Test human submitting a move against the fly and receiving fly response."""
        root = Path(__file__).resolve().parent.parent
        graph_path = str(root / "data" / "flywire-subgraph.npz")
        ckpt_path = str(root / "outputs" / "fly-rl-pilot.npz" if (root / "outputs" / "fly-rl-pilot.npz").is_file() else root / "outputs" / "fly-200.npz")

        human = HumanPlayer()
        fly = FlyPolicyPlayer(ckpt_path, graph_path, name="TestFly")

        match = ChessMatch(white_player=human, black_player=fly, timed=False)
        match.start_background()
        time.sleep(0.05)

        state = match.get_state()
        self.assertTrue(state["is_human_turn"])

        # Human plays e2e4
        success = match.submit_human_move("e2e4")
        self.assertTrue(success, "Human move e2e4 must succeed")

        # Wait for fly reply in background loop
        for _ in range(50):
            time.sleep(0.05)
            state = match.get_state()
            if len(state["moves"]) >= 2:
                break

        match.game_over = True
        self.assertGreaterEqual(len(state["moves"]), 2, "Fly must reply to human move")
        fly_move = state["moves"][1]
        print(f"[Smoke Check] Human-vs-Fly turn complete: 1. e2e4 {fly_move['san']}")

    def test_05_fly_versus_stockfish_turn(self):
        """Test one complete turn between Fly Policy and Stockfish."""
        root = Path(__file__).resolve().parent.parent
        graph_path = str(root / "data" / "flywire-subgraph.npz")
        ckpt_path = str(root / "outputs" / "fly-rl-pilot.npz" if (root / "outputs" / "fly-rl-pilot.npz").is_file() else root / "outputs" / "fly-200.npz")

        fly = FlyPolicyPlayer(ckpt_path, graph_path, name="TestFly")
        sf = StockfishPlayer(difficulty="weak", name="TestSF")

        match = ChessMatch(white_player=fly, black_player=sf, timed=False)
        match.start_background()

        # Wait for 1 complete turn (2 plies: Fly White + Stockfish Black)
        for _ in range(100):
            time.sleep(0.05)
            state = match.get_state()
            if len(state["moves"]) >= 2:
                break

        match.game_over = True
        self.assertGreaterEqual(len(state["moves"]), 2, "Both players must complete their move in turn 1")
        m1 = state["moves"][0]
        m2 = state["moves"][1]
        print(f"[Smoke Check] Fly-vs-Stockfish turn complete: 1. {m1['san']} {m2['san']}")

    def test_06_game_persistence_and_replay_reconstruction(self):
        """Test saving a game and reconstructing board positions across all plies."""
        root = Path(__file__).resolve().parent.parent
        graph_path = str(root / "data" / "flywire-subgraph.npz")
        ckpt_path = str(root / "outputs" / "fly-200.npz")

        fly = FlyPolicyPlayer(ckpt_path, graph_path)
        sf = StockfishPlayer(difficulty="weak")
        match = ChessMatch(white_player=fly, black_player=sf, timed=False)
        match.start_background()

        for _ in range(100):
            time.sleep(0.05)
            if len(match.get_state()["moves"]) >= 4:
                break

        match.game_over = True
        state = match.get_state()
        self.assertGreaterEqual(len(state["moves"]), 4)
        saved_id = save_game(state)
        self.assertEqual(saved_id, match.game_id)

        # Load back
        loaded = load_game(match.game_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded["moves"]), len(state["moves"]))

        # Reconstruct board at each ply
        r_board = chess.Board(loaded.get("initial_fen", chess.STARTING_FEN))
        for m in loaded["moves"]:
            r_board.push_uci(m["uci"])

        self.assertEqual(r_board.fen(), state["fen"])
        print(f"[Smoke Check] Replay reconstruction passed for game {match.game_id}")


if __name__ == "__main__":
    unittest.main()
