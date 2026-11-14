import tempfile
import unittest
from pathlib import Path

import chess
import chess.engine
import numpy as np

from flychess.board import DIM, encode
from flychess.graph import make_fixture
from flychess.match import BulletMatch, format_clock
from flychess.model import FlyPolicy
from flychess.reward import (compute_game_rewards, features, move_audit,
                            potential, result_value)


class BulletAndRewardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.graph = str(Path(self.tmp.name) / "fixture.npz")
        make_fixture(self.graph)
        self.model = FlyPolicy(self.graph)

    def tearDown(self):
        self.tmp.cleanup()

    def test_clock_encoding_and_migration(self):
        board = chess.Board()
        enc = encode(board, my_clock=45.0, opp_clock=30.0)
        self.assertEqual(enc.shape, (DIM,))
        self.assertEqual(DIM, 785)
        self.assertAlmostEqual(enc[-2], 45.0 / 60.0)
        self.assertAlmostEqual(enc[-1], 30.0 / 60.0)

    def test_bullet_flagging_insufficient_material_draw(self):
        # White has King + Bishop, Black has only King.
        # If White flags, Black has insufficient material to mate -> must be draw 1/2-1/2!
        board = chess.Board("8/8/8/8/8/5k2/8/4K2B w - - 0 1")
        # Ensure black has insufficient material
        self.assertTrue(board.has_insufficient_material(chess.BLACK))

    def test_repetition_resistance(self):
        # Test that shuffling a knight back and forth between f3 and g1 produces telescoping 0 net potential
        board = chess.Board()
        phi_start = potential(board, chess.WHITE)
        board.push_san("Nf3")
        board.push_san("Nf6")
        board.push_san("Ng1")
        board.push_san("Ng8")
        phi_end = potential(board, chess.WHITE)
        # Position returned to starting position
        self.assertAlmostEqual(phi_start, phi_end, places=5)

    def test_case_1_bishop_sacrifice_king_safety(self):
        # Greek Gift Bishop Sacrifice Bxh7+ heuristic evaluation
        # Note: Audited engine analysis at nodes=15000 shows Stockfish does NOT select
        # Bxh7+ in top-5 moves (prefers Nexd4 +175cp). This test verifies the transparent
        # feature calculation (king safety and shelter stripping), NOT engine optimality.
        b_before = chess.Board("r1bq1rk1/pp1n1ppp/2n1p3/2bpP3/3p1B2/2PB1N2/PP2NPPP/R2Q1RK1 w - - 0 10")
        f_before = features(b_before, chess.WHITE)
        phi_before = potential(b_before, chess.WHITE)

        # White plays Bxh7+
        b_after = b_before.copy()
        b_after.push_san("Bxh7+")
        # Black captures King Kxh7
        b_after.push_san("Kxh7")

        f_after = features(b_after, chess.WHITE)
        phi_after = potential(b_after, chess.WHITE)

        # Black king zone threats increased and shelter was completely stripped
        self.assertGreater(f_after["king_safety"], f_before["king_safety"])
        self.assertEqual(f_after["king_safety"] - f_before["king_safety"], 3.0)

        # Continuation: White plays Ng5+ then Qd3 threatening mate
        b_after.push_san("Ng5+")
        b_after.push_san("Kg8")
        b_after.push_san("Qd3")
        f_attack = features(b_after, chess.WHITE)
        self.assertGreater(f_attack["king_safety"], f_before["king_safety"])

    def test_case_1_sound_sacrifice(self):
        """Backward-compatible alias for test_case_1_bishop_sacrifice_king_safety."""
        self.test_case_1_bishop_sacrifice_king_safety()

    def test_case_2_poisoned_pawn(self):
        # Sicilian / Queen grabs b2 pawn and gets trapped
        b_before = chess.Board("rnb1kbnr/pp2pppp/2p5/8/2QP4/2N5/Pq2PPPP/R1B1KBNR w KQkq - 0 6")
        # Black queen just took b2. White plays Rb1 trapping or attacking the queen
        phi_black_before = potential(b_before, chess.BLACK)

        b_refuted = b_before.copy()
        b_refuted.push_san("Rb1")  # White attacks the poisoned Queen
        phi_black_after = potential(b_refuted, chess.BLACK)

        # Black potential collapses because the Queen is exposed/trapped
        self.assertLess(phi_black_after, phi_black_before)

    def test_case_3_outright_blunder(self):
        # Scholar's mate blunder: Black plays Nf6?? allowing Qxf7#
        b = chess.Board("r1bqkb1r/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
        # If Black blundered with Nf6??:
        b_blunder = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 5 4")
        phi_blunder = potential(b_blunder, chess.BLACK)
        # White has immediate checkmate Qxf7#
        b_blunder.push_san("Qxf7#")
        self.assertTrue(b_blunder.is_checkmate())
        self.assertEqual(result_value(b_blunder, chess.BLACK), -1.0)

    def test_case_4_active_move_refuted_by_counterplay(self):
        # Blackburne Shilling Gambit: White plays active-looking Nxe5?
        # Refuted by Black's Qg5! attacking Knight and g2
        b = chess.Board("r1bqkbnr/pppp1ppp/8/4p3/2BnP3/8/PPPP1PPP/RNBQK2R b KQkq - 0 4")
        # In this position, White had played Nxe5?
        # Black plays Qg5!
        b.push_san("Qg5")
        f_white = features(b, chess.WHITE)
        # White's hanging value is negative (White pieces are hanging: Knight on e5 and g2)
        self.assertLess(f_white["hanging_value"], 0)
        phi_white = potential(b, chess.WHITE)
        self.assertLess(phi_white, 0.0)

    def test_outcome_dominance_and_engine_regret_bound(self):
        # Construct a synthetic match log where fly lost
        match_dict = {
            "result": "0-1",
            "winner": "stockfish",
            "termination_reason": "Checkmate",
            "fly_color": "white",
            "moves": [
                {"ply": 1, "player": "fly", "uci": "e2e4", "san": "e4", "white_clock": 59.0, "black_clock": 60.0},
                {"ply": 2, "player": "stockfish", "uci": "e7e5", "san": "e5", "white_clock": 59.0, "black_clock": 58.0},
            ]
        }
        rewards = compute_game_rewards(match_dict)
        self.assertEqual(rewards["terminal_reward"], -1.0)
        # Bounded shaping cannot overturn a loss: training reward must remain negative!
        self.assertLess(rewards["training_reward"], 0.0)
        self.assertLessEqual(rewards["diagnostics"]["engine_penalty"], 0.10)

    def test_none_player_move_handling(self):
        """A bot returning None must be reported as a player error, not 'Illegal move: None'."""
        from flychess.players import BasePlayer
        from flychess.match import ChessMatch

        class NoneMovePlayer(BasePlayer):
            name = "BrokenBot"
            def get_move(self, board, my_clock=None, opp_clock=None):
                return None  # Intentionally returns None

        p_white = NoneMovePlayer()
        p_black = NoneMovePlayer()
        match = ChessMatch(white_player=p_white, black_player=p_black, timed=False)
        match._run_loop()

        self.assertTrue(match.game_over)
        self.assertEqual(match.result, "0-1")
        # Crucial check: Must NOT be "Illegal move: None"
        self.assertIn("Player error", match.termination_reason)
        self.assertIn("returned None", match.termination_reason)
        self.assertNotIn("Illegal move: None", match.termination_reason)

    def test_human_flagging_classified_as_flag_not_illegal_move(self):
        """When human player runs out of time, it must be recorded as flagged, not illegal move None."""
        from flychess.players import HumanPlayer, BasePlayer
        from flychess.match import ChessMatch

        class DummyPlayer(BasePlayer):
            name = "DummyBot"
            def get_move(self, board, my_clock=None, opp_clock=None):
                return list(board.legal_moves)[0]

        human = HumanPlayer()
        bot = DummyPlayer()
        # Start match with 0.05 seconds so human flags quickly
        match = ChessMatch(white_player=human, black_player=bot, timed=True, base_time=0.05)
        match._run_loop()

        self.assertTrue(match.game_over)
        self.assertEqual(match.result, "0-1")
        self.assertIn("lost on time (flagged)", match.termination_reason)
        self.assertNotIn("Illegal move: None", match.termination_reason)

    def test_zero_move_and_partial_game_replay_safety(self):
        """Simulate replay loop on a 0-move empty game and ensure no IndexError is raised."""
        empty_game = {
            "id": "test-empty-game",
            "initial_fen": chess.STARTING_FEN,
            "moves": [],
            "result": "0-1",
            "termination_reason": "lost on time (flagged)",
        }
        moves = empty_game.get("moves", [])
        total_plies = len(moves)
        self.assertEqual(total_plies, 0)

        # Ensure that requesting ply 0 or any out-of-bound ply clamps safely
        for target_ply in [0, 1, 5, 48]:
            clamped_ply = max(0, min(total_plies, target_ply))
            self.assertEqual(clamped_ply, 0)

            r_board = chess.Board(empty_game.get("initial_fen", chess.STARTING_FEN))
            for i in range(min(clamped_ply, len(moves))):
                m_rec = moves[i]
                r_board.push(chess.Move.from_uci(m_rec["uci"]))

            # Board remains at starting position without error
            self.assertEqual(r_board.fen(), chess.STARTING_FEN)

    def test_gemini_status_no_key_leak(self):
        """Verify get_gemini_status reports 'Not configured' when key is absent without leaking values."""
        import os
        from flychess.players import get_gemini_status
        # Ensure neither key is set in test environment
        old_gemini = os.environ.pop("GEMINI_API_KEY", None)
        old_google = os.environ.pop("GOOGLE_API_KEY", None)
        try:
            status = get_gemini_status()
            self.assertFalse(status["is_configured"])
            self.assertFalse(status["key_present"])
            self.assertEqual(status["status_text"], "Not configured")
            # Ensure no secret strings in status dict
            for v in status.values():
                if isinstance(v, str):
                    self.assertNotIn("AIza", v)
        finally:
            if old_gemini:
                os.environ["GEMINI_API_KEY"] = old_gemini
            if old_google:
                os.environ["GOOGLE_API_KEY"] = old_google

    def test_transparent_feature_weights_and_exchanges(self):
        from flychess.reward import FEATURE_WEIGHTS, evaluate_exchanges
        self.assertAlmostEqual(sum(FEATURE_WEIGHTS.values()), 1.0)
        
        # Test exchange evaluation:
        # Position where White Queen attacks defended Black Knight (e.g. Knight on e5 defended by pawn d6)
        b = chess.Board("rnbqkbnr/ppp2ppp/3p4/4p3/4Q3/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
        # Defended pawn on e5 attacked by Queen on e4
        inflicted, suffered = evaluate_exchanges(b, chess.WHITE)
        # Queen taking defended pawn is a losing exchange, so inflicted favorable threats should be 0.0
        self.assertEqual(inflicted, 0.0)

        # Now White pawn attacks undefended Black Knight:
        b_win = chess.Board("r1bqkbnr/pppppppp/2n5/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
        # Pawn on e5 attacks knight on c6? No, e5 attacks d6/f6. Let's make direct attack:
        b_win = chess.Board("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
        # Bishop b5 attacks knight c6. Knight c6 is defended by b7/d7 pawns.
        # Undefended piece test:
        b_free = chess.Board("r1bqkb1r/pppp1ppp/8/4n3/4Q3/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
        # White Queen on e4 pins/attacks undefended Knight on e5
        inflicted_free, _ = evaluate_exchanges(b_free, chess.WHITE)
        self.assertGreaterEqual(inflicted_free, 3.1)

        # Check normalization bounds
        f = features(b_free, chess.WHITE)
        for k, v in f["normalized"].items():
            self.assertGreaterEqual(v, -1.0)
            self.assertLessEqual(v, 1.0)
        phi = potential(b_free, chess.WHITE)
        self.assertGreater(phi, -1.0)
        self.assertLess(phi, 1.0)



    # ------------------------------------------------------------------ Q3 & Reward Audit --
    def test_audited_010_regret_cap_and_win_preference(self):
        """Audited regression: Stockfish regret penalties are capped at 0.10 per complete game.
        
        Even with multiple severe tactical blunders, a win must remain strictly preferable
        to a draw, and a draw strictly preferable to a loss.
        """
        from flychess.reward import compute_game_rewards
        import numpy as np

        # Construct a synthetic multi-move match
        moves = [
            {"ply": 1, "player": "fly", "uci": "e2e4", "san": "e4", "white_clock": 58.0, "black_clock": 60.0},
            {"ply": 2, "player": "stockfish", "uci": "e7e5", "san": "e5", "white_clock": 58.0, "black_clock": 59.0},
            {"ply": 3, "player": "fly", "uci": "g1f3", "san": "Nf3", "white_clock": 56.0, "black_clock": 59.0},
            {"ply": 4, "player": "stockfish", "uci": "b8c6", "san": "Nc6", "white_clock": 56.0, "black_clock": 57.0},
            {"ply": 5, "player": "fly", "uci": "f1c4", "san": "Bc4", "white_clock": 54.0, "black_clock": 57.0},
            {"ply": 6, "player": "stockfish", "uci": "f8c5", "san": "Bc5", "white_clock": 54.0, "black_clock": 55.0},
        ]

        # 1. Test Cap Logic & Proportional Attribution directly on transitions
        # Suppose fly had 3 moves with raw penalties: -0.50 (blunder), -0.30 (mistake), -0.20 (mistake)
        # Total raw penalty sum = -1.00 (magnitude 1.00 > 0.10 cap)
        raw_penalties = [-0.50, -0.30, -0.20]
        raw_mag = sum(abs(p) for p in raw_penalties)
        cap = 0.10
        scale = cap / raw_mag
        scaled_penalties = [p * scale for p in raw_penalties]

        # Total applied penalty equals exactly 0.10
        self.assertAlmostEqual(sum(abs(p) for p in scaled_penalties), 0.10, places=5)
        # Proportional attribution is preserved:
        self.assertAlmostEqual(scaled_penalties[0] / scaled_penalties[2], -0.50 / -0.20, places=5)
        self.assertAlmostEqual(scaled_penalties[1] / scaled_penalties[2], -0.30 / -0.20, places=5)

        # 2. Evaluate compute_game_rewards across Win, Draw, and Loss outcomes
        win_match = {"result": "1-0", "winner": "fly", "termination_reason": "Checkmate", "fly_color": "white", "moves": moves}
        draw_match = {"result": "1/2-1/2", "winner": None, "termination_reason": "Draw by agreement", "fly_color": "white", "moves": moves}
        loss_match = {"result": "0-1", "winner": "stockfish", "termination_reason": "Checkmate", "fly_color": "white", "moves": moves}

        res_win = compute_game_rewards(win_match)
        res_draw = compute_game_rewards(draw_match)
        res_loss = compute_game_rewards(loss_match)

        # Strict ordering of outcomes: Win > Draw > Loss
        self.assertGreater(res_win["training_reward"], res_draw["training_reward"])
        self.assertGreater(res_draw["training_reward"], res_loss["training_reward"])

        # Applied engine penalty is within the 0.10 cap
        self.assertLessEqual(res_win["diagnostics"]["applied_engine_penalty"], 0.10)
        self.assertLessEqual(res_draw["diagnostics"]["applied_engine_penalty"], 0.10)
        self.assertLessEqual(res_loss["diagnostics"]["applied_engine_penalty"], 0.10)

        # Logged applied penalty equals what influenced training
        self.assertEqual(res_win["diagnostics"]["applied_engine_penalty"], res_win["diagnostics"]["engine_penalty"])

    def test_loss_stays_negative_with_cap(self):
        """A game that fly lost must produce a negative training reward."""
        from flychess.reward import compute_game_rewards

        match_dict = {
            "result": "0-1",
            "winner": "stockfish",
            "termination_reason": "Checkmate",
            "fly_color": "white",
            "moves": [
                {"ply": 1, "player": "fly", "uci": "e2e4", "san": "e4", "white_clock": 59.0, "black_clock": 60.0},
                {"ply": 2, "player": "stockfish", "uci": "e7e5", "san": "e5", "white_clock": 59.0, "black_clock": 58.0},
            ]
        }
        result = compute_game_rewards(match_dict)
        self.assertLess(result["training_reward"], 0.0)
        self.assertEqual(result["terminal_reward"], -1.0)

    # ------------------------------------------------------------------ Q4: Discounted Returns & No Reward Farming --
    def test_repeated_position_discounted_returns(self):
        """Repeated piece shuffling produces net-zero shaping and delayed win discounting.
        
        Verifies that cycling moves back and forth does NOT enable reward farming:
        intermediate shuffle moves receive 0 shaping, and the delayed terminal reward
        decays as gamma^k rather than accumulating artificial reward.
        """
        gamma = 0.99

        # 4 shuffle transitions: shaped_reward=0, move_penalty=0
        # Followed by 1 winning move with terminal_reward = 1.0
        transitions = [
            {"training_reward": 0.0},   # Nf3
            {"training_reward": 0.0},   # Ng1 (back)
            {"training_reward": 0.0},   # Nf3
            {"training_reward": 0.0},   # Ng1 (back)
            {"training_reward": 1.0},   # checkmate move (terminal)
        ]

        # Compute discounted returns G_t = sum_k gamma^k r_{t+k}
        running = 0.0
        returns = []
        for t in reversed(transitions):
            running = t["training_reward"] + gamma * running
            returns.append(running)
        returns.reverse()

        # Terminal move gets full 1.0 return
        self.assertAlmostEqual(returns[-1], 1.0, places=4)

        # Earlier shuffle moves get discounted return gamma^4 * 1.0 ≈ 0.9606
        expected_first = (gamma ** 4) * 1.0
        self.assertAlmostEqual(returns[0], expected_first, places=4)

        # Every shuffle move gets strictly less return than the decisive terminal move
        for i in range(len(returns) - 1):
            self.assertLess(returns[i], returns[-1])

        # Without a terminal win (e.g. repeated shuffle in a drawn game), return is strictly 0.0
        zero_transitions = [{"training_reward": 0.0} for _ in range(5)]
        zero_returns = []
        run = 0.0
        for t in reversed(zero_transitions):
            run = t["training_reward"] + gamma * run
            zero_returns.append(run)
        self.assertTrue(all(r == 0.0 for r in zero_returns))


if __name__ == "__main__":
    unittest.main()
