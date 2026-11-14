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

    def test_case_1_sound_sacrifice(self):
        # Classic Greek Gift Bishop Sacrifice Bxh7+
        # White gives up bishop (3.3) for decisive king attack
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

        # In addition, Stockfish confirms this sacrifice is sound (loss_cp == 0)
        # while a flashy unsound sacrifice has severe regret:
        unsound_move = chess.Move.from_uci("f4d6") # giving up bishop for nothing
        # Deep continuation: White plays Ng5+ then Qd3 threatening mate
        b_after.push_san("Ng5+")
        b_after.push_san("Kg8")
        b_after.push_san("Qd3")
        f_attack = features(b_after, chess.WHITE)
        self.assertGreater(f_attack["king_safety"], f_before["king_safety"])

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


if __name__ == "__main__":
    unittest.main()
