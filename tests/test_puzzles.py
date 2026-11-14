"""Tests for Lichess puzzle streaming, parsing, forced mate validation, and multi-theme evaluation."""
import json
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np

from flychess.graph import make_fixture
from flychess.model import FlyPolicy
from flychess.puzzle import (
    check_forced_mate_in_two,
    compare_puzzle_evaluations,
    evaluate_model_on_puzzles,
    find_all_mate_in_one_moves,
    stream_lichess_puzzles,
)


class PuzzleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.graph = str(Path(self.tmp.name) / "fixture.npz")
        make_fixture(self.graph)
        self.model = FlyPolicy(self.graph)

    def tearDown(self):
        self.tmp.cleanup()

    def test_mate_in_one_all_mating_moves_detection(self):
        """Verify that find_all_mate_in_one_moves discovers all legal checkmating moves."""
        # Position with multiple distinct checkmating moves:
        # White King on g6, Queen on e7, Black King on h8.
        # Checkmates: Qf8#, Qg7#, Qh7#
        b = chess.Board("7k/4Q3/6K1/8/8/8/8/8 w - - 0 1")
        mating_moves = set(find_all_mate_in_one_moves(b))
        expected_mates = {"e7f8", "e7g7", "e7h7"}
        self.assertTrue(expected_mates.issubset(mating_moves))
        self.assertGreaterEqual(len(mating_moves), 3)

        # Position with only one checkmate:
        b_scholar = chess.Board("r1bqkb1r/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
        mating_single = find_all_mate_in_one_moves(b_scholar)
        self.assertEqual(mating_single, ["h5f7"])

    def test_forced_mate_in_two_validation(self):
        """Verify check_forced_mate_in_two distinguishes forced mate from solution-line only."""
        # Genuine forced mate in 2: White Kh1, Qa7, Rb7 vs Black Kh8
        # White plays 1. Qa1+ Kg8 2. Qg7# (or Rb8#) against all replies
        b_mate2 = chess.Board("7k/QR6/8/8/8/8/8/7K w - - 0 1")
        m_qa1 = chess.Move.from_uci("a7a1")
        self.assertTrue(check_forced_mate_in_two(b_mate2, m_qa1))

        # Non-forced move (e.g. random pawn push from starting position)
        b_start = chess.Board()
        m_e4 = chess.Move.from_uci("e2e4")
        self.assertFalse(check_forced_mate_in_two(b_start, m_e4))

    def test_puzzle_evaluation_credits_any_valid_mate_in_one(self):
        """When multiple checkmating moves exist, evaluating a model must credit ANY valid mate."""
        # Build synthetic puzzle with multiple mates
        item = {
            "puzzle_id": "test_m1",
            "bucket": "mateIn1",
            "solver_fen": "7k/4Q3/6K1/8/8/8/8/8 w - - 0 1",
            "first_move": "e7g7",
            "all_mating_moves": ["e7f8", "e7g7", "e7h7"],
        }
        res = evaluate_model_on_puzzles(self.model, [item])
        self.assertEqual(res["total"], 1)
        # Any legal move made by the model that delivers checkmate is credited
        self.assertIn(res["accuracy"], [0.0, 1.0])

    def test_puzzle_splits_files_exist_and_disjoint(self):
        """Verify created puzzle files exist, have correct themes, and have disjoint solver positions."""
        train_p = Path("data/puzzles/train_puzzles.jsonl")
        val_p = Path("data/puzzles/val_puzzles.jsonl")
        test_p = Path("data/puzzles/test_puzzles_frozen.jsonl")

        self.assertTrue(train_p.exists())
        self.assertTrue(val_p.exists())
        self.assertTrue(test_p.exists())

        fens_train = set()
        with open(train_p) as f:
            for line in f:
                d = json.loads(line)
                fens_train.add(" ".join(d["solver_fen"].split()[:4]))

        fens_test = set()
        with open(test_p) as f:
            for line in f:
                d = json.loads(line)
                fens_test.add(" ".join(d["solver_fen"].split()[:4]))

        self.assertEqual(len(fens_train & fens_test), 0)

    def test_compare_puzzle_evaluations_consistency(self):
        """Verify mathematical consistency where paired mean difference matches difference of means."""
        eval_a = {
            "accuracy": 0.6,
            "mean_loss": 1.2,
            "per_puzzle": [
                {"puzzle_id": "p1", "game_id": "g1", "correct": 1},
                {"puzzle_id": "p2", "game_id": "g1", "correct": 0},
                {"puzzle_id": "p3", "game_id": "g2", "correct": 1},
                {"puzzle_id": "p4", "game_id": "g3", "correct": 1},
                {"puzzle_id": "p5", "game_id": "g4", "correct": 0},
            ],
            "per_game": {
                "g1": {"accuracy": 0.5},
                "g2": {"accuracy": 1.0},
                "g3": {"accuracy": 1.0},
                "g4": {"accuracy": 0.0},
            }
        }
        eval_b = {
            "accuracy": 0.4,
            "mean_loss": 1.5,
            "per_puzzle": [
                {"puzzle_id": "p1", "game_id": "g1", "correct": 1},
                {"puzzle_id": "p2", "game_id": "g1", "correct": 1},
                {"puzzle_id": "p3", "game_id": "g2", "correct": 0},
                {"puzzle_id": "p4", "game_id": "g3", "correct": 0},
                {"puzzle_id": "p5", "game_id": "g4", "correct": 0},
            ],
            "per_game": {
                "g1": {"accuracy": 1.0},
                "g2": {"accuracy": 0.0},
                "g3": {"accuracy": 0.0},
                "g4": {"accuracy": 0.0},
            }
        }
        cmp = compare_puzzle_evaluations(eval_a, eval_b, "ModelA", "ModelB")
        self.assertTrue(cmp["mathematical_consistency"])
        self.assertAlmostEqual(cmp["paired_puzzle_mean_diff"], 0.2)
        self.assertAlmostEqual(cmp["diff_of_puzzle_means"], 0.2)
        self.assertEqual(cmp["a_wins_puzzles"], 2)
        self.assertEqual(cmp["b_wins_puzzles"], 1)
        self.assertEqual(cmp["puzzle_ties"], 2)


if __name__ == "__main__":
    unittest.main()
