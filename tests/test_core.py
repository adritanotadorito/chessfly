import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np

from flychess.board import DIM, encode
from flychess.env import ChessEnv
from flychess.graph import build_graph, make_fixture, load_graph
from flychess.model import FlyPolicy
from flychess.reward import potential, result_value


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.graph = str(Path(self.tmp.name) / "fixture.npz")
        make_fixture(self.graph)

    def tearDown(self):
        self.tmp.cleanup()

    def test_board_and_legal_policy(self):
        board = chess.Board()
        self.assertEqual(encode(board).shape, (DIM,))
        model = FlyPolicy(self.graph)
        self.assertIn(model.choose(board), board.legal_moves)
        self.assertFalse(model.graph_meta["authentic_flywire"])

    def test_promotions(self):
        board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
        model = FlyPolicy(self.graph)
        legal, _, probs, _ = model.forward(board)
        self.assertIn(chess.Move.from_uci("a7a8n"), legal)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_training_updates_fly_edges(self):
        model = FlyPolicy(self.graph)
        before = model.params["gain"].copy()
        board = chess.Board()
        loss, grads = model.gradients(board, chess.Move.from_uci("e2e4"))
        self.assertGreater(loss, 0)
        self.assertGreater(float(np.linalg.norm(grads["gain"])), 0)
        model.update(grads)
        self.assertFalse(np.array_equal(before, model.params["gain"]))

    def test_env_budgets_and_finish(self):
        env = ChessEnv(max_agent_moves=2)
        try:
            self.assertEqual(len(env.observe()["legal_moves"]), 20)
            env.inspect_line(["e2e4", "e7e5"])
            self.assertEqual(env.observe()["inspections_left"], 3)
            self.assertIn("error", env.play("e2e5"))
            self.assertFalse(env.done)
            for _ in range(2):
                result = env.play(env.observe()["legal_moves"][0])
            self.assertTrue(result["observation"]["done"])
            self.assertEqual(len(env.history), 4)
        finally:
            env.close()

    def test_real_csv_provenance_and_degree_preserving_control(self):
        path = Path(self.tmp.name) / "connections.csv"
        with path.open("w") as f:
            f.write("pre_root_id,post_root_id,syn_count\n")
            for i in range(12):
                f.write(f"{i},{(i+1)%12},{100-i}\n")
                f.write(f"{i},{(i+3)%12},{20-i}\n")
        out = str(Path(self.tmp.name) / "graph.npz")
        meta = build_graph(str(path), out, nodes=8, pool=100)
        self.assertEqual(meta["nodes"], 8)
        src, dst, _, _, n = load_graph(out)
        s2, d2, _, _, _ = load_graph(out, shuffled=True)
        np.testing.assert_array_equal(np.bincount(src, minlength=n), np.bincount(s2, minlength=n))
        np.testing.assert_array_equal(np.bincount(dst, minlength=n), np.bincount(d2, minlength=n))

    def test_result_and_bounded_potential(self):
        self.assertLess(abs(potential(chess.Board(), chess.WHITE)), 1.0)
        mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(result_value(mate, chess.WHITE), 1.0)


if __name__ == "__main__":
    unittest.main()
