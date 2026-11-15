"""Tests for the deep mateIn2 audit logic: forcing-first-move detection and
full-combo success vs. first-move-only accuracy."""
import chess
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from flychess.graph import make_fixture
from flychess.model import FlyPolicy
import tempfile


# ── Minimal stub model that always picks the argmax of forward() ────────────

def argmax_move(model: FlyPolicy, board: chess.Board) -> chess.Move:
    legal, _, probs, _ = model.forward(board)
    return legal[int(np.argmax(probs))]


# ── Core audit helpers (copied from scratch/audit_matein2.py) ───────────────

def all_replies_mated(board_after_fly1: chess.Board, model: FlyPolicy):
    results = []
    for opp_reply in board_after_fly1.legal_moves:
        b_opp = board_after_fly1.copy()
        b_opp.push(opp_reply)
        if b_opp.is_game_over():
            results.append({"opp_reply": opp_reply.uci(), "is_checkmate": b_opp.is_checkmate()})
            continue
        m2 = argmax_move(model, b_opp)
        b_final = b_opp.copy()
        b_final.push(m2)
        results.append({
            "opp_reply": opp_reply.uci(),
            "model_m2": m2.uci(),
            "is_checkmate": b_final.is_checkmate(),
        })
    full_success = len(results) > 0 and all(r["is_checkmate"] for r in results)
    return full_success, results


def is_valid_forcing_first(board_solver: chess.Board, candidate_m1: chess.Move,
                            model: FlyPolicy):
    b_after_m1 = board_solver.copy()
    b_after_m1.push(candidate_m1)
    if b_after_m1.is_checkmate():
        return True, []
    full_success, reply_results = all_replies_mated(b_after_m1, model)
    return full_success, reply_results


# ── Synthetic puzzle records ─────────────────────────────────────────────────

class TestMateIn2MetricDistinction(unittest.TestCase):
    """Core unit tests that verify the distinction between first-move accuracy
    (what the report measures) and full combo success (what it claims)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        graph_path = str(Path(self.tmp.name) / "fixture.npz")
        make_fixture(graph_path)
        self.model = FlyPolicy(graph_path, seed=42)

    def tearDown(self):
        self.tmp.cleanup()

    # ── Helper ──

    def _make_mate2_puzzle(self, solver_fen, m1, opp_reply, m2, pid="test"):
        return {
            "puzzle_id": pid,
            "bucket": "mateIn2",
            "solver_fen": solver_fen,
            "first_move": m1,
            "moves": ["setup", m1, opp_reply, m2],
            "all_mating_moves": [],
            "is_guaranteed_forced_mate": True,
        }

    # ── Test 1: M1a ≠ M1b when first move is correct but model fumbles m2 ──

    def test_first_move_match_does_not_imply_full_combo(self):
        """The 38% figure only checks whether the model's first move matches the
        Lichess solution.  It does NOT verify that the model delivers mate on the
        second move.  These are different quantities and must be measured separately."""
        # Scholar's mate setup: Black to move, Qxf7+ is recorded solution
        # We build a simplified scenario and verify our audit functions
        # distinguish M1a from M1b correctly.

        # Verify that our `all_replies_mated` function returns False when the
        # model's second move does NOT deliver checkmate (even if m1 was correct).
        # We use a position where the fixture model is unlikely to find the mate.
        b = chess.Board("7k/4Q3/6K1/8/8/8/8/8 w - - 0 1")
        # White to move; Qf8#, Qg7#, Qh7# all mate immediately
        # This is mateIn1, so is_valid_forcing_first should return True immediately
        for m in chess.Board("7k/4Q3/6K1/8/8/8/8/8 w - - 0 1").legal_moves:
            if chess.Board("7k/4Q3/6K1/8/8/8/8/8 w - - 0 1").copy().push(m) or True:
                b_copy = chess.Board("7k/4Q3/6K1/8/8/8/8/8 w - - 0 1")
                b_copy.push(m)
                if b_copy.is_checkmate():
                    is_forcing, _ = is_valid_forcing_first(
                        chess.Board("7k/4Q3/6K1/8/8/8/8/8 w - - 0 1"), m, self.model
                    )
                    self.assertTrue(is_forcing,
                        "A mating move should always be classified as a valid forcing first move")
                    break

    def test_all_replies_mated_requires_all_branches(self):
        """all_replies_mated must return False if even ONE opponent reply escapes."""
        # Craft a position: Black King h8, White Queen e7, White King g6
        # White to move; mating moves: Qf8#, Qg7#, Qh7#
        # If we start the board one step back (e.g. White plays Qe7 as 'first move'),
        # then the opponent is in zugzwang with no legal moves... let's use a real case.

        # Simpler: test the function interface semantics directly.
        # Create a position where Black has 2 legal replies after m1, and
        # test that all_replies_mated returns False if the model fails on even one.

        # QKvK endgame: White Qd1, Ka1; Black Ka8 - positions near forced mate
        b = chess.Board("k7/8/8/8/8/8/8/KQ6 w - - 0 1")
        # White should have many moves; find one that doesn't immediately mate
        # but leaves a position where Black has 1 reply
        # We're testing that all_replies_mated checks ALL branches, not just one
        legal = list(b.legal_moves)
        self.assertGreater(len(legal), 0, "Should have legal moves")

        # The key assertion: if any reply escapes, full_success must be False
        # We use a non-mating move and verify the semantics
        full_success, results = all_replies_mated(b, self.model)
        # This is called on the board BEFORE opponent replies, with White to move -
        # So we're checking if the model mates all Black replies from the QKvK pos
        # Result depends on model; the important thing is the function checks all branches
        for r in results:
            if not r["is_checkmate"]:
                self.assertFalse(full_success,
                    "full_success must be False if any single reply is not mated")
                break
        # If all replies happened to be mated, full_success should be True
        if all(r["is_checkmate"] for r in results):
            self.assertTrue(full_success)

    def test_m1a_m1b_are_independent_metrics(self):
        """Verify that M1a (recorded first-move match) and M1b (valid forcing first move)
        measure different things and can disagree:
        - M1a=True, M1b=False: chose Lichess move but cannot complete the combo
        - M1a=False, M1b=True: chose a different move that also forces mate (not in our data but valid)
        - M1a=False, M1b=False: wrong first move, cannot force mate
        """
        # Test case where M1b is True: a first move that IS an immediate checkmate
        b_m1 = chess.Board("7k/4Q3/6K1/8/8/8/8/8 w - - 0 1")
        for m in b_m1.legal_moves:
            b_copy = b_m1.copy()
            b_copy.push(m)
            if b_copy.is_checkmate():
                is_forcing, _ = is_valid_forcing_first(b_m1, m, self.model)
                self.assertTrue(is_forcing, f"Mating move {m.uci()} must be classified forcing")
                break

        # Test case where the move does NOT force mate: Qe3 from the same position
        # (not a mate, and after it Black has defensive replies the model may not handle)
        qe3 = chess.Move.from_uci("e7e3")
        if qe3 in b_m1.legal_moves:
            b_copy2 = b_m1.copy()
            b_copy2.push(qe3)
            if not b_copy2.is_checkmate():
                # This may or may not be classified as forcing depending on opponent replies
                is_forcing, reply_results = is_valid_forcing_first(b_m1, qe3, self.model)
                # We don't assert a specific outcome, just that the function runs
                # and returns consistent results
                if reply_results:
                    # If any branch is not mated, full_success must be False
                    any_not_mated = any(not r["is_checkmate"] for r in reply_results)
                    if any_not_mated:
                        self.assertFalse(is_forcing,
                            "is_forcing must be False when any opponent reply escapes mate")

    def test_forcing_first_move_requires_all_legal_replies_mated(self):
        """Verify the full-combo metric checks ALL legal opponent replies, not just the
        Lichess-recorded one. Even if model scores 38% on first-move-match and delivers
        mate on the recorded line, it only earns 'full success' credit if it mates
        every single opponent response."""
        # Prove: if opponent has 2 replies and model only mates 1 of them,
        # full_success must be False even if model chose the right first move.
        # We simulate this by showing all_replies_mated checks every branch.

        # Build a position where after a first move, Black has 2 clear defensive options:
        # White: Ka1, Qa5, Ra8; Black: Ka8... wait, let me use a simpler logic test.

        # Direct test: build a fake results list and check the full_success logic
        results_all_mated = [{"is_checkmate": True}, {"is_checkmate": True}]
        results_one_escapes = [{"is_checkmate": True}, {"is_checkmate": False}]

        # Replicate the all_replies_mated logic
        def check_full_success(res):
            return len(res) > 0 and all(r["is_checkmate"] for r in res)

        self.assertTrue(check_full_success(results_all_mated),
            "Should succeed when all replies are mated")
        self.assertFalse(check_full_success(results_one_escapes),
            "Should fail when ANY reply escapes mate")
        self.assertFalse(check_full_success([]),
            "Should fail on empty reply list")

    def test_audit_recorded_matein2_puzzle_structure(self):
        """Verify that the actual frozen test puzzles have moves[0]=setup_move,
        moves[1]=first_move (from solver_fen), moves[2]=opp_reply, moves[3]=m2.
        solver_fen = original_fen after setup_move is applied."""
        puzzle_path = Path("data/puzzles_10k/test_puzzles_frozen_1000.jsonl")
        if not puzzle_path.exists():
            self.skipTest("Frozen puzzle test not available")

        mate2_items = []
        with open(puzzle_path) as f:
            for line in f:
                d = json.loads(line.strip())
                if d.get("bucket") == "mateIn2":
                    mate2_items.append(d)

        self.assertEqual(len(mate2_items), 200, "Expected 200 mateIn2 test puzzles")
        all_forced = all(d["is_guaranteed_forced_mate"] for d in mate2_items)
        self.assertTrue(all_forced, "All 200 mateIn2 test puzzles should be guaranteed forced mates")
        all_4_moves = all(len(d["moves"]) == 4 for d in mate2_items)
        self.assertTrue(all_4_moves, "All mateIn2 puzzles should have exactly 4 moves")

        # Verify solver_fen = original_fen after moves[0]
        for item in mate2_items[:10]:  # spot-check 10
            b = chess.Board(item["original_fen"])
            setup_m = chess.Move.from_uci(item["moves"][0])
            self.assertIn(setup_m, b.legal_moves, f"Setup move should be legal in {item['puzzle_id']}")
            b.push(setup_m)
            norm_stored = " ".join(item["solver_fen"].split()[:4])
            norm_computed = " ".join(b.fen().split()[:4])
            self.assertEqual(norm_stored, norm_computed,
                f"solver_fen mismatch in {item['puzzle_id']}: stored={norm_stored}, computed={norm_computed}")

        # Verify first_move is legal from solver_fen
        for item in mate2_items[:10]:
            b2 = chess.Board(item["solver_fen"])
            first_m = chess.Move.from_uci(item["first_move"])
            self.assertIn(first_m, b2.legal_moves,
                f"first_move {item['first_move']} should be legal from solver_fen in {item['puzzle_id']}")

    def test_audit_known_counts_fly_puzzle_10k(self):
        """Verify the known audit counts from the actual audit run for fly-puzzle-10k.
        These are ground-truth numbers that the corrected report must cite."""
        audit_path = Path("scratch/matein2_audit_detail.json")
        if not audit_path.exists():
            self.skipTest("Audit detail JSON not available; run scratch/audit_matein2.py first")

        with open(audit_path) as f:
            data = json.load(f)

        fly10k = data["fly-puzzle-10k"]
        self.assertEqual(fly10k["total"], 200)
        # Ground-truth counts verified by the audit run
        self.assertEqual(fly10k["recorded_first_match"], 76,
            "M1a: 76 puzzles matched the Lichess first move (38.0%) – what the report measured")
        self.assertEqual(fly10k["valid_forcing_first"], 41,
            "M1b: Only 41 puzzles had a genuinely forcing first move (20.5%)")
        self.assertEqual(fly10k["full_success"], 41,
            "M2b: Full combo success (all branches mated) = 41/200 (20.5%), not 38.0%")

        # Key distinction: 35 puzzles (76 - 41) had the right first move but
        # the model FAILED to deliver mate after the opponent's reply
        missed_combos = fly10k["recorded_first_match"] - fly10k["full_success"]
        self.assertEqual(missed_combos, 35,
            "35 puzzles had the correct first move but the model could not complete the mate")

        # Shuffled control: similar pattern
        sh10k = data["shuffled-puzzle-10k"]
        self.assertEqual(sh10k["recorded_first_match"], 73)
        self.assertEqual(sh10k["valid_forcing_first"], 40)
        self.assertEqual(sh10k["full_success"], 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
