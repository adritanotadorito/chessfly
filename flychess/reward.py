"""Interpretable diagnostics and bounded training potential; not Stockfish's NNUE."""
import chess
import numpy as np

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3.1, chess.BISHOP: 3.3,
          chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def features(board: chess.Board, color: chess.Color) -> dict:
    def side(c):
        material = activity = coordination = exposed = passed = 0.0
        for sq, piece in board.piece_map().items():
            if piece.color != c:
                continue
            material += VALUES[piece.piece_type]
            if piece.piece_type != chess.KING:
                defenders = len(board.attackers(c, sq))
                coordination += min(defenders, 2)
                attackers = len(board.attackers(not c, sq))
                exposed += VALUES[piece.piece_type] * float(attackers > defenders and attackers > 0)
                for target in board.attacks(sq):
                    own = board.piece_at(target)
                    if own is None or own.color != c:
                        activity += 1.0 if not board.is_attacked_by(not c, target) else 0.2
            if piece.piece_type == chess.PAWN:
                rank, file = chess.square_rank(sq), chess.square_file(sq)
                ahead = range(rank+1, 8) if c else range(rank-1, -1, -1)
                if all(board.piece_at(chess.square(f, r)) != chess.Piece(chess.PAWN, not c)
                       for f in range(max(0, file-1), min(7, file+1)+1) for r in ahead):
                    passed += 1
        king = board.king(c)
        king_threat = (sum(board.is_attacked_by(not c, s) for s in board.attacks(king))
                       if king is not None else 8)
        return np.array([material, activity, coordination, -exposed, passed, -king_threat],
                        dtype=np.float32)
    diff = side(color) - side(not color)
    return dict(zip(("material", "safe_activity", "coordination", "hanging_value",
                     "passed_pawns", "king_safety"), map(float, diff)))


def potential(board: chess.Board, color: chess.Color) -> float:
    """A bounded heuristic for dense feedback; actual results are separate."""
    f = features(board, color)
    value = (0.40*f["material"] + 0.018*f["safe_activity"] +
             0.035*f["coordination"] + 0.20*f["hanging_value"] +
             0.20*f["passed_pawns"] + 0.12*f["king_safety"])
    return float(np.tanh(value / 5.0))


def result_value(board: chess.Board, color: chess.Color):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    return 0.0 if outcome.winner is None else (1.0 if outcome.winner == color else -1.0)


def move_audit(engine, board: chess.Board, move: chess.Move, nodes: int = 10000):
    """Engine evaluation before and after a move, always from mover's view."""
    import chess.engine
    color = board.turn
    limit = chess.engine.Limit(nodes=nodes)
    best = engine.analyse(board, limit)["score"].pov(color).score(mate_score=100000)
    after = board.copy(stack=True)
    after.push(move)
    chosen = engine.analyse(after, limit)["score"].pov(color).score(mate_score=100000)
    return {"best_cp": best, "chosen_cp": chosen,
            "loss_cp": max(0, best - chosen), "nodes_per_analysis": nodes}
