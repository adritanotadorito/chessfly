import chess
import numpy as np

PIECES = "PNBRQKpnbrqk"
PROMOS = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
# 768 piece planes + 1 turn + 4 castling + 8 en passant + 1 halfmove + 1 repetition + 2 clocks (my_clock, opp_clock)
DIM = 12 * 64 + 1 + 4 + 8 + 1 + 1 + 2


def encode(board: chess.Board, my_clock: float = None, opp_clock: float = None) -> np.ndarray:
    """Piece planes, turn, castling, en passant, halfmove, repetition flag, and remaining clocks."""
    x = np.zeros(DIM, dtype=np.float32)
    for sq, piece in board.piece_map().items():
        x[PIECES.index(piece.symbol()) * 64 + sq] = 1.0
    offset = 12 * 64
    x[offset] = 1.0 if board.turn == chess.WHITE else -1.0
    offset += 1
    x[offset:offset+4] = [float(board.has_kingside_castling_rights(chess.WHITE)),
                          float(board.has_queenside_castling_rights(chess.WHITE)),
                          float(board.has_kingside_castling_rights(chess.BLACK)),
                          float(board.has_queenside_castling_rights(chess.BLACK))]
    offset += 4
    if board.ep_square is not None:
        x[offset + chess.square_file(board.ep_square)] = 1.0
    offset += 8
    x[offset] = min(board.halfmove_clock, 100) / 100.0
    x[offset + 1] = float(board.is_repetition(2))
    offset += 2
    # Clocks normalized to [0, 1] relative to standard 60-second bullet base time.
    x[offset] = 1.0 if my_clock is None else float(np.clip(my_clock / 60.0, 0.0, 1.0))
    x[offset + 1] = 1.0 if opp_clock is None else float(np.clip(opp_clock / 60.0, 0.0, 1.0))
    return x


def move_parts(move: chess.Move) -> tuple[int, int, int]:
    return move.from_square, move.to_square, PROMOS[move.promotion]
