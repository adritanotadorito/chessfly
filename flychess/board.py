import chess
import numpy as np

PIECES = "PNBRQKpnbrqk"
PROMOS = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
DIM = 12 * 64 + 1 + 4 + 8 + 1 + 1


def encode(board: chess.Board) -> np.ndarray:
    """Piece planes, turn, castling, en passant, halfmove and repetition flag."""
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
    return x


def move_parts(move: chess.Move) -> tuple[int, int, int]:
    return move.from_square, move.to_square, PROMOS[move.promotion]
