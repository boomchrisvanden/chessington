from __future__ import annotations

from src.core.board import Board
from src.core.types import Color, PieceType

PIECE_VALUES = {
    PieceType.PAWN: 1,
    PieceType.KNIGHT: 3,
    PieceType.BISHOP: 3,
    PieceType.ROOK: 5,
    PieceType.QUEEN: 9,
    PieceType.KING: 0,
}


def evaluate(board: Board) -> int:
    """
    Basic material-only evaluation.

    Returns score from the side to move's perspective.
    """
    total = 0
    for piece in board.squares:
        if piece is None:
            continue
        color, pt = piece
        value = PIECE_VALUES.get(pt, 0)
        total += value if color == Color.WHITE else -value

    return total if board.side_to_move == Color.WHITE else -total
