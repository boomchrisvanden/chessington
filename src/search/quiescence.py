from __future__ import annotations

from typing import Optional

from src.core.board import Board
from src.core.types import Move, PieceType
from src.search.eval import evaluate
from src.search.tt import TranspositionTable

INF = 100_000

# Piece values for SEE-based delta pruning in quiescence.
_QS_PIECE_VAL = {
    PieceType.PAWN: 100,
    PieceType.KNIGHT: 300,
    PieceType.BISHOP: 300,
    PieceType.ROOK: 500,
    PieceType.QUEEN: 900,
    PieceType.KING: 0,
}

DELTA_MARGIN = 200  # Must gain at least this much to keep searching captures.


def _is_capture(board: Board, move: Move) -> bool:
    """Check if a move is a capture (including en passant)."""
    if board.squares[move.to_sq] is not None:
        return True
    # En passant: pawn moves diagonally to empty square == ep_square.
    if board.ep_square is not None and move.to_sq == board.ep_square:
        piece = board.squares[move.from_sq]
        if piece is not None and piece[1] == PieceType.PAWN:
            return True
    return False


def _capture_value(board: Board, move: Move) -> int:
    """Return the material value of the captured piece (for MVV ordering)."""
    target = board.squares[move.to_sq]
    if target is not None:
        return _QS_PIECE_VAL.get(target[1], 0)
    # En passant captures a pawn.
    if board.ep_square is not None and move.to_sq == board.ep_square:
        return _QS_PIECE_VAL[PieceType.PAWN]
    return 0


def quiescence(
    board: Board,
    alpha: int,
    beta: int,
    ply: int = 0,
    tt: Optional[TranspositionTable] = None,
) -> int:
    """
    Quiescence search: resolve captures so the static eval is reliable.

    Searches all capture moves until the position is quiet, using
    stand-pat, delta pruning, and MVV ordering.
    """
    stand_pat = evaluate(board)

    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat

    moves = board.generate_pseudo_legal()
    captures = [m for m in moves if _is_capture(board, m)]

    # Order captures by victim value descending (MVV).
    captures.sort(key=lambda m: _capture_value(board, m), reverse=True)

    for move in captures:
        # Delta pruning: skip if even capturing the piece can't raise alpha.
        cap_val = _capture_value(board, move)
        if stand_pat + cap_val + DELTA_MARGIN < alpha:
            continue

        undo = board.make_move(move)
        if undo is None:
            continue
        if board.in_check(undo.side_to_move):
            board.unmake_move(undo)
            continue

        score = -quiescence(board, -beta, -alpha, ply + 1, tt=tt)
        board.unmake_move(undo)

        if score >= beta:
            return score
        if score > alpha:
            alpha = score

    return alpha
