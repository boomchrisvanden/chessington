from __future__ import annotations

from typing import Optional

from src.core.board import Board
from src.core.types import PieceType
from src.search.eval import evaluate
from src.search.tt import Bound, TTEntry, TranspositionTable

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


def _capture_value(board: Board, move) -> int:
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
    stand-pat, delta pruning, MVV ordering, and TT probing.
    """
    # TT probe.
    alpha_orig = alpha
    tt_score = None
    if tt is not None:
        entry = tt.get(board.hash)
        if entry is not None and entry.depth >= 0:
            if entry.bound == Bound.EXACT:
                return entry.score
            if entry.bound == Bound.LOWER and entry.score >= beta:
                return entry.score
            if entry.bound == Bound.UPPER and entry.score <= alpha:
                return entry.score
            tt_score = entry.score

    stand_pat = evaluate(board)

    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat

    captures = board.generate_captures()

    # Order captures by victim value descending (MVV).
    captures.sort(key=lambda m: _capture_value(board, m), reverse=True)

    best = stand_pat

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

        if score > best:
            best = score
        if score >= beta:
            if tt is not None:
                tt.store(TTEntry(board.hash, 0, score, Bound.LOWER, None))
            return score
        if score > alpha:
            alpha = score

    # TT store.
    if tt is not None:
        bound = Bound.EXACT if best > alpha_orig else Bound.UPPER
        tt.store(TTEntry(board.hash, 0, best, bound, None))

    return best
