from __future__ import annotations

from typing import Optional, Tuple

from src.core.board import Board
from src.core.types import Move
from src.search.eval import evaluate
from src.search.tt import Bound, TTEntry, TranspositionTable

INF = 100_000
MATE_SCORE = 10_000


def alphabeta(
    board: Board,
    depth: int,
    alpha: int,
    beta: int,
    ply: int = 0,
    tt: Optional[TranspositionTable] = None,
) -> int:
    if depth <= 0:
        return evaluate(board)

    alpha_orig = alpha
    tt_move: Optional[Move] = None
    if tt is not None:
        entry = tt.get(board.hash)
        if entry is not None and entry.depth >= depth:
            if entry.bound == Bound.EXACT:
                return entry.score
            if entry.bound == Bound.LOWER and entry.score >= beta:
                return entry.score
            if entry.bound == Bound.UPPER and entry.score <= alpha:
                return entry.score
        if entry is not None:
            tt_move = entry.best_move

    legal = board.generate_legal()
    if not legal:
        if board.in_check(board.side_to_move):
            return -MATE_SCORE + ply
        return 0

    if tt_move is not None:
        try:
            idx = legal.index(tt_move)
        except ValueError:
            idx = -1
        if idx > 0:
            legal[0], legal[idx] = legal[idx], legal[0]

    best = -INF
    best_move: Optional[Move] = None
    for move in legal:
        undo = board.make_move(move)
        if undo is None:
            continue
        score = -alphabeta(board, depth - 1, -beta, -alpha, ply + 1, tt=tt)
        board.unmake_move(undo)
        if score > best:
            best = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            if tt is not None:
                tt.store(TTEntry(board.hash, depth, best, Bound.LOWER, best_move))
            break

    if tt is not None and alpha < beta:
        bound = Bound.EXACT if best > alpha_orig else Bound.UPPER
        tt.store(TTEntry(board.hash, depth, best, bound, best_move))
    return best


def search(
    board: Board, depth: int, tt: Optional[TranspositionTable] = None
) -> Tuple[int, Optional[Move]]:
    """
    Alpha-beta root search. Returns (score, best_move).
    """
    if depth <= 0:
        return evaluate(board), None

    legal = board.generate_legal()
    if not legal:
        if board.in_check(board.side_to_move):
            return -MATE_SCORE, None
        return 0, None

    alpha = -INF
    beta = INF
    best_score = -INF
    best_move: Optional[Move] = None

    tt_move: Optional[Move] = None
    if tt is not None:
        entry = tt.get(board.hash)
        if entry is not None:
            tt_move = entry.best_move
    if tt_move is not None:
        try:
            idx = legal.index(tt_move)
        except ValueError:
            idx = -1
        if idx > 0:
            legal[0], legal[idx] = legal[idx], legal[0]

    for move in legal:
        undo = board.make_move(move)
        if undo is None:
            continue
        score = -alphabeta(board, depth - 1, -beta, -alpha, ply=1, tt=tt)
        board.unmake_move(undo)
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score

    if tt is not None:
        tt.store(TTEntry(board.hash, depth, best_score, Bound.EXACT, best_move))
    return best_score, best_move
