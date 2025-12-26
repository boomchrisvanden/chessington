from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.core.board import Board
from src.core.types import Move
from src.search.alphabeta import search as alphabeta_search
from src.search.tt import TranspositionTable


@dataclass(slots=True)
class SearchResult:
    score_cp: int
    best_move: Optional[Move]
    depth: int


def iterative_deepening(
    board: Board,
    max_depth: int,
    time_ms: int = 0,
    tt: Optional[TranspositionTable] = None,
) -> SearchResult:
    """
    Basic iterative deepening over alpha-beta search.

    time_ms/tt are placeholders for future work.
    """
    if max_depth <= 0:
        return SearchResult(score_cp=0, best_move=None, depth=0)

    best_score = 0
    best_move: Optional[Move] = None
    depth_reached = 0

    for depth in range(1, max_depth + 1):
        score, move = alphabeta_search(board, depth, tt=tt)
        depth_reached = depth
        best_score = score
        best_move = move
        if move is None:
            break

    return SearchResult(score_cp=best_score, best_move=best_move, depth=depth_reached)
