from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from src.core.board import Board
from src.core.types import Move
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
    Very small (and currently very weak) search stub.

    Returns the first legal move, ignoring max_depth/time_ms/tt for now.
    """
    legal = board.generate_legal()
    if not legal:
        return SearchResult(score_cp=0, best_move=None, depth=0)

    return SearchResult(score_cp=0, best_move=legal[0], depth=1)

