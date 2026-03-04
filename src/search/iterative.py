from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.core.board import Board
from src.core.types import Move
from src.search.alphabeta import INF, search as alphabeta_search
from src.search.ordering import MoveOrderer
from src.search.tt import TranspositionTable

# Aspiration window constants.
_ASP_INITIAL_DELTA = 25  # Starting window half-width in centipawns.
_ASP_MIN_DEPTH = 4       # Use full window below this depth.


@dataclass(slots=True)
class SearchResult:
    score_cp: int
    best_move: Optional[Move]
    depth: int
    nodes: int = 0


def iterative_deepening(
    board: Board,
    max_depth: int,
    time_ms: int = 0,
    tt: Optional[TranspositionTable] = None,
) -> SearchResult:
    """
    Iterative deepening over PVS search with aspiration windows.
    """
    if max_depth <= 0:
        return SearchResult(score_cp=0, best_move=None, depth=0)

    orderer = MoveOrderer()
    best_score = 0
    best_move: Optional[Move] = None
    depth_reached = 0
    total_nodes = 0

    for depth in range(1, max_depth + 1):
        if depth < _ASP_MIN_DEPTH:
            score, move, nodes = alphabeta_search(
                board, depth, tt=tt, orderer=orderer,
            )
        else:
            # Aspiration window: search with a narrow window around the
            # previous depth's score. Widen on fail-high/fail-low.
            delta = _ASP_INITIAL_DELTA
            alpha = best_score - delta
            beta = best_score + delta

            while True:
                score, move, nodes_iter = alphabeta_search(
                    board, depth, tt=tt, orderer=orderer,
                    alpha=alpha, beta=beta,
                )
                total_nodes += nodes_iter

                if score <= alpha:
                    # Fail low — widen alpha.
                    alpha = max(score - delta, -INF)
                    delta *= 2
                elif score >= beta:
                    # Fail high — widen beta.
                    beta = min(score + delta, INF)
                    delta *= 2
                else:
                    # Score is inside the window.
                    break

            # Skip the normal total_nodes += below since we accumulated above.
            depth_reached = depth
            best_score = score
            best_move = move
            orderer.age_history()
            if move is None:
                break
            continue

        depth_reached = depth
        best_score = score
        best_move = move
        total_nodes += nodes
        orderer.age_history()
        if move is None:
            break

    return SearchResult(
        score_cp=best_score,
        best_move=best_move,
        depth=depth_reached,
        nodes=total_nodes,
    )
