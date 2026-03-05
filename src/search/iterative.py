from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from src.core.board import Board
from src.core.types import Move
from src.search.alphabeta import INF, SearchAborted, search as alphabeta_search
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
    info_callback=None,
) -> SearchResult:
    """
    Iterative deepening over PVS search with aspiration windows.

    *info_callback*, if provided, is called after each completed depth with
    ``(depth, score_cp, nodes, elapsed_ms)`` so the caller (e.g. UCI) can
    emit ``info`` lines.
    """
    if max_depth <= 0:
        return SearchResult(score_cp=0, best_move=None, depth=0)

    orderer = MoveOrderer()
    best_score = 0
    best_move: Optional[Move] = None
    depth_reached = 0
    total_nodes = 0
    start = time.monotonic()

    # Hard stop time: the search must not exceed this deadline.
    stop_time = (start + time_ms / 1000.0) if time_ms > 0 else 0.0

    for depth in range(1, max_depth + 1):
        try:
            if depth < _ASP_MIN_DEPTH:
                score, move, nodes = alphabeta_search(
                    board, depth, tt=tt, orderer=orderer,
                    stop_time=stop_time,
                )
                total_nodes += nodes
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
                        stop_time=stop_time,
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
        except SearchAborted:
            # Time ran out mid-depth; use the best result from previous depth.
            break

        depth_reached = depth
        best_score = score
        best_move = move
        orderer.age_history()

        elapsed_ms = int((time.monotonic() - start) * 1000)
        if info_callback is not None:
            info_callback(depth, best_score, total_nodes, elapsed_ms)

        if move is None:
            break

        # Stop deepening if we've used enough of our allocated time.
        # Use half the budget so we have a safety margin for the next depth.
        if time_ms > 0 and elapsed_ms >= time_ms // 2:
            break

    return SearchResult(
        score_cp=best_score,
        best_move=best_move,
        depth=depth_reached,
        nodes=total_nodes,
    )
