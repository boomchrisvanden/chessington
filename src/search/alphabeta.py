from __future__ import annotations

import math
import time
from typing import Optional, Tuple

from src.core.board import Board
from src.core.types import Color, Move, PieceType
from src.core.zobrist import ZOBRIST
from src.search.eval import evaluate
from src.search.ordering import MoveOrderer
from src.search.quiescence import quiescence
from src.search.tt import Bound, TTEntry, TranspositionTable

INF = 100_000
MATE_SCORE = 10_000

# Node interval for time checks inside the search tree.
_TIME_CHECK_INTERVAL = 2048


class SearchAborted(Exception):
    """Raised when the search must stop due to time running out."""
    pass

# Null-move pruning constants.
_NMP_REDUCTION = 2
_NMP_DEPTH_MIN = 3  # Only try NMP at depth >= this.

# Late-move reduction constants.
_LMR_DEPTH_MIN = 3
_LMR_MOVES_MIN = 4  # Full-window moves before reductions kick in.

# Reverse futility pruning (static null-move pruning).
_RFP_DEPTH_MAX = 3
_RFP_MARGIN_PER_DEPTH = 200  # Centipawns per depth.

# Futility pruning.
_FP_DEPTH_MAX = 2
_FP_MARGIN = [0, 200, 400]  # Indexed by depth.

# Pre-compute LMR reduction table: _LMR_TABLE[depth][move_index].
_LMR_MAX_DEPTH = 64
_LMR_MAX_MOVES = 64
_LMR_TABLE = [[0] * _LMR_MAX_MOVES for _ in range(_LMR_MAX_DEPTH)]
for _d in range(1, _LMR_MAX_DEPTH):
    for _m in range(1, _LMR_MAX_MOVES):
        _LMR_TABLE[_d][_m] = max(0, int(0.75 + math.log(_d) * math.log(_m) / 2.25))


def _is_capture(board: Board, move: Move) -> bool:
    if board.squares[move.to_sq] is not None:
        return True
    if board.ep_square is not None and move.to_sq == board.ep_square:
        piece = board.squares[move.from_sq]
        if piece is not None and piece[1] == PieceType.PAWN:
            return True
    return False


def _has_non_pawn_material(board: Board, color: Color) -> bool:
    """Check if a side has any non-pawn, non-king material (for NMP guard)."""
    c = int(color)
    return (
        board.piece_bb[c][int(PieceType.KNIGHT)]
        | board.piece_bb[c][int(PieceType.BISHOP)]
        | board.piece_bb[c][int(PieceType.ROOK)]
        | board.piece_bb[c][int(PieceType.QUEEN)]
    ) != 0


def _pvs(
    board: Board,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    tt: Optional[TranspositionTable],
    orderer: MoveOrderer,
    do_null: bool,
    nodes: list[int],
    stop_time: float = 0.0,
) -> int:
    """
    Principal Variation Search with NMP, LMR, and full move ordering.
    """
    nodes[0] += 1

    # Periodic time check.
    if stop_time and nodes[0] % _TIME_CHECK_INTERVAL == 0:
        if time.monotonic() >= stop_time:
            raise SearchAborted

    # Draw detection (skip at root ply).
    if ply > 0:
        if board.halfmove_clock >= 100:
            return 0
        if board.is_repetition(2):
            return 0

    # Quiescence at horizon.
    if depth <= 0:
        return quiescence(board, alpha, beta, ply, tt=tt)

    in_check = board.in_check(board.side_to_move)

    # Check extension: don't reduce when in check.
    if in_check:
        depth += 1

    alpha_orig = alpha
    tt_move: Optional[Move] = None

    # TT probe.
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

    # --- Reverse Futility Pruning (Static Null Move Pruning) ---
    # At shallow depths, if static eval is far above beta, prune.
    if (
        not in_check
        and depth <= _RFP_DEPTH_MAX
        and abs(beta) < MATE_SCORE - 100
    ):
        static_eval = evaluate(board)
        if static_eval - _RFP_MARGIN_PER_DEPTH * depth >= beta:
            return static_eval

    # --- Null Move Pruning ---
    if (
        do_null
        and depth >= _NMP_DEPTH_MIN
        and not in_check
        and _has_non_pawn_material(board, board.side_to_move)
    ):
        # Make null move (pass the turn).
        old_ep = board.ep_square
        old_hash = board.hash
        if board.ep_square is not None:
            board.hash ^= ZOBRIST.ep_file[board.ep_square % 8]
        board.ep_square = None
        board.side_to_move = board.side_to_move.other()
        board.hash ^= ZOBRIST.side

        null_score = -_pvs(
            board, depth - 1 - _NMP_REDUCTION, -beta, -beta + 1,
            ply + 1, tt, orderer, False, nodes, stop_time,
        )

        # Unmake null move.
        board.side_to_move = board.side_to_move.other()
        board.ep_square = old_ep
        board.hash = old_hash

        if null_score >= beta:
            return null_score

    # Generate and order moves.
    moves = board.generate_pseudo_legal()
    if not moves:
        if in_check:
            return -MATE_SCORE + ply
        return 0

    moves = orderer.order_moves(board, moves, ply, tt_move)

    best = -INF
    best_move: Optional[Move] = None
    found_legal = False
    moves_searched = 0

    # Pre-compute futility pruning eligibility.
    can_futility_prune = (
        not in_check
        and depth <= _FP_DEPTH_MAX
        and abs(alpha) < MATE_SCORE - 100
    )
    if can_futility_prune:
        fp_eval = evaluate(board)
        fp_margin = _FP_MARGIN[depth]
    else:
        fp_eval = 0
        fp_margin = 0

    for move in moves:
        undo = board.make_move(move)
        if undo is None:
            continue
        if board.in_check(undo.side_to_move):
            board.unmake_move(undo)
            continue

        found_legal = True
        capture = _is_capture(board, move)
        is_promotion = move.promotion is not None
        gives_check = board.in_check(board.side_to_move)

        # --- Futility Pruning ---
        # At frontier nodes, skip quiet moves that have no hope of raising alpha.
        if (
            can_futility_prune
            and moves_searched > 0
            and not capture
            and not is_promotion
            and not gives_check
            and fp_eval + fp_margin <= alpha
        ):
            board.unmake_move(undo)
            continue

        if moves_searched == 0:
            # First move (PV): full window.
            score = -_pvs(
                board, depth - 1, -beta, -alpha,
                ply + 1, tt, orderer, True, nodes, stop_time,
            )
        else:
            # --- Late Move Reductions ---
            reduction = 0
            if (
                depth >= _LMR_DEPTH_MIN
                and moves_searched >= _LMR_MOVES_MIN
                and not capture
                and not is_promotion
                and not in_check
                and not gives_check
            ):
                d_idx = min(depth, _LMR_MAX_DEPTH - 1)
                m_idx = min(moves_searched, _LMR_MAX_MOVES - 1)
                reduction = _LMR_TABLE[d_idx][m_idx]
                # Don't reduce into quiescence.
                if reduction >= depth - 1:
                    reduction = depth - 2

            # PVS: zero-width window.
            score = -_pvs(
                board, depth - 1 - reduction, -alpha - 1, -alpha,
                ply + 1, tt, orderer, True, nodes, stop_time,
            )

            # Re-search at full depth if LMR failed high.
            if reduction > 0 and score > alpha:
                score = -_pvs(
                    board, depth - 1, -alpha - 1, -alpha,
                    ply + 1, tt, orderer, True, nodes, stop_time,
                )

            # Re-search with full window if PVS failed high.
            if score > alpha and score < beta:
                score = -_pvs(
                    board, depth - 1, -beta, -alpha,
                    ply + 1, tt, orderer, True, nodes, stop_time,
                )

        board.unmake_move(undo)

        if score > best:
            best = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            # Beta cutoff: update killers and history for quiet moves.
            if not capture and not is_promotion:
                orderer.record_killer(ply, move)
                orderer.record_history(undo.side_to_move, move, depth)
            if tt is not None:
                tt.store(TTEntry(board.hash, depth, best, Bound.LOWER, best_move))
            break

        moves_searched += 1

    if not found_legal:
        if in_check:
            return -MATE_SCORE + ply
        return 0

    # TT store.
    if tt is not None:
        bound = Bound.EXACT if best > alpha_orig else Bound.UPPER
        tt.store(TTEntry(board.hash, depth, best, bound, best_move))

    return best


def search(
    board: Board,
    depth: int,
    tt: Optional[TranspositionTable] = None,
    orderer: Optional[MoveOrderer] = None,
    alpha: int = -INF,
    beta: int = INF,
    stop_time: float = 0.0,
) -> Tuple[int, Optional[Move], int]:
    """
    Root search. Returns (score, best_move, nodes).
    Accepts optional alpha/beta for aspiration windows.
    """
    if orderer is None:
        orderer = MoveOrderer()

    if depth <= 0:
        return evaluate(board), None, 1

    nodes = [0]
    tt_move: Optional[Move] = None
    if tt is not None:
        entry = tt.get(board.hash)
        if entry is not None:
            tt_move = entry.best_move

    moves = board.generate_pseudo_legal()
    if not moves:
        if board.in_check(board.side_to_move):
            return -MATE_SCORE, None, 1
        return 0, None, 1

    moves = orderer.order_moves(board, moves, 0, tt_move)

    best_score = -INF
    best_move: Optional[Move] = None
    found_legal = False
    moves_searched = 0

    for move in moves:
        undo = board.make_move(move)
        if undo is None:
            continue
        if board.in_check(undo.side_to_move):
            board.unmake_move(undo)
            continue
        found_legal = True

        if moves_searched == 0:
            score = -_pvs(
                board, depth - 1, -beta, -alpha,
                1, tt, orderer, True, nodes, stop_time,
            )
        else:
            # PVS zero-window.
            score = -_pvs(
                board, depth - 1, -alpha - 1, -alpha,
                1, tt, orderer, True, nodes, stop_time,
            )
            if score > alpha and score < beta:
                score = -_pvs(
                    board, depth - 1, -beta, -alpha,
                    1, tt, orderer, True, nodes, stop_time,
                )

        board.unmake_move(undo)

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score

        moves_searched += 1

    if not found_legal:
        if board.in_check(board.side_to_move):
            return -MATE_SCORE, None, nodes[0]
        return 0, None, nodes[0]

    if tt is not None:
        tt.store(TTEntry(board.hash, depth, best_score, Bound.EXACT, best_move))

    return best_score, best_move, nodes[0]
