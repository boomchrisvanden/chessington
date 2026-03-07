from __future__ import annotations

from typing import List, Optional

from src.core.board import Board
from src.core.types import Color, Move, PieceType
from src.search.see import see

# Piece values for MVV-LVA scoring.
_PIECE_VAL = {
    PieceType.PAWN: 100,
    PieceType.KNIGHT: 300,
    PieceType.BISHOP: 300,
    PieceType.ROOK: 500,
    PieceType.QUEEN: 900,
    PieceType.KING: 0,
}

# Score tiers (higher = searched first).
_TT_MOVE_SCORE = 10_000_000
_GOOD_CAPTURE_BASE = 1_000_000
_KILLER_SCORE = 900_000
# History scores are 0..~800_000 range.
_BAD_CAPTURE_BASE = -1_000_000

_MAX_KILLERS = 2
_MAX_PLY = 128


class MoveOrderer:
    """Maintains killer and history tables across the search."""

    def __init__(self) -> None:
        self.killers: List[List[Optional[Move]]] = [
            [None] * _MAX_KILLERS for _ in range(_MAX_PLY)
        ]
        # history[color][from_sq][to_sq]
        self.history: List[List[List[int]]] = [
            [[0] * 64 for _ in range(64)] for _ in range(2)
        ]

    def clear(self) -> None:
        for ply_killers in self.killers:
            for i in range(len(ply_killers)):
                ply_killers[i] = None
        for c in range(2):
            for f in range(64):
                for t in range(64):
                    self.history[c][f][t] = 0

    def age_history(self) -> None:
        """Halve history scores between iterations to prevent overflow."""
        for c in range(2):
            for f in range(64):
                for t in range(64):
                    self.history[c][f][t] >>= 1

    def record_killer(self, ply: int, move: Move) -> None:
        if ply >= _MAX_PLY:
            return
        ply_killers = self.killers[ply]
        # Don't store duplicates.
        if ply_killers[0] == move:
            return
        # Shift: slot 1 = old slot 0, slot 0 = new killer.
        ply_killers[1] = ply_killers[0]
        ply_killers[0] = move

    def record_history(self, color: Color, move: Move, depth: int) -> None:
        bonus = depth * depth
        val = self.history[int(color)][move.from_sq][move.to_sq]
        # Gravity: approach _KILLER_SCORE asymptotically.
        self.history[int(color)][move.from_sq][move.to_sq] = val + bonus - val * bonus // _KILLER_SCORE

    def score_move(
        self,
        board: Board,
        move: Move,
        ply: int,
        tt_move: Optional[Move],
    ) -> int:
        """Return a sorting key for a move (higher = search first)."""
        # 1. TT move gets highest priority.
        if tt_move is not None and move == tt_move:
            return _TT_MOVE_SCORE

        # 2. Captures: use SEE to separate good from bad.
        victim = board.squares[move.to_sq]
        is_ep = (
            victim is None
            and board.ep_square is not None
            and move.to_sq == board.ep_square
            and board.squares[move.from_sq] is not None
            and board.squares[move.from_sq][1] == PieceType.PAWN
        )
        if victim is not None or is_ep:
            victim_val = (
                _PIECE_VAL.get(victim[1], 0) if victim is not None
                else _PIECE_VAL[PieceType.PAWN]
            )
            attacker = board.squares[move.from_sq]
            attacker_val = _PIECE_VAL.get(attacker[1], 0) if attacker else 0
            # Clearly winning or equal capture: skip SEE.
            if victim_val >= attacker_val:
                return _GOOD_CAPTURE_BASE + victim_val * 10 - attacker_val
            # Potentially losing capture: use SEE.
            see_val = see(board, move)
            if see_val >= 0:
                return _GOOD_CAPTURE_BASE + victim_val * 10 - attacker_val
            # Losing capture: order below quiet moves.
            return _BAD_CAPTURE_BASE + see_val

        # 3. Killer moves.
        if ply < _MAX_PLY:
            if self.killers[ply][0] == move:
                return _KILLER_SCORE
            if self.killers[ply][1] == move:
                return _KILLER_SCORE - 1

        # 4. History heuristic.
        attacker = board.squares[move.from_sq]
        if attacker is not None:
            return self.history[int(attacker[0])][move.from_sq][move.to_sq]

        return 0

    def order_moves(
        self,
        board: Board,
        moves: List[Move],
        ply: int,
        tt_move: Optional[Move],
    ) -> List[Move]:
        """Return moves sorted by score descending."""
        scored = [(self.score_move(board, m, ply, tt_move), m) for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]
