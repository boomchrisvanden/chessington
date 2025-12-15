from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Optional


class Color(IntEnum):
    WHITE = 0
    BLACK = 1

    def other(self) -> "Color":
        return Color.BLACK if self == Color.WHITE else Color.WHITE


class PieceType(IntEnum):
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6


class CastlingRights(IntFlag):
    NONE = 0
    WK = 1
    WQ = 2
    BK = 4
    BQ = 8


def str_to_square(s: str) -> int:
    """
    'a1' -> 0, 'b1' -> 1, ..., 'h8' -> 63.
    Rank 1 is index 0..7, rank 8 is index 56..63.
    """
    if len(s) != 2:
        raise ValueError(f"invalid square: {s!r}")
    file = ord(s[0]) - ord("a")
    rank = ord(s[1]) - ord("1")
    if not (0 <= file < 8 and 0 <= rank < 8):
        raise ValueError(f"invalid square: {s!r}")
    return rank * 8 + file


def square_to_str(idx: int) -> str:
    if not (0 <= idx < 64):
        raise ValueError(f"square out of range: {idx}")
    file = idx % 8
    rank = idx // 8
    return chr(ord("a") + file) + chr(ord("1") + rank)


def piece_to_promo_letter(pt: Optional[PieceType]) -> str:
    if pt is None:
        return ""
    return {
        PieceType.QUEEN: "q",
        PieceType.ROOK: "r",
        PieceType.BISHOP: "b",
        PieceType.KNIGHT: "n",
    }[pt]


def promo_letter_to_piece(promo: Optional[str]) -> Optional[PieceType]:
    if promo is None:
        return None
    return {
        "q": PieceType.QUEEN,
        "r": PieceType.ROOK,
        "b": PieceType.BISHOP,
        "n": PieceType.KNIGHT,
    }.get(promo.lower())


@dataclass(frozen=True, slots=True)
class Move:
    """UCI move (e2e4, e7e8q)."""

    from_sq: int
    to_sq: int
    promotion: Optional[PieceType] = None

    def uci(self) -> str:
        return (
            square_to_str(self.from_sq)
            + square_to_str(self.to_sq)
            + (piece_to_promo_letter(self.promotion) if self.promotion else "")
        )


def move_from_uci(text: str) -> Optional[Move]:
    text = text.strip()
    if len(text) not in (4, 5):
        return None

    try:
        from_sq = str_to_square(text[0:2])
        to_sq = str_to_square(text[2:4])
    except ValueError:
        return None

    promotion = promo_letter_to_piece(text[4]) if len(text) == 5 else None
    if len(text) == 5 and promotion is None:
        return None

    return Move(from_sq, to_sq, promotion=promotion)

