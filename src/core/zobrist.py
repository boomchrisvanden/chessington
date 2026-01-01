from __future__ import annotations

import random

from src.core.types import Color, PieceType


def piece_index(color: Color, pt: PieceType) -> int:
    return (int(pt) - 1) + (0 if color == Color.WHITE else 6)


class Zobrist:
    """Holds random keys; board XORs them to maintain hash incrementally."""

    def __init__(self, seed: int = 0xC0FFEE) -> None:
        rng = random.Random(seed)
        self.piece_square = [[rng.getrandbits(64) for _ in range(64)] for _ in range(12)]
        self.side = rng.getrandbits(64)
        self.castling = [rng.getrandbits(64) for _ in range(16)]
        self.ep_file = [rng.getrandbits(64) for _ in range(8)]


ZOBRIST = Zobrist()
