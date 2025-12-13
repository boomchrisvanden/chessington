'''
Author: Chris Vanden Boom
11/14/2025
'''
from dataclasses import dataclass

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

class CastlingRights(IntEnum):
    NONE = 0
    WK = 1
    WQ = 2
    BK = 4
    BQ = 8


@dataclass(frozen=True)
class Move:
    """Minimal move representation for a mailbox/array board."""
    from_sq: int               # 0..63
    to_sq: int                 # 0..63
    promotion: Optional[PieceType] = None
    is_capture: bool = False
    is_en_passant: bool = False
    is_double_push: bool = False
    is_castle: bool = False

    def uci(self) -> str:
        """e2e4, e7e8q, etc."""
        # implement square->long algebraic; if promotion, append letter

        
        