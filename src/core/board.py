'''
Author: Chris Vanden Boom
11/14/2025
'''

class Board:
    """
    Mailbox/array board of 64 squares.
    """
    __slots__ = (
        "squares", "stm", "castling", "ep_square",
        "halfmove_clock", "fullmove_number", "_history", "_zobrist", "_hash"
    )

    def __init__(self):
        self.squares: List[Optional[Tuple[Color, PieceType]]] = [None] * 64
        self.stm: Color = Color.WHITE
        self.castling: int = CastlingRights.WK | CastlingRights.WQ | CastlingRights.BK | CastlingRights.BQ
        self.ep_square: Optional[int] = None
        self.halfmove_clock: int = 0
        self.fullmove_number: int = 1
        self._history: List["_Undo"] = []
        self._zobrist: Optional[Zobrist] = None
        self._hash: int = 0

    # --- setup / IO ---

    @staticmethod
    def from_startpos() -> "Board":
        """Create standard initial position."""
        b = Board()
        # TODO: place pieces


        b._init_zobrist()
        b._recompute_hash()
        return b

    @staticmethod
    def from_fen(fen: str) -> "Board":
        """Load position from FEN."""
        b = Board()
        # ...
        b._init_zobrist()
        b._recompute_hash()
        return b

    def to_fen(self) -> str:
        """Serialize to FEN."""
        # ...

    # --- move generation ---

    def generate_pseudo_legal(self) -> List[Move]:
        """All moves that ignore self-check."""
        pass

    def in_check(self, color: Color) -> bool:
        """Is color's king currently in check?"""
        pass

    def generate_legal(self) -> List[Move]:
        """Filter pseudo-legal by self-check."""
        moves = []
        for m in self.generate_pseudo_legal():
            self.make_move(m)
            illegal = self.in_check(self.stm.other())
            self.unmake_move()
            if not illegal:
                moves.append(m)
        return moves

    # --- make/unmake ---

    class _Undo(NamedTuple):
        move: Move
        captured: Optional[Tuple[Color, PieceType]]
        castling: int
        ep_square: Optional[int]
        halfmove_clock: int
        fullmove_number: int
        hash_before: int

    def make_move(self, move: Move) -> None:
        """Apply move, push undo info for unmake."""
        pass

    def unmake_move(self) -> None:
        """Revert last move using history stack."""
        pass