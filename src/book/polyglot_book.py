"""
Polyglot Opening Book Reader

Reads and queries Polyglot .bin format opening books.

Polyglot book file format:
- Entries are sorted by key (ascending)
- Each entry is 16 bytes:
    - bytes 0-7:  key (uint64, big-endian)
    - bytes 8-9:  move (uint16, big-endian)
    - bytes 10-11: weight (uint16, big-endian)
    - bytes 12-15: learn (uint32, big-endian) - usually unused

Move encoding (16 bits):
    - bits 0-5:   to_square (0-63)
    - bits 6-11:  from_square (0-63)
    - bits 12-14: promotion piece (0=none, 1=knight, 2=bishop, 3=rook, 4=queen)
    - bit 15:     unused

Special castling encoding in Polyglot:
    - Castling is encoded as king moving to the rook's square:
        - e1h1 = white kingside castling
        - e1a1 = white queenside castling
        - e8h8 = black kingside castling
        - e8a8 = black queenside castling

Author: Chris Vanden Boom
"""

import struct
import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, BinaryIO
import random

from .polyglot_zobrist import (
    compute_polyglot_hash,
    compute_polyglot_hash_from_board,
    square_to_algebraic,
    algebraic_to_square,
    file_of,
    rank_of,
    square_from_file_rank,
)


# Entry size in bytes
ENTRY_SIZE = 16


@dataclass
class BookMove:
    """
    A move as encoded in a Polyglot book.
    
    Attributes:
        from_sq: Source square (0-63, Polyglot convention)
        to_sq: Destination square (0-63, Polyglot convention)
        promotion: Promotion piece (0=none, 1=knight, 2=bishop, 3=rook, 4=queen)
        raw_move: The original 16-bit encoded move
    """
    from_sq: int
    to_sq: int
    promotion: int
    raw_move: int
    
    def to_uci(self) -> str:
        """
        Convert to UCI notation (e.g., 'e2e4', 'e7e8q').
        
        Note: This handles the Polyglot castling convention where
        castling is encoded as king to rook square, but UCI uses
        king to final square (e.g., e1g1 for white kingside).
        """
        from_str = square_to_algebraic(self.from_sq)
        to_str = square_to_algebraic(self.to_sq)
        
        # Handle castling conversion from Polyglot to UCI
        # Polyglot: king moves to rook square
        # UCI: king moves to final square
        if self._is_castling():
            from_str, to_str = self._castling_to_uci()
        
        promo_str = ""
        if self.promotion > 0:
            promo_str = ["", "n", "b", "r", "q"][self.promotion]
        
        return from_str + to_str + promo_str
    
    def _is_castling(self) -> bool:
        """Check if this move is a castling move in Polyglot encoding."""
        # White castling: e1 to a1 or h1
        if self.from_sq == algebraic_to_square("e1"):
            if self.to_sq in [algebraic_to_square("a1"), algebraic_to_square("h1")]:
                return True
        # Black castling: e8 to a8 or h8
        if self.from_sq == algebraic_to_square("e8"):
            if self.to_sq in [algebraic_to_square("a8"), algebraic_to_square("h8")]:
                return True
        return False
    
    def _castling_to_uci(self) -> Tuple[str, str]:
        """Convert Polyglot castling to UCI notation."""
        from_str = square_to_algebraic(self.from_sq)
        
        # White kingside: e1h1 -> e1g1
        if self.from_sq == algebraic_to_square("e1") and self.to_sq == algebraic_to_square("h1"):
            return "e1", "g1"
        # White queenside: e1a1 -> e1c1
        if self.from_sq == algebraic_to_square("e1") and self.to_sq == algebraic_to_square("a1"):
            return "e1", "c1"
        # Black kingside: e8h8 -> e8g8
        if self.from_sq == algebraic_to_square("e8") and self.to_sq == algebraic_to_square("h8"):
            return "e8", "g8"
        # Black queenside: e8a8 -> e8c8
        if self.from_sq == algebraic_to_square("e8") and self.to_sq == algebraic_to_square("a8"):
            return "e8", "c8"
        
        # Not actually castling, return original
        return from_str, square_to_algebraic(self.to_sq)
    
    @staticmethod
    def from_raw(raw_move: int) -> "BookMove":
        """
        Decode a raw 16-bit Polyglot move.
        
        Encoding:
            to_sq = raw_move & 0x3f
            from_sq = (raw_move >> 6) & 0x3f
            promotion = (raw_move >> 12) & 0x7
        """
        to_sq = raw_move & 0x3f
        from_sq = (raw_move >> 6) & 0x3f
        promotion = (raw_move >> 12) & 0x7
        
        return BookMove(
            from_sq=from_sq,
            to_sq=to_sq,
            promotion=promotion,
            raw_move=raw_move
        )
    
    @staticmethod
    def to_raw(from_sq: int, to_sq: int, promotion: int = 0) -> int:
        """
        Encode a move to raw 16-bit Polyglot format.
        
        Args:
            from_sq: Source square (0-63)
            to_sq: Destination square (0-63)
            promotion: 0=none, 1=knight, 2=bishop, 3=rook, 4=queen
        
        Returns:
            16-bit encoded move
        """
        return (to_sq & 0x3f) | ((from_sq & 0x3f) << 6) | ((promotion & 0x7) << 12)


@dataclass
class BookEntry:
    """
    A complete entry from a Polyglot book.
    
    Attributes:
        key: 64-bit Zobrist hash of the position
        move: Decoded move
        weight: Move weight (higher = more likely to be played)
        learn: Learning data (usually 0)
    """
    key: int
    move: BookMove
    weight: int
    learn: int
    
    @staticmethod
    def from_bytes(data: bytes) -> "BookEntry":
        """Parse a 16-byte book entry."""
        if len(data) != ENTRY_SIZE:
            raise ValueError(f"Expected {ENTRY_SIZE} bytes, got {len(data)}")
        
        # Big-endian format: key(8) + move(2) + weight(2) + learn(4)
        key, raw_move, weight, learn = struct.unpack(">QHHI", data)
        
        return BookEntry(
            key=key,
            move=BookMove.from_raw(raw_move),
            weight=weight,
            learn=learn
        )
    
    def to_bytes(self) -> bytes:
        """Serialize to 16-byte book entry format."""
        return struct.pack(
            ">QHHI",
            self.key,
            self.move.raw_move,
            self.weight,
            self.learn
        )


class PolyglotBook:
    """
    Reader for Polyglot .bin opening book files.
    
    Supports:
    - Binary search lookup by position hash
    - Multiple moves per position with weights
    - Weighted random move selection
    """
    
    def __init__(self, path: Optional[str] = None):
        """
        Initialize the book reader.
        
        Args:
            path: Path to the .bin file. If None, creates an empty book.
        """
        self._entries: List[BookEntry] = []
        self._keys: List[int] = []  # Sorted keys for binary search
        self._path: Optional[Path] = None
        
        if path is not None:
            self.load(path)
    
    def load(self, path: str) -> None:
        """
        Load a Polyglot book from file.
        
        Args:
            path: Path to the .bin file
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        self._path = Path(path)
        
        if not self._path.exists():
            raise FileNotFoundError(f"Book file not found: {path}")
        
        file_size = self._path.stat().st_size
        if file_size % ENTRY_SIZE != 0:
            raise ValueError(
                f"Invalid book file size {file_size}, "
                f"must be multiple of {ENTRY_SIZE}"
            )
        
        num_entries = file_size // ENTRY_SIZE
        self._entries = []
        self._keys = []
        
        with open(self._path, "rb") as f:
            for _ in range(num_entries):
                data = f.read(ENTRY_SIZE)
                if len(data) != ENTRY_SIZE:
                    break
                entry = BookEntry.from_bytes(data)
                self._entries.append(entry)
                self._keys.append(entry.key)
    
    def __len__(self) -> int:
        """Return number of entries in the book."""
        return len(self._entries)
    
    def lookup(self, key: int) -> List[BookEntry]:
        """
        Look up all entries matching a position hash.
        
        Uses binary search for efficient lookup.
        
        Args:
            key: 64-bit Polyglot Zobrist hash
        
        Returns:
            List of BookEntry objects for this position (may be empty)
        """
        if not self._entries:
            return []
        
        # Binary search for first entry with this key
        idx = bisect.bisect_left(self._keys, key)
        
        # Collect all entries with this key
        entries = []
        while idx < len(self._entries) and self._entries[idx].key == key:
            entries.append(self._entries[idx])
            idx += 1
        
        return entries
    
    def lookup_position(
        self,
        squares: List[Optional[Tuple[int, int]]],
        side_to_move: int,
        castling_rights: int,
        ep_square: Optional[int]
    ) -> List[BookEntry]:
        """
        Look up book entries for a position.
        
        Args:
            squares: 64-element board array
            side_to_move: 0=white, 1=black
            castling_rights: Bitmask (1=WK, 2=WQ, 4=BK, 8=BQ)
            ep_square: En passant square or None
        
        Returns:
            List of BookEntry objects for this position
        """
        key = compute_polyglot_hash(squares, side_to_move, castling_rights, ep_square)
        return self.lookup(key)
    
    def lookup_board(self, board) -> List[BookEntry]:
        """
        Look up book entries for a Board object.
        
        Args:
            board: Board object with squares, stm, castling, ep_square
        
        Returns:
            List of BookEntry objects for this position
        """
        key = compute_polyglot_hash_from_board(board)
        return self.lookup(key)
    
    def get_best_move(self, key: int) -> Optional[BookMove]:
        """
        Get the highest-weighted move for a position.
        
        Args:
            key: 64-bit Polyglot Zobrist hash
        
        Returns:
            The move with highest weight, or None if not in book
        """
        entries = self.lookup(key)
        if not entries:
            return None
        
        best = max(entries, key=lambda e: e.weight)
        return best.move
    
    def get_weighted_random_move(
        self,
        key: int,
        rng: Optional[random.Random] = None
    ) -> Optional[BookMove]:
        """
        Get a random move weighted by book weights.
        
        Args:
            key: 64-bit Polyglot Zobrist hash
            rng: Random number generator (uses default if None)
        
        Returns:
            A randomly selected move (weighted), or None if not in book
        """
        entries = self.lookup(key)
        if not entries:
            return None
        
        if rng is None:
            rng = random.Random()
        
        total_weight = sum(e.weight for e in entries)
        if total_weight == 0:
            # All weights are 0, pick uniformly
            return rng.choice(entries).move
        
        # Weighted random selection
        r = rng.randint(0, total_weight - 1)
        cumulative = 0
        for entry in entries:
            cumulative += entry.weight
            if r < cumulative:
                return entry.move
        
        # Fallback (shouldn't happen)
        return entries[-1].move
    
    def get_all_moves(self, key: int) -> List[Tuple[BookMove, int]]:
        """
        Get all moves for a position with their weights.
        
        Args:
            key: 64-bit Polyglot Zobrist hash
        
        Returns:
            List of (move, weight) tuples, sorted by weight descending
        """
        entries = self.lookup(key)
        result = [(e.move, e.weight) for e in entries]
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    def contains(self, key: int) -> bool:
        """Check if a position is in the book."""
        return len(self.lookup(key)) > 0
    
    def get_stats(self) -> dict:
        """Get statistics about the loaded book."""
        if not self._entries:
            return {
                "path": str(self._path) if self._path else None,
                "entries": 0,
                "unique_positions": 0,
            }
        
        unique_keys = len(set(self._keys))
        total_weight = sum(e.weight for e in self._entries)
        
        return {
            "path": str(self._path) if self._path else None,
            "entries": len(self._entries),
            "unique_positions": unique_keys,
            "total_weight": total_weight,
            "avg_moves_per_position": len(self._entries) / unique_keys if unique_keys else 0,
        }


# =============================================================================
# Conversion utilities
# =============================================================================

def uci_to_polyglot_move(uci: str, is_castling: bool = False) -> BookMove:
    """
    Convert UCI move notation to a BookMove.
    
    Note: For castling, the caller should set is_castling=True or this
    function will detect it based on the squares.
    
    Args:
        uci: Move in UCI format (e.g., 'e2e4', 'e7e8q', 'e1g1')
        is_castling: Hint that this is a castling move
    
    Returns:
        BookMove in Polyglot format
    """
    uci = uci.lower().strip()
    
    from_sq = algebraic_to_square(uci[0:2])
    to_sq = algebraic_to_square(uci[2:4])
    
    # Detect and convert castling from UCI to Polyglot format
    # UCI: e1g1 (kingside), e1c1 (queenside)
    # Polyglot: e1h1 (kingside), e1a1 (queenside)
    if is_castling or _is_uci_castling(from_sq, to_sq):
        from_sq, to_sq = _uci_castling_to_polyglot(from_sq, to_sq)
    
    # Handle promotion
    promotion = 0
    if len(uci) == 5:
        promo_map = {'n': 1, 'b': 2, 'r': 3, 'q': 4}
        promotion = promo_map.get(uci[4], 0)
    
    return BookMove(
        from_sq=from_sq,
        to_sq=to_sq,
        promotion=promotion,
        raw_move=BookMove.to_raw(from_sq, to_sq, promotion)
    )


def _is_uci_castling(from_sq: int, to_sq: int) -> bool:
    """Check if move looks like UCI castling notation."""
    e1 = algebraic_to_square("e1")
    e8 = algebraic_to_square("e8")
    
    # White castling
    if from_sq == e1:
        if to_sq == algebraic_to_square("g1"):  # Kingside
            return True
        if to_sq == algebraic_to_square("c1"):  # Queenside
            return True
    
    # Black castling
    if from_sq == e8:
        if to_sq == algebraic_to_square("g8"):  # Kingside
            return True
        if to_sq == algebraic_to_square("c8"):  # Queenside
            return True
    
    return False


def _uci_castling_to_polyglot(from_sq: int, to_sq: int) -> Tuple[int, int]:
    """Convert UCI castling move to Polyglot format."""
    e1 = algebraic_to_square("e1")
    e8 = algebraic_to_square("e8")
    
    # White kingside: e1g1 -> e1h1
    if from_sq == e1 and to_sq == algebraic_to_square("g1"):
        return e1, algebraic_to_square("h1")
    # White queenside: e1c1 -> e1a1
    if from_sq == e1 and to_sq == algebraic_to_square("c1"):
        return e1, algebraic_to_square("a1")
    # Black kingside: e8g8 -> e8h8
    if from_sq == e8 and to_sq == algebraic_to_square("g8"):
        return e8, algebraic_to_square("h8")
    # Black queenside: e8c8 -> e8a8
    if from_sq == e8 and to_sq == algebraic_to_square("c8"):
        return e8, algebraic_to_square("a8")
    
    return from_sq, to_sq
