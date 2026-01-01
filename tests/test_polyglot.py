"""
Tests for Polyglot opening book implementation.

Verifies:
1. Zobrist key computation matches Polyglot specification
2. Square indexing is correct (a1=0, b1=1, etc.)
3. Move encoding/decoding works correctly
4. Known test positions produce correct hashes
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from book.polyglot_zobrist import (
    PolyglotZobrist,
    get_polyglot_zobrist,
    compute_polyglot_hash,
    square_from_file_rank,
    file_of,
    rank_of,
    square_to_algebraic,
    algebraic_to_square,
)
from book.polyglot_book import (
    BookMove,
    BookEntry,
    PolyglotBook,
    uci_to_polyglot_move,
)


class TestSquareIndexing:
    """Test that square indexing matches Polyglot convention."""
    
    def test_a1_is_0(self):
        assert algebraic_to_square("a1") == 0
    
    def test_b1_is_1(self):
        assert algebraic_to_square("b1") == 1
    
    def test_h1_is_7(self):
        assert algebraic_to_square("h1") == 7
    
    def test_a2_is_8(self):
        assert algebraic_to_square("a2") == 8
    
    def test_h8_is_63(self):
        assert algebraic_to_square("h8") == 63
    
    def test_e4_is_28(self):
        # e4 = file 4 (e), rank 3 (4th rank, 0-indexed)
        # square = 3 * 8 + 4 = 28
        assert algebraic_to_square("e4") == 28
    
    def test_square_roundtrip(self):
        for sq in range(64):
            alg = square_to_algebraic(sq)
            assert algebraic_to_square(alg) == sq
    
    def test_file_rank_extraction(self):
        # Test e4 (square 28)
        sq = algebraic_to_square("e4")
        assert file_of(sq) == 4  # e-file
        assert rank_of(sq) == 3  # 4th rank (0-indexed)
    
    def test_square_from_file_rank(self):
        assert square_from_file_rank(0, 0) == 0   # a1
        assert square_from_file_rank(7, 7) == 63  # h8
        assert square_from_file_rank(4, 3) == 28  # e4


class TestMoveEncoding:
    """Test Polyglot move encoding/decoding."""
    
    def test_decode_simple_move(self):
        # e2e4: from=12 (e2), to=28 (e4)
        # encoded: to | (from << 6) = 28 | (12 << 6) = 28 | 768 = 796
        raw = 796
        move = BookMove.from_raw(raw)
        assert move.from_sq == 12
        assert move.to_sq == 28
        assert move.promotion == 0
    
    def test_encode_simple_move(self):
        raw = BookMove.to_raw(from_sq=12, to_sq=28, promotion=0)
        assert raw == 796
    
    def test_decode_promotion(self):
        # e7e8q: from=52 (e7), to=60 (e8), promotion=4 (queen)
        # encoded: 60 | (52 << 6) | (4 << 12) = 60 | 3328 | 16384 = 19772
        raw = 60 | (52 << 6) | (4 << 12)
        move = BookMove.from_raw(raw)
        assert move.from_sq == 52
        assert move.to_sq == 60
        assert move.promotion == 4
    
    def test_encode_promotion(self):
        raw = BookMove.to_raw(from_sq=52, to_sq=60, promotion=4)
        move = BookMove.from_raw(raw)
        assert move.from_sq == 52
        assert move.to_sq == 60
        assert move.promotion == 4
    
    def test_uci_conversion_simple(self):
        move = uci_to_polyglot_move("e2e4")
        assert move.from_sq == algebraic_to_square("e2")
        assert move.to_sq == algebraic_to_square("e4")
        assert move.promotion == 0
    
    def test_uci_conversion_promotion(self):
        move = uci_to_polyglot_move("e7e8q")
        assert move.from_sq == algebraic_to_square("e7")
        assert move.to_sq == algebraic_to_square("e8")
        assert move.promotion == 4  # queen
    
    def test_uci_to_polyglot_castling_kingside_white(self):
        # UCI: e1g1 -> Polyglot: e1h1
        move = uci_to_polyglot_move("e1g1")
        assert move.from_sq == algebraic_to_square("e1")
        assert move.to_sq == algebraic_to_square("h1")
    
    def test_uci_to_polyglot_castling_queenside_white(self):
        # UCI: e1c1 -> Polyglot: e1a1
        move = uci_to_polyglot_move("e1c1")
        assert move.from_sq == algebraic_to_square("e1")
        assert move.to_sq == algebraic_to_square("a1")
    
    def test_polyglot_to_uci_castling_kingside_white(self):
        # Polyglot: e1h1 -> UCI: e1g1
        move = BookMove(
            from_sq=algebraic_to_square("e1"),
            to_sq=algebraic_to_square("h1"),
            promotion=0,
            raw_move=0
        )
        assert move.to_uci() == "e1g1"
    
    def test_polyglot_to_uci_castling_queenside_black(self):
        # Polyglot: e8a8 -> UCI: e8c8
        move = BookMove(
            from_sq=algebraic_to_square("e8"),
            to_sq=algebraic_to_square("a8"),
            promotion=0,
            raw_move=0
        )
        assert move.to_uci() == "e8c8"


class TestPolyglotZobrist:
    """Test Polyglot Zobrist key generation."""
    
    def test_zobrist_singleton(self):
        z1 = get_polyglot_zobrist()
        z2 = get_polyglot_zobrist()
        assert z1 is z2
    
    def test_zobrist_deterministic(self):
        z1 = PolyglotZobrist()
        z2 = PolyglotZobrist()
        assert z1.turn_key == z2.turn_key
        assert z1.piece_keys[0][0] == z2.piece_keys[0][0]
    
    def test_piece_keys_count(self):
        z = PolyglotZobrist()
        assert len(z.piece_keys) == 12  # 6 piece types * 2 colors
        for keys in z.piece_keys:
            assert len(keys) == 64
    
    def test_castle_keys_count(self):
        z = PolyglotZobrist()
        assert len(z.castle_keys) == 4
    
    def test_ep_keys_count(self):
        z = PolyglotZobrist()
        assert len(z.ep_keys) == 8


class TestKnownPositionHashes:
    """
    Test hash computation against known Polyglot position hashes.
    
    These are well-known test cases from various Polyglot implementations.
    """
    
    def setup_method(self):
        """Create standard starting position."""
        self.startpos = self._create_startpos()
    
    def _create_startpos(self):
        """Create the standard chess starting position."""
        squares = [None] * 64
        
        # White pieces (color=0)
        squares[algebraic_to_square("a1")] = (0, 4)  # Rook
        squares[algebraic_to_square("b1")] = (0, 2)  # Knight
        squares[algebraic_to_square("c1")] = (0, 3)  # Bishop
        squares[algebraic_to_square("d1")] = (0, 5)  # Queen
        squares[algebraic_to_square("e1")] = (0, 6)  # King
        squares[algebraic_to_square("f1")] = (0, 3)  # Bishop
        squares[algebraic_to_square("g1")] = (0, 2)  # Knight
        squares[algebraic_to_square("h1")] = (0, 4)  # Rook
        for f in range(8):
            squares[square_from_file_rank(f, 1)] = (0, 1)  # Pawns
        
        # Black pieces (color=1)
        squares[algebraic_to_square("a8")] = (1, 4)  # Rook
        squares[algebraic_to_square("b8")] = (1, 2)  # Knight
        squares[algebraic_to_square("c8")] = (1, 3)  # Bishop
        squares[algebraic_to_square("d8")] = (1, 5)  # Queen
        squares[algebraic_to_square("e8")] = (1, 6)  # King
        squares[algebraic_to_square("f8")] = (1, 3)  # Bishop
        squares[algebraic_to_square("g8")] = (1, 2)  # Knight
        squares[algebraic_to_square("h8")] = (1, 4)  # Rook
        for f in range(8):
            squares[square_from_file_rank(f, 6)] = (1, 1)  # Pawns
        
        return squares
    
    def test_starting_position_hash(self):
        """
        Test the starting position hash.
        
        The correct Polyglot hash for the starting position is:
        0x463b96181691fc9c
        """
        key = compute_polyglot_hash(
            squares=self.startpos,
            side_to_move=0,  # White
            castling_rights=0xF,  # All castling rights
            ep_square=None
        )
        
        # Known correct hash for starting position
        expected = 0x463b96181691fc9c
        assert key == expected, f"Expected {expected:#018x}, got {key:#018x}"
    
    def test_position_after_e4(self):
        """
        Test position after 1. e4
        
        Expected hash: 0x823c9b50fd114196
        """
        squares = self.startpos.copy()
        
        # Move e2 pawn to e4
        squares[algebraic_to_square("e2")] = None
        squares[algebraic_to_square("e4")] = (0, 1)  # White pawn
        
        key = compute_polyglot_hash(
            squares=squares,
            side_to_move=1,  # Black to move
            castling_rights=0xF,
            ep_square=algebraic_to_square("e3")  # EP square
        )
        
        expected = 0x823c9b50fd114196
        assert key == expected, f"Expected {expected:#018x}, got {key:#018x}"
    
    def test_position_after_d4(self):
        """
        Test position after 1. d4
        
        EP square is d3, but no black pawn can capture, so EP key not included.
        Expected hash: 0x830eb9b20758d1de
        """
        squares = self.startpos.copy()
        
        # Move d2 pawn to d4
        squares[algebraic_to_square("d2")] = None
        squares[algebraic_to_square("d4")] = (0, 1)  # White pawn
        
        key = compute_polyglot_hash(
            squares=squares,
            side_to_move=1,  # Black to move
            castling_rights=0xF,
            ep_square=algebraic_to_square("d3")  # EP square (but no capture possible)
        )
        
        expected = 0x830eb9b20758d1de
        assert key == expected, f"Expected {expected:#018x}, got {key:#018x}"
    
    def test_position_after_e4_d5(self):
        """
        Test position after 1. e4 d5 (Scandinavian)
        
        EP square is d6, but no white pawn can capture en passant.
        (The e4 pawn is on rank 4, not rank 5 where it would need to be)
        Expected hash: 0x0756b94461c50fb0 (EP key NOT included)
        """
        squares = self.startpos.copy()
        
        # 1. e4
        squares[algebraic_to_square("e2")] = None
        squares[algebraic_to_square("e4")] = (0, 1)
        
        # 1... d5
        squares[algebraic_to_square("d7")] = None
        squares[algebraic_to_square("d5")] = (1, 1)
        
        key = compute_polyglot_hash(
            squares=squares,
            side_to_move=0,  # White to move
            castling_rights=0xF,
            ep_square=algebraic_to_square("d6")  # EP square (but no capture possible)
        )
        
        # Note: EP key is only included if a pawn can actually capture
        # In this position, the e4 pawn is NOT adjacent to d5 (different ranks)
        expected = 0x0756b94461c50fb0
        assert key == expected, f"Expected {expected:#018x}, got {key:#018x}"
    
    def test_position_after_e4_e5(self):
        """
        Test position after 1. e4 e5
        
        EP square is e6, but no white pawn can capture there
        (the e4 pawn is on rank 4, not rank 5).
        Expected hash: 0x0844931a6ef4b9a0 (EP key NOT included)
        """
        squares = self.startpos.copy()
        
        # 1. e4
        squares[algebraic_to_square("e2")] = None
        squares[algebraic_to_square("e4")] = (0, 1)
        
        # 1... e5
        squares[algebraic_to_square("e7")] = None
        squares[algebraic_to_square("e5")] = (1, 1)
        
        # No EP possible (no white pawn adjacent to e5 on the 5th rank)
        key = compute_polyglot_hash(
            squares=squares,
            side_to_move=0,  # White to move
            castling_rights=0xF,
            ep_square=algebraic_to_square("e6")  # EP set but no capture possible
        )
        
        # The EP key should NOT be included since no pawn can capture
        expected = 0x0844931a6ef4b9a0
        assert key == expected, f"Expected {expected:#018x}, got {key:#018x}"


    def test_position_with_actual_ep_capture(self):
        """
        Test position where en passant IS actually possible.
        
        Position after 1. e4 Nf6 2. e5 d5
        White's e5 pawn CAN capture the d5 pawn en passant on d6.
        
        Expected hash: 0x2158459ff499f8e3
        """
        squares = self._create_startpos()
        
        # 1. e4
        squares[algebraic_to_square("e2")] = None
        squares[algebraic_to_square("e4")] = (0, 1)
        
        # 1... Nf6
        squares[algebraic_to_square("g8")] = None
        squares[algebraic_to_square("f6")] = (1, 2)  # Black knight
        
        # 2. e5
        squares[algebraic_to_square("e4")] = None
        squares[algebraic_to_square("e5")] = (0, 1)  # White pawn
        
        # 2... d5
        squares[algebraic_to_square("d7")] = None
        squares[algebraic_to_square("d5")] = (1, 1)  # Black pawn
        
        key = compute_polyglot_hash(
            squares=squares,
            side_to_move=0,  # White to move
            castling_rights=0xF,
            ep_square=algebraic_to_square("d6")  # EP square - white CAN capture
        )
        
        # The e5 pawn IS on rank 5, adjacent to d5, so EP is possible
        expected = 0x2158459ff499f8e3
        assert key == expected, f"Expected {expected:#018x}, got {key:#018x}"


class TestBookEntry:
    """Test book entry parsing."""
    
    def test_entry_from_bytes(self):
        import struct
        
        # Create a test entry
        key = 0x463b96181691fc9c
        raw_move = 796  # e2e4
        weight = 100
        learn = 0
        
        data = struct.pack(">QHHI", key, raw_move, weight, learn)
        entry = BookEntry.from_bytes(data)
        
        assert entry.key == key
        assert entry.move.from_sq == 12  # e2
        assert entry.move.to_sq == 28    # e4
        assert entry.weight == 100
        assert entry.learn == 0
    
    def test_entry_roundtrip(self):
        import struct
        
        key = 0x123456789abcdef0
        raw_move = BookMove.to_raw(12, 28, 0)
        weight = 50
        learn = 10
        
        data = struct.pack(">QHHI", key, raw_move, weight, learn)
        entry = BookEntry.from_bytes(data)
        
        serialized = entry.to_bytes()
        assert serialized == data


class TestPolyglotBook:
    """Test the book reader (requires a test book file)."""
    
    def test_empty_book(self):
        book = PolyglotBook()
        assert len(book) == 0
        assert book.lookup(0x463b96181691fc9c) == []
    
    def test_book_stats_empty(self):
        book = PolyglotBook()
        stats = book.get_stats()
        assert stats["entries"] == 0
        assert stats["unique_positions"] == 0
    
    def test_load_test_book(self):
        """Test loading the test book file."""
        import os
        book_path = os.path.join(os.path.dirname(__file__), 'test_book.bin')
        if not os.path.exists(book_path):
            pytest.skip("Test book file not found")
        
        book = PolyglotBook(book_path)
        assert len(book) == 4
    
    def test_lookup_starting_position(self):
        """Test looking up moves for the starting position."""
        import os
        book_path = os.path.join(os.path.dirname(__file__), 'test_book.bin')
        if not os.path.exists(book_path):
            pytest.skip("Test book file not found")
        
        book = PolyglotBook(book_path)
        
        # Starting position key
        start_key = 0x463b96181691fc9c
        entries = book.lookup(start_key)
        
        assert len(entries) == 4
        
        # Check all expected moves are present
        moves_uci = [e.move.to_uci() for e in entries]
        assert "e2e4" in moves_uci
        assert "d2d4" in moves_uci
        assert "c2c4" in moves_uci
        assert "g1f3" in moves_uci
    
    def test_get_best_move(self):
        """Test getting the highest-weighted move."""
        import os
        book_path = os.path.join(os.path.dirname(__file__), 'test_book.bin')
        if not os.path.exists(book_path):
            pytest.skip("Test book file not found")
        
        book = PolyglotBook(book_path)
        start_key = 0x463b96181691fc9c
        
        best = book.get_best_move(start_key)
        assert best is not None
        assert best.to_uci() == "e2e4"  # Highest weight (100)
    
    def test_get_all_moves_sorted(self):
        """Test getting all moves sorted by weight."""
        import os
        book_path = os.path.join(os.path.dirname(__file__), 'test_book.bin')
        if not os.path.exists(book_path):
            pytest.skip("Test book file not found")
        
        book = PolyglotBook(book_path)
        start_key = 0x463b96181691fc9c
        
        moves = book.get_all_moves(start_key)
        assert len(moves) == 4
        
        # Should be sorted by weight descending
        weights = [w for _, w in moves]
        assert weights == [100, 80, 50, 40]
        
        # Check order
        ucis = [m.to_uci() for m, _ in moves]
        assert ucis == ["e2e4", "d2d4", "c2c4", "g1f3"]
    
    def test_contains(self):
        """Test checking if a position is in the book."""
        import os
        book_path = os.path.join(os.path.dirname(__file__), 'test_book.bin')
        if not os.path.exists(book_path):
            pytest.skip("Test book file not found")
        
        book = PolyglotBook(book_path)
        
        # Starting position should be in book
        assert book.contains(0x463b96181691fc9c)
        
        # Random position should not be in book
        assert not book.contains(0x0000000000000001)
    
    def test_lookup_position_with_board(self):
        """Test looking up moves using position data."""
        import os
        book_path = os.path.join(os.path.dirname(__file__), 'test_book.bin')
        if not os.path.exists(book_path):
            pytest.skip("Test book file not found")
        
        book = PolyglotBook(book_path)
        
        # Create starting position
        squares = [None] * 64
        
        # White pieces
        squares[algebraic_to_square("a1")] = (0, 4)  # Rook
        squares[algebraic_to_square("b1")] = (0, 2)  # Knight
        squares[algebraic_to_square("c1")] = (0, 3)  # Bishop
        squares[algebraic_to_square("d1")] = (0, 5)  # Queen
        squares[algebraic_to_square("e1")] = (0, 6)  # King
        squares[algebraic_to_square("f1")] = (0, 3)  # Bishop
        squares[algebraic_to_square("g1")] = (0, 2)  # Knight
        squares[algebraic_to_square("h1")] = (0, 4)  # Rook
        for f in range(8):
            squares[square_from_file_rank(f, 1)] = (0, 1)  # Pawns
        
        # Black pieces
        squares[algebraic_to_square("a8")] = (1, 4)  # Rook
        squares[algebraic_to_square("b8")] = (1, 2)  # Knight
        squares[algebraic_to_square("c8")] = (1, 3)  # Bishop
        squares[algebraic_to_square("d8")] = (1, 5)  # Queen
        squares[algebraic_to_square("e8")] = (1, 6)  # King
        squares[algebraic_to_square("f8")] = (1, 3)  # Bishop
        squares[algebraic_to_square("g8")] = (1, 2)  # Knight
        squares[algebraic_to_square("h8")] = (1, 4)  # Rook
        for f in range(8):
            squares[square_from_file_rank(f, 6)] = (1, 1)  # Pawns
        
        entries = book.lookup_position(
            squares=squares,
            side_to_move=0,  # White
            castling_rights=0xF,
            ep_square=None
        )
        
        assert len(entries) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
