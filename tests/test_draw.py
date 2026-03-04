from src.core.board import Board
from src.core.types import Color, Move, PieceType, str_to_square


def _move(fr: str, to: str) -> Move:
    return Move(str_to_square(fr), str_to_square(to))


class TestThreefoldRepetition:
    def test_threefold_via_knight_shuffle(self):
        """Nf3/Ng1 repeated → 3-fold after 4 round trips."""
        b = Board.from_startpos()
        moves = [
            _move("g1", "f3"), _move("g8", "f6"),
            _move("f3", "g1"), _move("f6", "g8"),  # back to start (2nd)
            _move("g1", "f3"), _move("g8", "f6"),
            _move("f3", "g1"), _move("f6", "g8"),  # back to start (3rd)
        ]
        for m in moves:
            b.make_move(m)
        assert b.is_repetition(3)

    def test_no_threefold_after_two_occurrences(self):
        b = Board.from_startpos()
        moves = [
            _move("g1", "f3"), _move("g8", "f6"),
            _move("f3", "g1"), _move("f6", "g8"),  # back to start (2nd)
        ]
        for m in moves:
            b.make_move(m)
        assert not b.is_repetition(3)

    def test_twofold_detection(self):
        """2-fold used in search."""
        b = Board.from_startpos()
        moves = [
            _move("g1", "f3"), _move("g8", "f6"),
            _move("f3", "g1"), _move("f6", "g8"),  # 2nd occurrence
        ]
        for m in moves:
            b.make_move(m)
        assert b.is_repetition(2)


class TestFiftyMoveRule:
    def test_fifty_move_rule_at_100(self):
        fen = "8/8/4k3/8/8/4K3/8/8 w - - 99 80"
        b = Board.from_fen(fen)
        assert not b.is_fifty_move_rule()
        # One quiet king move → halfmove_clock = 100
        b.make_move(_move("e3", "d3"))
        assert b.is_fifty_move_rule()

    def test_fifty_move_reset_on_capture(self):
        # Place a pawn to capture
        fen = "8/8/4k3/8/3p4/4K3/8/8 w - - 99 80"
        b = Board.from_fen(fen)
        b.make_move(_move("e3", "d4"))  # capture resets clock
        assert not b.is_fifty_move_rule()


class TestInsufficientMaterial:
    def test_k_vs_k(self):
        fen = "8/8/4k3/8/8/4K3/8/8 w - - 0 1"
        b = Board.from_fen(fen)
        assert b.is_insufficient_material()

    def test_k_knight_vs_k(self):
        fen = "8/8/4k3/8/8/4KN2/8/8 w - - 0 1"
        b = Board.from_fen(fen)
        assert b.is_insufficient_material()

    def test_k_bishop_vs_k(self):
        fen = "8/8/4k3/8/8/4KB2/8/8 w - - 0 1"
        b = Board.from_fen(fen)
        assert b.is_insufficient_material()

    def test_k_bishop_vs_k_bishop_same_color(self):
        # Both bishops on light squares
        fen = "8/8/4k1b1/8/8/4KB2/8/8 w - - 0 1"
        b = Board.from_fen(fen)
        # f3 = rank 2, file 5 → (2+5)%2 = 1 (dark)
        # g6 = rank 5, file 6 → (5+6)%2 = 1 (dark)
        assert b.is_insufficient_material()

    def test_k_bishop_vs_k_bishop_different_color(self):
        # Bishops on different colored squares
        fen = "8/8/4kb2/8/8/4KB2/8/8 w - - 0 1"
        b = Board.from_fen(fen)
        # f3 = (2+5)%2 = 1 (dark), f6 = (5+5)%2 = 0 (light)
        assert not b.is_insufficient_material()

    def test_k_rook_vs_k_sufficient(self):
        fen = "8/8/4k3/8/8/4KR2/8/8 w - - 0 1"
        b = Board.from_fen(fen)
        assert not b.is_insufficient_material()

    def test_k_pawn_vs_k_sufficient(self):
        fen = "8/8/4k3/8/4P3/4K3/8/8 w - - 0 1"
        b = Board.from_fen(fen)
        assert not b.is_insufficient_material()


class TestHashHistory:
    def test_hash_history_grows_on_make(self):
        b = Board.from_startpos()
        assert len(b.hash_history) == 1
        b.make_move(_move("e2", "e4"))
        assert len(b.hash_history) == 2

    def test_hash_history_shrinks_on_unmake(self):
        b = Board.from_startpos()
        undo = b.make_move(_move("e2", "e4"))
        assert len(b.hash_history) == 2
        b.unmake_move(undo)
        assert len(b.hash_history) == 1

    def test_hash_history_consistent_through_cycle(self):
        b = Board.from_startpos()
        initial_hash = b.hash_history[0]
        moves = [
            _move("g1", "f3"), _move("g8", "f6"),
            _move("f3", "g1"), _move("f6", "g8"),
        ]
        for m in moves:
            b.make_move(m)
        assert b.hash_history[-1] == initial_hash

    def test_hash_restored_after_unmake(self):
        b = Board.from_startpos()
        initial_hash = b.hash
        undo = b.make_move(_move("e2", "e4"))
        assert b.hash != initial_hash
        b.unmake_move(undo)
        assert b.hash == initial_hash
        assert b.hash_history == [initial_hash]


class TestStalemate:
    def test_stalemate_position(self):
        """Black king in corner, white queen on b6, white king on a6 — stalemate for black."""
        fen = "k7/8/KQ6/8/8/8/8/8 b - - 0 1"
        b = Board.from_fen(fen)
        # Black has no legal moves but is not in check
        legal = b.generate_legal()
        assert len(legal) == 0
        assert not b.in_check(Color.BLACK)
