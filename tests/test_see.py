from src.core.board import Board
from src.core.types import Move, PieceType, str_to_square
from src.search.see import see


def test_undefended_capture():
    """PxR on an undefended rook: gain = 500."""
    board = Board.from_fen("4k3/8/8/3r4/4P3/8/8/4K3 w - - 0 1")
    move = Move(str_to_square("e4"), str_to_square("d5"))
    assert see(board, move) == 500


def test_defended_capture_winning():
    """PxR where rook is defended by pawn, but PxR still trades favourably.
    P(100) captures R(500), then defender P captures our P: 500 - 100 = 400."""
    board = Board.from_fen("4k3/8/2p5/3r4/4P3/8/8/4K3 w - - 0 1")
    move = Move(str_to_square("e4"), str_to_square("d5"))
    assert see(board, move) == 400


def test_queen_captures_defended_pawn():
    """QxP where pawn is defended by pawn: 100 - 900 = -800."""
    board = Board.from_fen("4k3/8/2p5/3p4/4Q3/8/8/4K3 w - - 0 1")
    move = Move(str_to_square("e4"), str_to_square("d5"))
    assert see(board, move) == 100 - 900


def test_rook_takes_defended_bishop():
    """RxB where bishop defended by pawn: 300 - 500 = -200."""
    board = Board.from_fen("4k3/8/2p5/3b4/8/8/8/3RK3 w - - 0 1")
    move = Move(str_to_square("d1"), str_to_square("d5"))
    assert see(board, move) == 300 - 500


def test_equal_trade():
    """NxN: 300 - 300 = 0 if recaptured, but if not defended, gain = 300."""
    board = Board.from_fen("4k3/8/8/3n4/8/8/8/3NK3 w - - 0 1")
    move = Move(str_to_square("d1"), str_to_square("f2"))
    # Knight on d1 can't reach d5 in one move; let's set up properly.
    board2 = Board.from_fen("4k3/8/8/3n4/8/4N3/8/4K3 w - - 0 1")
    move2 = Move(str_to_square("e3"), str_to_square("d5"))
    # Undefended knight: gain = 300.
    assert see(board2, move2) == 300


def test_equal_trade_defended():
    """NxN where target is defended by a pawn: 300 - 300 = 0."""
    board = Board.from_fen("4k3/8/2p5/3n4/8/4N3/8/4K3 w - - 0 1")
    move = Move(str_to_square("e3"), str_to_square("d5"))
    assert see(board, move) == 0


def test_xray_rook_behind_rook():
    """RxR with another rook behind: R(500) takes R(500), defended by R,
    but our second rook recaptures. Net = 0."""
    board = Board.from_fen("3rk3/8/8/8/8/8/8/3RRK2 w - - 0 1")
    # White rooks on d1 and e1, black rook on d8.
    # Rd1xRd8, if black has no defenders: gain = 500.
    # Let's add a black defender.
    board2 = Board.from_fen("2rrk3/8/8/8/8/8/8/2RRK3 w - - 0 1")
    # White Rc1 captures Rc8. Black Rd8 recaptures. White Rd1 x-ray recaptures.
    move = Move(str_to_square("c1"), str_to_square("c8"))
    assert see(board2, move) == 0


def test_promotion_capture():
    """Pawn promotes to queen while capturing: gains piece + promotion bonus."""
    board = Board.from_fen("3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1")
    move = Move(str_to_square("e7"), str_to_square("d8"), promotion=PieceType.QUEEN)
    # Capture rook (500) + promotion bonus (900 - 100 = 800) = 1300.
    # If king recaptures: 1300 - 900 = 400.
    # King can recapture on d8 since e8 king reaches d8.
    assert see(board, move) == 400


def test_en_passant():
    """En passant capture: gain = 100 (pawn value)."""
    board = Board.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    move = Move(str_to_square("e5"), str_to_square("d6"))
    assert see(board, move) == 100


def test_non_capture_returns_zero():
    """Non-capture move should return 0."""
    board = Board.from_startpos()
    move = Move(str_to_square("e2"), str_to_square("e4"))
    assert see(board, move) == 0
