from src.core.board import Board
from src.core.types import Color, Move, PieceType, str_to_square


def _bitboard_from_squares(board: Board, color: Color, pt: PieceType) -> int:
    bb = 0
    for sq, piece in enumerate(board.squares):
        if piece == (color, pt):
            bb |= 1 << sq
    return bb


def _assert_bitboards_match(board: Board) -> None:
    for color in (Color.WHITE, Color.BLACK):
        occ = 0
        for pt in (
            PieceType.PAWN,
            PieceType.KNIGHT,
            PieceType.BISHOP,
            PieceType.ROOK,
            PieceType.QUEEN,
            PieceType.KING,
        ):
            bb = _bitboard_from_squares(board, color, pt)
            assert board.piece_bb[int(color)][int(pt)] == bb
            occ |= bb
        assert board.occ_bb[int(color)] == occ
    assert board.occ_all == (board.occ_bb[int(Color.WHITE)] | board.occ_bb[int(Color.BLACK)])


def test_bitboards_sync_across_moves() -> None:
    board = Board.from_startpos()
    _assert_bitboards_match(board)

    undo1 = board.make_move(Move(str_to_square("e2"), str_to_square("e4")))
    assert undo1 is not None
    _assert_bitboards_match(board)

    undo2 = board.make_move(Move(str_to_square("d7"), str_to_square("d5")))
    assert undo2 is not None
    _assert_bitboards_match(board)

    undo3 = board.make_move(Move(str_to_square("e4"), str_to_square("d5")))
    assert undo3 is not None
    _assert_bitboards_match(board)

    board.unmake_move(undo3)
    _assert_bitboards_match(board)
    board.unmake_move(undo2)
    _assert_bitboards_match(board)
    board.unmake_move(undo1)
    _assert_bitboards_match(board)
