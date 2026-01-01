import pytest

from src.core.board import Board


def clone_board(board: Board) -> Board:
    cloned = Board()
    cloned.squares = board.squares[:]
    cloned.side_to_move = board.side_to_move
    cloned.castling_rights = board.castling_rights
    cloned.ep_square = board.ep_square
    cloned.halfmove_clock = board.halfmove_clock
    cloned.fullmove_number = board.fullmove_number
    cloned.hash = board.hash
    return cloned


def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for move in board.generate_legal():
        child = clone_board(board)
        child.make_move(move)
        nodes += perft(child, depth - 1)
    return nodes


PERFT_POSITIONS = [
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        {1: 20, 2: 400, 3: 8902},
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        {1: 48, 2: 2039},
    ),
    (
        "en-passant",
        "rnbqkbnr/pppp1ppp/8/4p3/3Pp3/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 2",
        {1: 28, 2: 864},
    ),
]


@pytest.mark.parametrize(
    "name,fen,depth,expected",
    [
        pytest.param(name, fen, depth, count, id=f"{name}-d{depth}")
        for name, fen, depths in PERFT_POSITIONS
        for depth, count in depths.items()
    ],
)
def test_perft_positions(name: str, fen: str, depth: int, expected: int) -> None:
    board = Board.from_fen(fen)
    assert perft(board, depth) == expected
