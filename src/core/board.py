from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.core.types import (
    CastlingRights,
    Color,
    Move,
    PieceType,
    square_to_str,
    str_to_square,
)
from src.core.zobrist import ZOBRIST, piece_index

Piece = Tuple[Color, PieceType]


@dataclass(slots=True)
class MoveUndo:
    move: Move
    moved_piece: Piece
    captured_piece: Optional[Piece]
    captured_sq: Optional[int]
    castling_rights: CastlingRights
    ep_square: Optional[int]
    halfmove_clock: int
    fullmove_number: int
    side_to_move: Color
    hash: int


class Board:
    def __init__(self) -> None:
        self.squares: List[Optional[Piece]] = [None] * 64
        self.side_to_move: Color = Color.WHITE
        self.castling_rights: CastlingRights = CastlingRights.NONE
        self.ep_square: Optional[int] = None
        self.halfmove_clock: int = 0
        self.fullmove_number: int = 1
        self.hash: int = 0
        self._recompute_hash()

    @staticmethod
    def from_startpos() -> "Board":
        b = Board()
        b._setup_startpos()
        return b

    def reset_to_startpos(self) -> None:
        self._setup_startpos()

    def _setup_startpos(self) -> None:
        self.squares = [None] * 64
        self.ep_square = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.side_to_move = Color.WHITE
        self.castling_rights = (
            CastlingRights.WK | CastlingRights.WQ | CastlingRights.BK | CastlingRights.BQ
        )

        # White pieces
        self.squares[str_to_square("a1")] = (Color.WHITE, PieceType.ROOK)
        self.squares[str_to_square("b1")] = (Color.WHITE, PieceType.KNIGHT)
        self.squares[str_to_square("c1")] = (Color.WHITE, PieceType.BISHOP)
        self.squares[str_to_square("d1")] = (Color.WHITE, PieceType.QUEEN)
        self.squares[str_to_square("e1")] = (Color.WHITE, PieceType.KING)
        self.squares[str_to_square("f1")] = (Color.WHITE, PieceType.BISHOP)
        self.squares[str_to_square("g1")] = (Color.WHITE, PieceType.KNIGHT)
        self.squares[str_to_square("h1")] = (Color.WHITE, PieceType.ROOK)
        for file_char in "abcdefgh":
            self.squares[str_to_square(f"{file_char}2")] = (Color.WHITE, PieceType.PAWN)

        # Black pieces
        self.squares[str_to_square("a8")] = (Color.BLACK, PieceType.ROOK)
        self.squares[str_to_square("b8")] = (Color.BLACK, PieceType.KNIGHT)
        self.squares[str_to_square("c8")] = (Color.BLACK, PieceType.BISHOP)
        self.squares[str_to_square("d8")] = (Color.BLACK, PieceType.QUEEN)
        self.squares[str_to_square("e8")] = (Color.BLACK, PieceType.KING)
        self.squares[str_to_square("f8")] = (Color.BLACK, PieceType.BISHOP)
        self.squares[str_to_square("g8")] = (Color.BLACK, PieceType.KNIGHT)
        self.squares[str_to_square("h8")] = (Color.BLACK, PieceType.ROOK)
        for file_char in "abcdefgh":
            self.squares[str_to_square(f"{file_char}7")] = (Color.BLACK, PieceType.PAWN)

        self._recompute_hash()

    def _toggle_piece(self, color: Color, pt: PieceType, sq: int) -> None:
        self.hash ^= ZOBRIST.piece_square[piece_index(color, pt)][sq]

    def _recompute_hash(self) -> None:
        h = 0
        for sq, piece in enumerate(self.squares):
            if piece is None:
                continue
            color, pt = piece
            h ^= ZOBRIST.piece_square[piece_index(color, pt)][sq]
        h ^= ZOBRIST.castling[int(self.castling_rights)]
        if self.ep_square is not None:
            h ^= ZOBRIST.ep_file[self.ep_square % 8]
        if self.side_to_move == Color.BLACK:
            h ^= ZOBRIST.side
        self.hash = h

    @staticmethod
    def from_fen(fen: str) -> "Board":
        parts = fen.strip().split()
        if len(parts) != 6:
            raise ValueError(f"invalid FEN (expected 6 fields): {fen!r}")

        piece_placement, stm, castling, ep, halfmove, fullmove = parts

        b = Board()
        b.squares = [None] * 64

        ranks = piece_placement.split("/")
        if len(ranks) != 8:
            raise ValueError(f"invalid FEN ranks: {fen!r}")

        piece_map = {
            "p": PieceType.PAWN,
            "n": PieceType.KNIGHT,
            "b": PieceType.BISHOP,
            "r": PieceType.ROOK,
            "q": PieceType.QUEEN,
            "k": PieceType.KING,
        }

        for fen_rank_idx, rank_str in enumerate(ranks):
            rank = 7 - fen_rank_idx  # FEN rank 8..1 -> internal rank 7..0
            file = 0
            for ch in rank_str:
                if ch.isdigit():
                    file += int(ch)
                    continue

                pt = piece_map.get(ch.lower())
                if pt is None:
                    raise ValueError(f"invalid FEN piece: {ch!r}")
                color = Color.WHITE if ch.isupper() else Color.BLACK
                if not (0 <= file < 8):
                    raise ValueError(f"invalid FEN file overflow: {fen!r}")
                b.squares[rank * 8 + file] = (color, pt)
                file += 1

            if file != 8:
                raise ValueError(f"invalid FEN rank width: {fen!r}")

        if stm == "w":
            b.side_to_move = Color.WHITE
        elif stm == "b":
            b.side_to_move = Color.BLACK
        else:
            raise ValueError(f"invalid FEN stm: {stm!r}")

        rights = CastlingRights.NONE
        if castling != "-":
            for ch in castling:
                rights |= {
                    "K": CastlingRights.WK,
                    "Q": CastlingRights.WQ,
                    "k": CastlingRights.BK,
                    "q": CastlingRights.BQ,
                }.get(ch, CastlingRights.NONE)
        b.castling_rights = rights

        b.ep_square = None if ep == "-" else str_to_square(ep)
        b.halfmove_clock = int(halfmove)
        b.fullmove_number = int(fullmove)
        b._recompute_hash()
        return b

    def validate_move(self, move: Move) -> Tuple[bool, str]:
        if not (0 <= move.from_sq < 64 and 0 <= move.to_sq < 64):
            return False, "square out of range"
        if move.from_sq == move.to_sq:
            return False, "from/to are the same square"

        piece = self.squares[move.from_sq]
        if piece is None:
            return False, f"no piece on {square_to_str(move.from_sq)}"

        color, pt = piece
        if color != self.side_to_move:
            return False, "wrong side to move"

        dst_piece = self.squares[move.to_sq]
        if dst_piece is not None and dst_piece[0] == color:
            return False, "destination occupied by own piece"
        if dst_piece is not None and dst_piece[0] != color and dst_piece[1] == PieceType.KING:
            return False, "cannot capture king"

        if pt != PieceType.PAWN and move.promotion is not None:
            return False, "promotion is only for pawns"

        ok, reason = self._validate_piece_move(color, pt, move)
        if not ok:
            return False, reason

        was_in_check = self.in_check(color)
        if self._would_leave_king_in_check(color, move):
            if pt == PieceType.KING:
                return False, "king would move into check"
            if was_in_check:
                return False, "move doesn't resolve check"
            return False, "move exposes king to check"

        return True, ""

    def generate_legal(self) -> List[Move]:
        def add_if_legal(move: Move, color: Color, out: List[Move]) -> None:
            undo = self.make_move(move)
            if undo is None:
                return
            legal = not self.in_check(color)
            self.unmake_move(undo)
            if legal:
                out.append(move)

        color = self.side_to_move
        promotion_rank = 7 if color == Color.WHITE else 0
        start_rank = 1 if color == Color.WHITE else 6
        rank_step = 1 if color == Color.WHITE else -1
        promotion_pieces = (
            PieceType.QUEEN,
            PieceType.ROOK,
            PieceType.BISHOP,
            PieceType.KNIGHT,
        )

        moves: List[Move] = []

        for from_sq, piece in enumerate(self.squares):
            if piece is None:
                continue
            p_color, pt = piece
            if p_color != color:
                continue

            from_rank, from_file = divmod(from_sq, 8)

            if pt == PieceType.PAWN:
                one_step_rank = from_rank + rank_step
                if 0 <= one_step_rank < 8:
                    one_step_sq = from_sq + 8 * rank_step
                    if self.squares[one_step_sq] is None:
                        if one_step_rank == promotion_rank:
                            for promo in promotion_pieces:
                                add_if_legal(Move(from_sq, one_step_sq, promotion=promo), color, moves)
                        else:
                            add_if_legal(Move(from_sq, one_step_sq), color, moves)

                        if from_rank == start_rank:
                            two_step_sq = from_sq + 16 * rank_step
                            if self.squares[two_step_sq] is None:
                                add_if_legal(Move(from_sq, two_step_sq), color, moves)

                for file_step in (-1, 1):
                    to_file = from_file + file_step
                    to_rank = from_rank + rank_step
                    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
                        continue
                    to_sq = to_rank * 8 + to_file
                    target = self.squares[to_sq]
                    if target is not None:
                        if target[0] != color and target[1] != PieceType.KING:
                            if to_rank == promotion_rank:
                                for promo in promotion_pieces:
                                    add_if_legal(
                                        Move(from_sq, to_sq, promotion=promo), color, moves
                                    )
                            else:
                                add_if_legal(Move(from_sq, to_sq), color, moves)
                    elif self.ep_square == to_sq:
                        add_if_legal(Move(from_sq, to_sq), color, moves)
                continue

            if pt == PieceType.KNIGHT:
                for dr, df in (
                    (2, 1),
                    (2, -1),
                    (-2, 1),
                    (-2, -1),
                    (1, 2),
                    (1, -2),
                    (-1, 2),
                    (-1, -2),
                ):
                    r = from_rank + dr
                    f = from_file + df
                    if 0 <= r < 8 and 0 <= f < 8:
                        to_sq = r * 8 + f
                        target = self.squares[to_sq]
                        if target is None or (target[0] != color and target[1] != PieceType.KING):
                            add_if_legal(Move(from_sq, to_sq), color, moves)
                continue

            if pt == PieceType.BISHOP:
                directions = ((1, 1), (1, -1), (-1, 1), (-1, -1))
            elif pt == PieceType.ROOK:
                directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
            elif pt == PieceType.QUEEN:
                directions = (
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                )
            elif pt == PieceType.KING:
                for dr in (-1, 0, 1):
                    for df in (-1, 0, 1):
                        if dr == 0 and df == 0:
                            continue
                        r = from_rank + dr
                        f = from_file + df
                        if 0 <= r < 8 and 0 <= f < 8:
                            to_sq = r * 8 + f
                            target = self.squares[to_sq]
                            if target is None or (
                                target[0] != color and target[1] != PieceType.KING
                            ):
                                add_if_legal(Move(from_sq, to_sq), color, moves)

                if color == Color.WHITE and from_sq == str_to_square("e1"):
                    for to_sq in (str_to_square("g1"), str_to_square("c1")):
                        m = Move(from_sq, to_sq)
                        if self._validate_castling(color, m)[0]:
                            add_if_legal(m, color, moves)
                elif color == Color.BLACK and from_sq == str_to_square("e8"):
                    for to_sq in (str_to_square("g8"), str_to_square("c8")):
                        m = Move(from_sq, to_sq)
                        if self._validate_castling(color, m)[0]:
                            add_if_legal(m, color, moves)
                continue
            else:
                continue

            for dr, df in directions:
                r = from_rank + dr
                f = from_file + df
                while 0 <= r < 8 and 0 <= f < 8:
                    to_sq = r * 8 + f
                    target = self.squares[to_sq]
                    if target is None:
                        add_if_legal(Move(from_sq, to_sq), color, moves)
                    else:
                        if target[0] != color and target[1] != PieceType.KING:
                            add_if_legal(Move(from_sq, to_sq), color, moves)
                        break
                    r += dr
                    f += df

        return moves

    def _validate_piece_move(self, color: Color, pt: PieceType, move: Move) -> Tuple[bool, str]:
        if pt == PieceType.PAWN:
            return self._validate_pawn_move(color, move)
        if pt == PieceType.KNIGHT:
            return self._validate_knight_move(move)
        if pt == PieceType.BISHOP:
            return self._validate_bishop_move(move)
        if pt == PieceType.ROOK:
            return self._validate_rook_move(move)
        if pt == PieceType.QUEEN:
            return self._validate_queen_move(move)
        if pt == PieceType.KING:
            return self._validate_king_move(color, move)
        return False, "unknown piece type"

    def _validate_pawn_move(self, color: Color, move: Move) -> Tuple[bool, str]:
        from_rank, from_file = divmod(move.from_sq, 8)
        to_rank, to_file = divmod(move.to_sq, 8)
        rank_step = 1 if color == Color.WHITE else -1
        start_rank = 1 if color == Color.WHITE else 6
        promotion_rank = 7 if color == Color.WHITE else 0

        rank_diff = to_rank - from_rank
        file_diff = to_file - from_file
        dst_piece = self.squares[move.to_sq]

        if to_rank == promotion_rank:
            if move.promotion is None:
                return False, "promotion required"
            if move.promotion not in (
                PieceType.QUEEN,
                PieceType.ROOK,
                PieceType.BISHOP,
                PieceType.KNIGHT,
            ):
                return False, "invalid promotion piece"
        else:
            if move.promotion is not None:
                return False, "unexpected promotion"

        if file_diff == 0:
            if dst_piece is not None:
                return False, "pawn push is blocked"

            if rank_diff == rank_step:
                return True, ""

            if rank_diff == 2 * rank_step:
                if from_rank != start_rank:
                    return False, "pawn double-push only from start rank"
                between_sq = move.from_sq + (8 * rank_step)
                if not (0 <= between_sq < 64) or self.squares[between_sq] is not None:
                    return False, "pawn double-push is blocked"
                return True, ""

            return False, "illegal pawn push distance"

        if abs(file_diff) == 1 and rank_diff == rank_step:
            if dst_piece is not None:
                if dst_piece[0] == color:
                    return False, "cannot capture own piece"
                return True, ""

            if self.ep_square is None or move.to_sq != self.ep_square:
                return False, "illegal pawn capture"

            captured_sq = move.to_sq - (8 * rank_step)
            if not (0 <= captured_sq < 64):
                return False, "illegal en passant"
            captured_piece = self.squares[captured_sq]
            if captured_piece != (color.other(), PieceType.PAWN):
                return False, "illegal en passant"
            return True, ""

        return False, "illegal pawn move vector"

    def _validate_bishop_move(self, move: Move) -> Tuple[bool, str]:
        from_rank, from_file = divmod(move.from_sq, 8)
        to_rank, to_file = divmod(move.to_sq, 8)

        rank_diff = to_rank - from_rank
        file_diff = to_file - from_file

        if abs(rank_diff) != abs(file_diff):
            return False, "illegal bishop move vector"

        rank_step = 1 if rank_diff > 0 else -1
        file_step = 1 if file_diff > 0 else -1

        distance = abs(rank_diff)
        for i in range(1, distance):
            r = from_rank + i * rank_step
            f = from_file + i * file_step
            sq = r * 8 + f
            if self.squares[sq] is not None:
                return False, "bishop path is blocked"

        return True, ""

    def _validate_knight_move(self, move: Move) -> Tuple[bool, str]:
        from_rank, from_file = divmod(move.from_sq, 8)
        to_rank, to_file = divmod(move.to_sq, 8)

        rank_diff = abs(to_rank - from_rank)
        file_diff = abs(to_file - from_file)

        if (rank_diff, file_diff) not in ((1, 2), (2, 1)):
            return False, "illegal knight move vector"

        return True, ""

    def _validate_rook_move(self, move: Move) -> Tuple[bool, str]:
        from_rank, from_file = divmod(move.from_sq, 8)
        to_rank, to_file = divmod(move.to_sq, 8)

        rank_diff = to_rank - from_rank
        file_diff = to_file - from_file

        if rank_diff != 0 and file_diff != 0:
            return False, "illegal rook move vector"

        if rank_diff == 0:
            file_step = 1 if file_diff > 0 else -1
            for f in range(from_file + file_step, to_file, file_step):
                sq = from_rank * 8 + f
                if self.squares[sq] is not None:
                    return False, "rook path is blocked"
            return True, ""

        rank_step = 1 if rank_diff > 0 else -1
        for r in range(from_rank + rank_step, to_rank, rank_step):
            sq = r * 8 + from_file
            if self.squares[sq] is not None:
                return False, "rook path is blocked"

        return True, ""

    def _validate_queen_move(self, move: Move) -> Tuple[bool, str]:
        from_rank, from_file = divmod(move.from_sq, 8)
        to_rank, to_file = divmod(move.to_sq, 8)

        rank_diff = to_rank - from_rank
        file_diff = to_file - from_file

        if rank_diff == 0 or file_diff == 0:
            rank_step = 0 if rank_diff == 0 else (1 if rank_diff > 0 else -1)
            file_step = 0 if file_diff == 0 else (1 if file_diff > 0 else -1)
            distance = abs(rank_diff) if rank_diff != 0 else abs(file_diff)

            for i in range(1, distance):
                r = from_rank + i * rank_step
                f = from_file + i * file_step
                sq = r * 8 + f
                if self.squares[sq] is not None:
                    return False, "queen path is blocked"
            return True, ""

        if abs(rank_diff) == abs(file_diff):
            rank_step = 1 if rank_diff > 0 else -1
            file_step = 1 if file_diff > 0 else -1
            distance = abs(rank_diff)

            for i in range(1, distance):
                r = from_rank + i * rank_step
                f = from_file + i * file_step
                sq = r * 8 + f
                if self.squares[sq] is not None:
                    return False, "queen path is blocked"
            return True, ""

        return False, "illegal queen move vector"

    def _validate_king_move(self, color: Color, move: Move) -> Tuple[bool, str]:
        from_rank, from_file = divmod(move.from_sq, 8)
        to_rank, to_file = divmod(move.to_sq, 8)

        rank_diff = abs(to_rank - from_rank)
        file_diff = abs(to_file - from_file)

        if max(rank_diff, file_diff) == 1:
            return True, ""

        return self._validate_castling(color, move)

    def _validate_castling(self, color: Color, move: Move) -> Tuple[bool, str]:
        if color == Color.WHITE:
            if move.from_sq != str_to_square("e1"):
                return False, "illegal king move vector"
            if move.to_sq == str_to_square("g1"):
                if not (self.castling_rights & CastlingRights.WK):
                    return False, "castling not available"
                rook_sq = str_to_square("h1")
                between = (str_to_square("f1"), str_to_square("g1"))
                king_path = between
            elif move.to_sq == str_to_square("c1"):
                if not (self.castling_rights & CastlingRights.WQ):
                    return False, "castling not available"
                rook_sq = str_to_square("a1")
                between = (str_to_square("d1"), str_to_square("c1"), str_to_square("b1"))
                king_path = (str_to_square("d1"), str_to_square("c1"))
            else:
                return False, "illegal king move vector"
        else:
            if move.from_sq != str_to_square("e8"):
                return False, "illegal king move vector"
            if move.to_sq == str_to_square("g8"):
                if not (self.castling_rights & CastlingRights.BK):
                    return False, "castling not available"
                rook_sq = str_to_square("h8")
                between = (str_to_square("f8"), str_to_square("g8"))
                king_path = between
            elif move.to_sq == str_to_square("c8"):
                if not (self.castling_rights & CastlingRights.BQ):
                    return False, "castling not available"
                rook_sq = str_to_square("a8")
                between = (str_to_square("d8"), str_to_square("c8"), str_to_square("b8"))
                king_path = (str_to_square("d8"), str_to_square("c8"))
            else:
                return False, "illegal king move vector"

        if self.squares[move.to_sq] is not None:
            return False, "castling destination must be empty"

        enemy = color.other()
        if self.is_square_attacked(move.from_sq, enemy):
            return False, "cannot castle out of check"
        for sq in king_path:
            if self.is_square_attacked(sq, enemy):
                return False, "cannot castle through check"

        rook = self.squares[rook_sq]
        if rook != (color, PieceType.ROOK):
            return False, "castling rook is missing"

        for sq in between:
            if self.squares[sq] is not None:
                return False, "castling path is blocked"

        return True, ""

    def _find_king(self, color: Color) -> Optional[int]:
        for sq, piece in enumerate(self.squares):
            if piece == (color, PieceType.KING):
                return sq
        return None

    def in_check(self, color: Color) -> bool:
        king_sq = self._find_king(color)
        if king_sq is None:
            return False
        return self.is_square_attacked(king_sq, color.other())

    def _would_leave_king_in_check(self, color: Color, move: Move) -> bool:
        undo = self.make_move(move)
        if undo is None:
            return True
        still_in_check = self.in_check(color)
        self.unmake_move(undo)
        return still_in_check

    def is_square_attacked(self, square: int, by_color: Color) -> bool:
        rank, file = divmod(square, 8)

        pawn_from_rank = rank - 1 if by_color == Color.WHITE else rank + 1
        if 0 <= pawn_from_rank < 8:
            for df in (-1, 1):
                f = file + df
                if 0 <= f < 8:
                    if self.squares[pawn_from_rank * 8 + f] == (by_color, PieceType.PAWN):
                        return True

        for dr, df in (
            (2, 1),
            (2, -1),
            (-2, 1),
            (-2, -1),
            (1, 2),
            (1, -2),
            (-1, 2),
            (-1, -2),
        ):
            r = rank + dr
            f = file + df
            if 0 <= r < 8 and 0 <= f < 8:
                if self.squares[r * 8 + f] == (by_color, PieceType.KNIGHT):
                    return True

        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                r = rank + dr
                f = file + df
                if 0 <= r < 8 and 0 <= f < 8:
                    if self.squares[r * 8 + f] == (by_color, PieceType.KING):
                        return True

        diag_dirs = ((1, 1), (1, -1), (-1, 1), (-1, -1))
        ortho_dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for dr, df in diag_dirs:
            r = rank + dr
            f = file + df
            while 0 <= r < 8 and 0 <= f < 8:
                piece = self.squares[r * 8 + f]
                if piece is None:
                    r += dr
                    f += df
                    continue
                if piece[0] == by_color and piece[1] in (PieceType.BISHOP, PieceType.QUEEN):
                    return True
                break

        for dr, df in ortho_dirs:
            r = rank + dr
            f = file + df
            while 0 <= r < 8 and 0 <= f < 8:
                piece = self.squares[r * 8 + f]
                if piece is None:
                    r += dr
                    f += df
                    continue
                if piece[0] == by_color and piece[1] in (PieceType.ROOK, PieceType.QUEEN):
                    return True
                break

        return False

    def make_move(self, move: Move) -> Optional[MoveUndo]:
        piece = self.squares[move.from_sq]
        if piece is None:
            return None

        color, pt = piece
        undo = MoveUndo(
            move=move,
            moved_piece=piece,
            captured_piece=None,
            captured_sq=None,
            castling_rights=self.castling_rights,
            ep_square=self.ep_square,
            halfmove_clock=self.halfmove_clock,
            fullmove_number=self.fullmove_number,
            side_to_move=self.side_to_move,
            hash=self.hash,
        )

        if self.ep_square is not None:
            self.hash ^= ZOBRIST.ep_file[self.ep_square % 8]
        self.hash ^= ZOBRIST.castling[int(self.castling_rights)]

        from_rank, from_file = divmod(move.from_sq, 8)
        to_rank, to_file = divmod(move.to_sq, 8)
        rank_step = 1 if color == Color.WHITE else -1

        self._toggle_piece(color, pt, move.from_sq)

        captured_piece = self.squares[move.to_sq]
        is_en_passant = False
        if pt == PieceType.PAWN and self.ep_square is not None:
            if (
                abs(to_file - from_file) == 1
                and (to_rank - from_rank) == rank_step
                and captured_piece is None
                and move.to_sq == self.ep_square
            ):
                is_en_passant = True

        captured_sq = None
        if is_en_passant:
            captured_sq = move.to_sq - (8 * rank_step)
            if 0 <= captured_sq < 64:
                captured_piece = self.squares[captured_sq]
                if captured_piece is not None:
                    self._toggle_piece(captured_piece[0], captured_piece[1], captured_sq)
                self.squares[captured_sq] = None
        elif captured_piece is not None:
            self._toggle_piece(captured_piece[0], captured_piece[1], move.to_sq)
            captured_sq = move.to_sq

        if pt == PieceType.KING:
            if color == Color.WHITE and move.from_sq == str_to_square("e1"):
                if move.to_sq == str_to_square("g1"):
                    self._move_piece(str_to_square("h1"), str_to_square("f1"))
                elif move.to_sq == str_to_square("c1"):
                    self._move_piece(str_to_square("a1"), str_to_square("d1"))
            elif color == Color.BLACK and move.from_sq == str_to_square("e8"):
                if move.to_sq == str_to_square("g8"):
                    self._move_piece(str_to_square("h8"), str_to_square("f8"))
                elif move.to_sq == str_to_square("c8"):
                    self._move_piece(str_to_square("a8"), str_to_square("d8"))

        self.squares[move.from_sq] = None
        new_pt = move.promotion if (pt == PieceType.PAWN and move.promotion is not None) else pt
        self.squares[move.to_sq] = (color, new_pt)
        self._toggle_piece(color, new_pt, move.to_sq)

        # Update castling rights (move/capture from starting rook squares, king moves).
        if pt == PieceType.KING:
            if color == Color.WHITE:
                self.castling_rights &= ~(CastlingRights.WK | CastlingRights.WQ)
            else:
                self.castling_rights &= ~(CastlingRights.BK | CastlingRights.BQ)
        elif pt == PieceType.ROOK:
            if move.from_sq == str_to_square("h1"):
                self.castling_rights &= ~CastlingRights.WK
            elif move.from_sq == str_to_square("a1"):
                self.castling_rights &= ~CastlingRights.WQ
            elif move.from_sq == str_to_square("h8"):
                self.castling_rights &= ~CastlingRights.BK
            elif move.from_sq == str_to_square("a8"):
                self.castling_rights &= ~CastlingRights.BQ

        if captured_piece == (Color.WHITE, PieceType.ROOK):
            if captured_sq == str_to_square("h1"):
                self.castling_rights &= ~CastlingRights.WK
            elif captured_sq == str_to_square("a1"):
                self.castling_rights &= ~CastlingRights.WQ
        elif captured_piece == (Color.BLACK, PieceType.ROOK):
            if captured_sq == str_to_square("h8"):
                self.castling_rights &= ~CastlingRights.BK
            elif captured_sq == str_to_square("a8"):
                self.castling_rights &= ~CastlingRights.BQ

        is_capture = captured_piece is not None
        if pt == PieceType.PAWN or is_capture:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        self.ep_square = None
        if pt == PieceType.PAWN and from_file == to_file and (to_rank - from_rank) == 2 * rank_step:
            self.ep_square = move.from_sq + (8 * rank_step)

        if color == Color.BLACK:
            self.fullmove_number += 1
        self.side_to_move = color.other()

        self.hash ^= ZOBRIST.castling[int(self.castling_rights)]
        if self.ep_square is not None:
            self.hash ^= ZOBRIST.ep_file[self.ep_square % 8]
        self.hash ^= ZOBRIST.side

        undo.captured_piece = captured_piece
        undo.captured_sq = captured_sq
        return undo

    def unmake_move(self, undo: MoveUndo) -> None:
        move = undo.move
        color, pt = undo.moved_piece

        self.side_to_move = undo.side_to_move
        self.castling_rights = undo.castling_rights
        self.ep_square = undo.ep_square
        self.halfmove_clock = undo.halfmove_clock
        self.fullmove_number = undo.fullmove_number

        self.squares[move.to_sq] = None

        if pt == PieceType.KING:
            if color == Color.WHITE and move.from_sq == str_to_square("e1"):
                if move.to_sq == str_to_square("g1"):
                    self.squares[str_to_square("h1")] = (Color.WHITE, PieceType.ROOK)
                    self.squares[str_to_square("f1")] = None
                elif move.to_sq == str_to_square("c1"):
                    self.squares[str_to_square("a1")] = (Color.WHITE, PieceType.ROOK)
                    self.squares[str_to_square("d1")] = None
            elif color == Color.BLACK and move.from_sq == str_to_square("e8"):
                if move.to_sq == str_to_square("g8"):
                    self.squares[str_to_square("h8")] = (Color.BLACK, PieceType.ROOK)
                    self.squares[str_to_square("f8")] = None
                elif move.to_sq == str_to_square("c8"):
                    self.squares[str_to_square("a8")] = (Color.BLACK, PieceType.ROOK)
                    self.squares[str_to_square("d8")] = None

        self.squares[move.from_sq] = undo.moved_piece

        if undo.captured_piece is not None and undo.captured_sq is not None:
            self.squares[undo.captured_sq] = undo.captured_piece

        self.hash = undo.hash

    def _move_piece(self, src: int, dst: int) -> None:
        piece = self.squares[src]
        if piece is None:
            return
        color, pt = piece
        self._toggle_piece(color, pt, src)
        self._toggle_piece(color, pt, dst)
        self.squares[src] = None
        self.squares[dst] = piece
