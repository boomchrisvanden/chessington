import os
import sys
import pygame
import shutil
import subprocess
import threading
import queue
from pathlib import Path
import re

# --------------------------------------------------------------------
# Import your engine core here
# --------------------------------------------------------------------
# Adjust paths/imports to match your project structure
# from chess_engine.core.board import Board
# from chess_engine.core.types import Color, PieceType, Move

# For now I’ll define tiny stand-ins; replace with your real ones.

from enum import IntEnum, auto, Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable

# Import theory practice module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from theory.practice import DifficultyLevel
from theory.gui import run_theory_practice

class Color(IntEnum):
    WHITE = 0
    BLACK = 1

    def other(self) -> "Color":
        return Color.BLACK if self == Color.WHITE else Color.WHITE

class PieceType(IntEnum):
    PAWN   = auto()
    KNIGHT = auto()
    BISHOP = auto()
    ROOK   = auto()
    QUEEN  = auto()
    KING   = auto()

@dataclass(frozen=True)
class Move:
    from_sq: int
    to_sq: int
    promotion: Optional[PieceType] = None

    def uci(self) -> str:
        return square_to_str(self.from_sq) + square_to_str(self.to_sq) + (
            piece_to_promo_letter(self.promotion) if self.promotion else ""
        )

def piece_to_promo_letter(pt: Optional[PieceType]) -> str:
    if pt is None:
        return ""
    return {
        PieceType.QUEEN:  "q",
        PieceType.ROOK:   "r",
        PieceType.BISHOP: "b",
        PieceType.KNIGHT: "n",
    }[pt]

class Board:
    """
    Dummy board just for wiring.
    Replace with your real Board (from_startpos, squares, generate_legal, make_move).
    """
    def __init__(self):
        # 8x8, index = rank*8 + file
        self.squares: List[Optional[Tuple[Color, PieceType]]] = [None] * 64
        self.side_to_move = Color.WHITE
        self.ep_square: Optional[int] = None  # en-passant capture destination square
        self._setup_startpos()

    @staticmethod
    def from_startpos() -> "Board":
        return Board()

    def _setup_startpos(self):
        self.squares = [None] * 64
        self.ep_square = None

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

        self.side_to_move = Color.WHITE

    def generate_legal(self) -> List[Move]:
        return []

    def reset_to_startpos(self) -> None:
        self._setup_startpos()

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

        # Incremental: other pieces are still unchecked (but basic turn/own-capture rules apply above).
        return True, ""

    def _validate_pawn_move(self, color: Color, move: Move) -> Tuple[bool, str]:
        from_rank, from_file = divmod(move.from_sq, 8)
        to_rank, to_file = divmod(move.to_sq, 8)
        rank_step = 1 if color == Color.WHITE else -1
        start_rank = 1 if color == Color.WHITE else 6
        promotion_rank = 7 if color == Color.WHITE else 0

        rank_diff = to_rank - from_rank
        file_diff = to_file - from_file
        dst_piece = self.squares[move.to_sq]

        # Promotion must only occur on last rank, and is mandatory there.
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

        # Single / double pushes.
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

        # Captures (including en passant).
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

        if rank_diff == 0 and file_diff == 0:
            return False, "illegal rook move vector"

        rank_step = 0 if rank_diff == 0 else (1 if rank_diff > 0 else -1)
        file_step = 0 if file_diff == 0 else (1 if file_diff > 0 else -1)
        distance = abs(rank_diff) if rank_diff != 0 else abs(file_diff)

        for i in range(1, distance):
            r = from_rank + i * rank_step
            f = from_file + i * file_step
            sq = r * 8 + f
            if self.squares[sq] is not None:
                return False, "rook path is blocked"

        return True, ""

    def _validate_queen_move(self, move: Move) -> Tuple[bool, str]:
        from_rank, from_file = divmod(move.from_sq, 8)
        to_rank, to_file = divmod(move.to_sq, 8)

        rank_diff = to_rank - from_rank
        file_diff = to_file - from_file

        if rank_diff == 0 and file_diff == 0:
            return False, "illegal queen move vector"

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
                rook_sq = str_to_square("h1")
                between = (str_to_square("f1"), str_to_square("g1"))
                king_path = between
            elif move.to_sq == str_to_square("c1"):
                rook_sq = str_to_square("a1")
                between = (str_to_square("d1"), str_to_square("c1"), str_to_square("b1"))
                king_path = (str_to_square("d1"), str_to_square("c1"))
            else:
                return False, "illegal king move vector"

        else:
            if move.from_sq != str_to_square("e8"):
                return False, "illegal king move vector"
            if move.to_sq == str_to_square("g8"):
                rook_sq = str_to_square("h8")
                between = (str_to_square("f8"), str_to_square("g8"))
                king_path = between
            elif move.to_sq == str_to_square("c8"):
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

    def has_any_legal_move(self) -> bool:
        promotion_rank = 7 if self.side_to_move == Color.WHITE else 0
        promotion_pieces = (
            PieceType.QUEEN,
            PieceType.ROOK,
            PieceType.BISHOP,
            PieceType.KNIGHT,
        )

        for from_sq, piece in enumerate(self.squares):
            if piece is None:
                continue
            color, pt = piece
            if color != self.side_to_move:
                continue

            if pt == PieceType.PAWN:
                for to_sq in range(64):
                    if to_sq == from_sq:
                        continue
                    to_rank = to_sq // 8
                    if to_rank == promotion_rank:
                        for promo in promotion_pieces:
                            if self.validate_move(Move(from_sq, to_sq, promotion=promo))[0]:
                                return True
                    else:
                        if self.validate_move(Move(from_sq, to_sq))[0]:
                            return True
                continue

            for to_sq in range(64):
                if to_sq == from_sq:
                    continue
                if self.validate_move(Move(from_sq, to_sq))[0]:
                    return True

        return False

    def is_checkmate(self) -> bool:
        return self.in_check(self.side_to_move) and not self.has_any_legal_move()

    def _would_leave_king_in_check(self, color: Color, move: Move) -> bool:
        squares_before = self.squares[:]
        ep_before = self.ep_square
        stm_before = self.side_to_move

        self.make_move(move)
        still_in_check = self.in_check(color)

        self.squares[:] = squares_before
        self.ep_square = ep_before
        self.side_to_move = stm_before

        return still_in_check

    def is_square_attacked(self, square: int, by_color: Color) -> bool:
        rank, file = divmod(square, 8)

        # Pawn attacks
        pawn_from_rank = rank - 1 if by_color == Color.WHITE else rank + 1
        if 0 <= pawn_from_rank < 8:
            for df in (-1, 1):
                f = file + df
                if 0 <= f < 8:
                    if self.squares[pawn_from_rank * 8 + f] == (by_color, PieceType.PAWN):
                        return True

        # Knight attacks
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

        # King attacks (adjacent squares)
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                r = rank + dr
                f = file + df
                if 0 <= r < 8 and 0 <= f < 8:
                    if self.squares[r * 8 + f] == (by_color, PieceType.KING):
                        return True

        # Sliding attacks
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

    def make_move(self, move: Move) -> None:
        piece = self.squares[move.from_sq]
        if piece is None:
            return

        color, pt = piece
        from_rank, from_file = divmod(move.from_sq, 8)
        to_rank, to_file = divmod(move.to_sq, 8)
        rank_step = 1 if color == Color.WHITE else -1

        # Handle castling (UI-only; no legality checks)
        if pt == PieceType.KING:
            if color == Color.WHITE and move.from_sq == str_to_square("e1"):
                if move.to_sq == str_to_square("g1"):  # O-O
                    self._move_piece(str_to_square("h1"), str_to_square("f1"))
                elif move.to_sq == str_to_square("c1"):  # O-O-O
                    self._move_piece(str_to_square("a1"), str_to_square("d1"))
            elif color == Color.BLACK and move.from_sq == str_to_square("e8"):
                if move.to_sq == str_to_square("g8"):  # O-O
                    self._move_piece(str_to_square("h8"), str_to_square("f8"))
                elif move.to_sq == str_to_square("c8"):  # O-O-O
                    self._move_piece(str_to_square("a8"), str_to_square("d8"))

        is_en_passant = False
        if pt == PieceType.PAWN and self.ep_square is not None:
            if (
                abs(to_file - from_file) == 1
                and (to_rank - from_rank) == rank_step
                and self.squares[move.to_sq] is None
                and move.to_sq == self.ep_square
            ):
                is_en_passant = True

        if is_en_passant:
            captured_sq = move.to_sq - (8 * rank_step)
            if 0 <= captured_sq < 64:
                self.squares[captured_sq] = None

        # Move the piece (captures are implicit)
        self.squares[move.from_sq] = None
        self.squares[move.to_sq] = (color, pt)

        # Handle promotion (if provided)
        if pt == PieceType.PAWN and move.promotion is not None:
            self.squares[move.to_sq] = (color, move.promotion)

        # En-passant target square is only set after a pawn double-push, and expires immediately.
        self.ep_square = None
        if pt == PieceType.PAWN and from_file == to_file and (to_rank - from_rank) == 2 * rank_step:
            self.ep_square = move.from_sq + (8 * rank_step)

        self.side_to_move = Color.BLACK if self.side_to_move == Color.WHITE else Color.WHITE

    def _move_piece(self, src: int, dst: int) -> None:
        piece = self.squares[src]
        if piece is None:
            return
        self.squares[src] = None
        self.squares[dst] = piece


def _convert_svg_to_png(svg_path: Path, png_path: Path, size_px: int) -> bool:
    """
    Best-effort SVG->PNG conversion.
    Returns True if png_path exists after the call.
    """
    try:
        import cairosvg  # type: ignore

        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            output_width=size_px,
            output_height=size_px,
        )
        return png_path.exists()
    except Exception:
        pass

    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        subprocess.run(
            [rsvg, "-w", str(size_px), "-h", str(size_px), "-o", str(png_path), str(svg_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return png_path.exists()

    inkscape = shutil.which("inkscape")
    if inkscape:
        subprocess.run(
            [
                inkscape,
                str(svg_path),
                "--export-type=png",
                f"--export-filename={png_path}",
                "-w",
                str(size_px),
                "-h",
                str(size_px),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return png_path.exists()

    return False

# --------------------------------------------------------------------
# Utility: square <-> string
# --------------------------------------------------------------------

def str_to_square(s: str) -> int:
    """
    'a1' -> 0, 'b1' -> 1, ..., 'h8' -> 63.
    Rank 1 is row 0 (bottom), rank 8 is row 7 (top).
    """
    file = ord(s[0]) - ord('a')
    rank = int(s[1]) - 1
    return rank * 8 + file

def square_to_str(idx: int) -> str:
    file = idx % 8
    rank = idx // 8
    return chr(ord('a') + file) + str(rank + 1)

def parse_uci_move(text: str) -> Optional[Tuple[int, int, Optional[str]]]:
    """
    e2e4 -> (from_idx, to_idx, None)
    e7e8q -> (from_idx, to_idx, 'q')
    Returns None if format invalid.
    """
    text = text.strip()
    if len(text) not in (4, 5):
        return None
    src = text[0:2]
    dst = text[2:4]
    for sq in (src, dst):
        if len(sq) != 2:
            return None
        if sq[0] < 'a' or sq[0] > 'h':
            return None
        if sq[1] < '1' or sq[1] > '8':
            return None
    promo = text[4].lower() if len(text) == 5 else None
    if promo is not None and promo not in "qrbn":
        return None
    return str_to_square(src), str_to_square(dst), promo

def promo_letter_to_piece(promo: Optional[str]) -> Optional[PieceType]:
    if promo is None:
        return None
    mapping = {
        "q": PieceType.QUEEN,
        "r": PieceType.ROOK,
        "b": PieceType.BISHOP,
        "n": PieceType.KNIGHT,
    }
    return mapping.get(promo.lower())

# --------------------------------------------------------------------
# Chess GUI
# --------------------------------------------------------------------

class ChessGUI:
    def __init__(self, board: Board, piece_dir: str):
        pygame.init()
        pygame.display.set_caption("UCI Test GUI")
        self.square_size = 80
        self.board_size = self.square_size * 8
        self.info_height = 100
        self.width = self.board_size
        self.height = self.board_size + self.info_height

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 24)
        self.small_font = pygame.font.SysFont("consolas", 18)
        self.piece_font = pygame.font.SysFont("consolas", 46, bold=True)

        self.board = board
        self.input_text = ""
        self.status_message = "Enter move in UCI format (e2e4, g8f6, e7e8q) or 'reset' / 'exit'"
        self.last_move: Optional[Move] = None
        self.running = True
        self.game_over = False
        self.dragging = False
        self.drag_from: Optional[int] = None
        self.drag_piece: Optional[Tuple[Color, PieceType]] = None
        self.drag_pos = (0, 0)

        self.play_vs_engine = False
        self.engine_depth = 8
        self.engine_side = Color.WHITE
        self.move_history: List[str] = []
        self.engine_proc: Optional[subprocess.Popen] = None
        self.engine_queue = queue.Queue()
        self.engine_thinking = False
        self.engine_epoch = 0
        self.engine_button_rect = pygame.Rect(self.width - 220, self.board_size + 8, 210, 32)
        self.menu_button_rect = pygame.Rect(10, self.board_size + 8, 80, 32)
        
        # Flag to indicate returning to main menu
        self.return_to_menu = False

        self.light_color = (240, 217, 181)
        self.dark_color = (181, 136, 99)
        self.highlight_color = (186, 202, 68)

        # Load piece images
        self.pieces = self.load_piece_images(piece_dir)
        self.fallback_pieces = self.build_fallback_piece_surfaces()

    def load_piece_images(self, piece_dir: str):
        pieces = {}
        piece_dir_path = Path(piece_dir)

        png_by_name = {}
        if piece_dir_path.exists():
            for p in piece_dir_path.iterdir():
                if p.is_file() and p.suffix.lower() == ".png":
                    png_by_name[p.name.lower()] = p

        piece_names = {
            "P": "pawn",
            "N": "knight",
            "B": "bishop",
            "R": "rook",
            "Q": "queen",
            "K": "king",
        }

        for color, ccode in ((Color.WHITE, 'w'), (Color.BLACK, 'b')):
            for pt, pcode in (
                (PieceType.PAWN,   'P'),
                (PieceType.KNIGHT, 'N'),
                (PieceType.BISHOP, 'B'),
                (PieceType.ROOK,   'R'),
                (PieceType.QUEEN,  'Q'),
                (PieceType.KING,   'K'),
            ):
                is_white = color == Color.WHITE
                piece_name = piece_names[pcode]

                human_color = "white" if is_white else "black"
                wiki_color = "l" if is_white else "d"  # light/dark (Wikipedia pieces)
                wiki_re = re.compile(rf"^chess_{pcode.lower()}{wiki_color}t\d+\.png$")

                candidate_names = [
                    f"{ccode}{pcode}.png",
                    f"{ccode}{pcode.lower()}.png",
                    f"{ccode}_{piece_name}.png",
                    f"{ccode}-{piece_name}.png",
                    f"{human_color}_{piece_name}.png",
                    f"{human_color}-{piece_name}.png",
                    f"{piece_name}_{human_color}.png",
                    f"{piece_name}-{human_color}.png",
                ]

                png_path = next((png_by_name.get(name.lower()) for name in candidate_names if name.lower() in png_by_name), None)

                if png_path is None:
                    wiki_match = next((p for n, p in png_by_name.items() if wiki_re.match(n)), None)
                    png_path = wiki_match

                if png_path is None:
                    # If only SVGs are present, try to convert on the fly.
                    svg_candidates = [
                        piece_dir_path / f"{ccode}{pcode}.svg",
                        piece_dir_path / f"{ccode}{pcode.lower()}.svg",
                    ]
                    svg_path = next((p for p in svg_candidates if p.exists()), None)
                    if svg_path is not None:
                        out_png = piece_dir_path / f"{ccode}{pcode}.png"
                        if _convert_svg_to_png(svg_path, out_png, self.square_size):
                            png_path = out_png

                if png_path is None or not png_path.exists():
                    continue

                img = pygame.image.load(str(png_path)).convert_alpha()
                img = pygame.transform.smoothscale(img, (self.square_size, self.square_size))
                pieces[(color, pt)] = img
        return pieces

    def build_fallback_piece_surfaces(self):
        fallback = {}
        for color in (Color.WHITE, Color.BLACK):
            for pt, letter in (
                (PieceType.PAWN, "P"),
                (PieceType.KNIGHT, "N"),
                (PieceType.BISHOP, "B"),
                (PieceType.ROOK, "R"),
                (PieceType.QUEEN, "Q"),
                (PieceType.KING, "K"),
            ):
                fg = (245, 245, 245) if color == Color.WHITE else (20, 20, 20)
                outline = (20, 20, 20) if color == Color.WHITE else (245, 245, 245)
                base = self.piece_font.render(letter, True, fg)
                shadow = self.piece_font.render(letter, True, outline)

                surf = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                rect = base.get_rect(center=(self.square_size // 2, self.square_size // 2))
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    surf.blit(shadow, rect.move(dx, dy))
                surf.blit(base, rect)
                fallback[(color, pt)] = surf
        return fallback

    def run(self) -> bool:
        """
        Main game loop.
        
        Returns:
            True if user wants to return to main menu, False otherwise
        """
        while self.running:
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_down(event)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_up(event)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_mouse_motion(event)
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event)

            self.poll_engine()
            self.draw()
            pygame.display.flip()

        self.shutdown_engine()
        return self.return_to_menu

    def _reset_board_state(self) -> None:
        self.board.reset_to_startpos()
        self.last_move = None
        self.game_over = False
        self.input_text = ""
        self.move_history = []

    def start_engine_game(self) -> None:
        if self.engine_thinking:
            self.status_message = "Engine is busy. Wait for it to finish."
            return
        self.play_vs_engine = True
        self.engine_side = Color.WHITE
        self.engine_epoch += 1
        self._reset_board_state()
        if not self._ensure_engine():
            self.play_vs_engine = False
            self.status_message = "Engine failed to start."
            return
        self._engine_send("ucinewgame")
        self.status_message = f"Engine game started (depth {self.engine_depth})."
        if self.board.side_to_move == self.engine_side:
            self._request_engine_move()

    def _ensure_engine(self) -> bool:
        if self.engine_proc is not None and self.engine_proc.poll() is None:
            return True
        engine_path = Path(__file__).resolve().parent / "cli.py"
        try:
            self.engine_proc = subprocess.Popen(
                [sys.executable, "-u", str(engine_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError:
            self.engine_proc = None
            return False
        if not self._engine_handshake():
            self.shutdown_engine()
            return False
        return True

    def _engine_handshake(self) -> bool:
        if not self._engine_send("uci"):
            return False
        if not self._engine_read_until("uciok"):
            return False
        if not self._engine_send("isready"):
            return False
        if not self._engine_read_until("readyok"):
            return False
        return True

    def _engine_send(self, cmd: str) -> bool:
        if self.engine_proc is None or self.engine_proc.stdin is None:
            return False
        try:
            self.engine_proc.stdin.write(cmd + "\n")
            self.engine_proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return False
        return True

    def _engine_read_line(self) -> Optional[str]:
        if self.engine_proc is None or self.engine_proc.stdout is None:
            return None
        line = self.engine_proc.stdout.readline()
        if not line:
            return None
        return line.strip()

    def _engine_read_until(self, token: str) -> bool:
        while True:
            line = self._engine_read_line()
            if line is None:
                return False
            if line == token or line.startswith(token):
                return True

    def _uci_position_command(self) -> str:
        if not self.move_history:
            return "position startpos"
        return "position startpos moves " + " ".join(self.move_history)

    def _request_engine_move(self) -> None:
        if self.engine_thinking or self.game_over:
            return
        if not self._ensure_engine():
            self.status_message = "Engine not available."
            return
        self.engine_thinking = True
        self.status_message = f"Engine thinking (depth {self.engine_depth})..."
        epoch = self.engine_epoch
        threading.Thread(target=self._engine_search, args=(epoch,), daemon=True).start()

    def _engine_search(self, epoch: int) -> None:
        if not self._ensure_engine():
            self.engine_queue.put((epoch, None, "Engine not available."))
            return
        if not self._engine_send(self._uci_position_command()):
            self.engine_queue.put((epoch, None, "Engine command failed."))
            return
        if not self._engine_send(f"go depth {self.engine_depth}"):
            self.engine_queue.put((epoch, None, "Engine command failed."))
            return
        bestmove: Optional[str] = None
        while True:
            line = self._engine_read_line()
            if line is None:
                break
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    bestmove = parts[1]
                break
        if bestmove in (None, "0000", "(none)"):
            self.engine_queue.put((epoch, None, "Engine returned no move."))
        else:
            self.engine_queue.put((epoch, bestmove, None))

    def poll_engine(self) -> None:
        while True:
            try:
                epoch, bestmove, error = self.engine_queue.get_nowait()
            except queue.Empty:
                break
            self.engine_thinking = False
            if epoch != self.engine_epoch:
                continue
            if error is not None:
                self.status_message = error
                continue
            if bestmove is None:
                self.status_message = "Engine returned no move."
                continue
            self.apply_engine_move(bestmove)

    def shutdown_engine(self) -> None:
        if self.engine_proc is None:
            return
        try:
            if self.engine_proc.stdin is not None:
                self.engine_proc.stdin.write("quit\n")
                self.engine_proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.engine_proc.terminate()
        except OSError:
            pass
        self.engine_proc = None

    def square_from_pos(self, pos: Tuple[int, int]) -> Optional[int]:
        x, y = pos
        if x < 0 or y < 0 or x >= self.board_size or y >= self.board_size:
            return None
        file = x // self.square_size
        rank = y // self.square_size
        return (7 - rank) * 8 + file

    def handle_mouse_down(self, event):
        if event.button != 1:
            return
        # Menu button - return to main menu
        if self.menu_button_rect.collidepoint(event.pos):
            self.return_to_menu = True
            self.running = False
            return
        if self.engine_button_rect.collidepoint(event.pos):
            self.start_engine_game()
            return
        if self.dragging:
            return
        square = self.square_from_pos(event.pos)
        if square is None:
            return
        piece = self.board.squares[square]
        if piece is None or piece[0] != self.board.side_to_move:
            return
        self.dragging = True
        self.drag_from = square
        self.drag_piece = piece
        self.drag_pos = event.pos

    def handle_mouse_motion(self, event):
        if not self.dragging:
            return
        self.drag_pos = event.pos

    def handle_mouse_up(self, event):
        if event.button != 1 or not self.dragging:
            return
        from_sq = self.drag_from
        piece = self.drag_piece
        drop_sq = self.square_from_pos(event.pos)
        self.dragging = False
        self.drag_from = None
        self.drag_piece = None
        if from_sq is None or piece is None or drop_sq is None:
            return
        if drop_sq == from_sq:
            return

        move_text = square_to_str(from_sq) + square_to_str(drop_sq)
        if piece[1] == PieceType.PAWN:
            promotion_rank = 7 if piece[0] == Color.WHITE else 0
            if drop_sq // 8 == promotion_rank:
                move_text += "q"
        self.input_text = move_text
        self.handle_move_input()

    def handle_key(self, event):
        if event.key == pygame.K_RETURN:
            self.handle_move_input()
        elif event.key == pygame.K_BACKSPACE:
            self.input_text = self.input_text[:-1]
        elif event.key == pygame.K_ESCAPE:
            # ESC returns to main menu
            self.return_to_menu = True
            self.running = False
        elif event.key == pygame.K_m:
            # 'm' returns to main menu
            self.return_to_menu = True
            self.running = False
        else:
            ch = event.unicode
            if ch and ch.isprintable():
                self.input_text += ch

    def handle_move_input(self):
        text = self.input_text.strip()
        self.input_text = ""
        self.apply_move_text(text)

    def apply_move_text(self, text: str) -> None:
        if text.lower() == "exit":
            self.running = False
            return
        if text.lower() == "reset":
            self.engine_epoch += 1
            self._reset_board_state()
            self.status_message = "Reset to start position."
            if self.play_vs_engine:
                if self.engine_thinking:
                    self.status_message = "Reset to start position (engine busy)."
                    return
                if self._ensure_engine():
                    self._engine_send("ucinewgame")
                    if self.board.side_to_move == self.engine_side:
                        self._request_engine_move()
                else:
                    self.status_message = "Reset to start position (engine unavailable)."
            return
        if self.game_over:
            self.status_message = "Game over. Type 'reset' to restart."
            return
        if self.play_vs_engine and self.engine_thinking and self.board.side_to_move == self.engine_side:
            self.status_message = f"Engine thinking (depth {self.engine_depth})..."
            return

        parsed = parse_uci_move(text)
        if parsed is None:
            self.status_message = f"Invalid syntax: '{text}'"
            return

        src, dst, promo_letter = parsed
        promo_piece = promo_letter_to_piece(promo_letter)
        move = Move(src, dst, promotion=promo_piece)
        ok, reason = self.board.validate_move(move)
        if not ok:
            self.status_message = f"Invalid move: {reason}"
            return

        self.apply_move(move, "Played")
        if self.play_vs_engine and not self.game_over and self.board.side_to_move == self.engine_side:
            self._request_engine_move()

    def apply_move(self, move: Move, label: str) -> None:
        self.board.make_move(move)
        self.last_move = move
        self.move_history.append(move.uci())
        self.status_message = f"{label}: {move.uci()}"
        if self.board.is_checkmate():
            winner = "White" if self.board.side_to_move == Color.BLACK else "Black"
            self.status_message = f"Checkmate! {winner} wins. Type 'reset' to restart."
            self.game_over = True
            return
        if self.board.in_check(self.board.side_to_move):
            self.status_message += " (check)"

    def apply_engine_move(self, move_text: str) -> None:
        parsed = parse_uci_move(move_text)
        if parsed is None:
            self.status_message = f"Engine returned invalid move: {move_text}"
            return
        src, dst, promo_letter = parsed
        promo_piece = promo_letter_to_piece(promo_letter)
        move = Move(src, dst, promotion=promo_piece)
        ok, reason = self.board.validate_move(move)
        if not ok:
            self.status_message = f"Engine move invalid: {reason}"
            return
        self.apply_move(move, "Engine")

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.draw_board()
        self.draw_info_panel()

    def draw_board(self):
        # Ranks 8..1 top to bottom, files a..h left to right
        for rank in range(8):
            for file in range(8):
                square_index = (7 - rank) * 8 + file  # internal rank0 is '1'
                is_light = (rank + file) % 2 == 0
                color = self.light_color if is_light else self.dark_color

                # Highlight last move squares if any
                if self.last_move is not None and square_index in (self.last_move.from_sq, self.last_move.to_sq):
                    color = self.highlight_color

                x = file * self.square_size
                y = rank * self.square_size
                pygame.draw.rect(self.screen, color, (x, y, self.square_size, self.square_size))

                piece = self.board.squares[square_index]
                if piece is not None:
                    if self.dragging and self.drag_from == square_index:
                        continue
                    img = self.pieces.get(piece)
                    if img is None:
                        img = self.fallback_pieces.get(piece)
                    if img is not None:
                        self.screen.blit(img, (x, y))

        # Optional rank/file labels
        for file in range(8):
            label = self.small_font.render(chr(ord('a') + file), True, (0, 0, 0))
            x = file * self.square_size + 5
            y = self.board_size - 20
            self.screen.blit(label, (x, y))

        for rank in range(8):
            label = self.small_font.render(str(rank + 1), True, (0, 0, 0))
            x = 5
            y = (7 - rank) * self.square_size + 5
            self.screen.blit(label, (x, y))

        if self.dragging and self.drag_piece is not None:
            img = self.pieces.get(self.drag_piece)
            if img is None:
                img = self.fallback_pieces.get(self.drag_piece)
            if img is not None:
                x = self.drag_pos[0] - self.square_size // 2
                y = self.drag_pos[1] - self.square_size // 2
                self.screen.blit(img, (x, y))

    def draw_info_panel(self):
        panel_y = self.board_size
        pygame.draw.rect(
            self.screen,
            (30, 30, 30),
            (0, panel_y, self.width, self.info_height),
        )

        # Menu button
        menu_color = (100, 100, 140)
        pygame.draw.rect(self.screen, menu_color, self.menu_button_rect, border_radius=4)
        menu_label = self.small_font.render("Menu", True, (255, 255, 255))
        menu_label_rect = menu_label.get_rect(center=self.menu_button_rect.center)
        self.screen.blit(menu_label, menu_label_rect)

        # Current side to move
        stm_text = "White to move" if self.board.side_to_move == Color.WHITE else "Black to move"
        stm_surf = self.font.render(stm_text, True, (220, 220, 220))
        self.screen.blit(stm_surf, (100, panel_y + 10))

        if self.engine_thinking:
            button_color = (80, 80, 80)
            button_text = "Engine Thinking..."
        elif self.play_vs_engine:
            button_color = (70, 140, 90)
            button_text = f"Restart Engine (d{self.engine_depth})"
        else:
            button_color = (60, 120, 180)
            button_text = f"Play Engine (d{self.engine_depth})"
        pygame.draw.rect(self.screen, button_color, self.engine_button_rect, border_radius=4)
        label_surf = self.small_font.render(button_text, True, (255, 255, 255))
        label_rect = label_surf.get_rect(center=self.engine_button_rect.center)
        self.screen.blit(label_surf, label_rect)

        # Input box
        input_label = self.small_font.render("Move (UCI):", True, (200, 200, 200))
        self.screen.blit(input_label, (100, panel_y + 50))

        # Draw input text
        display_text = self.input_text if self.input_text else ""
        input_surf = self.font.render(display_text, True, (255, 255, 255))
        self.screen.blit(input_surf, (220, panel_y + 45))

        # Status message
        status_surf = self.small_font.render(self.status_message, True, (200, 200, 0))
        self.screen.blit(status_surf, (10, panel_y + 80))


# --------------------------------------------------------------------
# Menu System
# --------------------------------------------------------------------

class MenuOption:
    """Represents a clickable menu option."""
    def __init__(self, text: str, rect: pygame.Rect, action: Optional[Callable] = None):
        self.text = text
        self.rect = rect
        self.action = action
        self.hover = False


class MenuScreen:
    """Base class for menu screens."""
    
    def __init__(self, width: int = 640, height: int = 480, title: str = "Menu"):
        self.width = width
        self.height = height
        self.title = title
        self.screen = pygame.display.get_surface()
        if self.screen is None or self.screen.get_size() != (width, height):
            self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 28)
        self.title_font = pygame.font.SysFont("consolas", 42, bold=True)
        self.small_font = pygame.font.SysFont("consolas", 18)
        self.options: List[MenuOption] = []
        self.running = True
        self.result: Optional[str] = None
        
        # Colors
        self.bg_color = (30, 40, 50)
        self.button_color = (60, 100, 140)
        self.button_hover_color = (80, 130, 180)
        self.text_color = (255, 255, 255)
        self.title_color = (240, 220, 180)
    
    def add_option(self, text: str, y: int, action: Optional[Callable] = None) -> MenuOption:
        """Add a menu option button."""
        button_width = 300
        button_height = 50
        x = (self.width - button_width) // 2
        rect = pygame.Rect(x, y, button_width, button_height)
        option = MenuOption(text, rect, action)
        self.options.append(option)
        return option
    
    def handle_events(self) -> None:
        """Process pygame events."""
        mouse_pos = pygame.mouse.get_pos()
        
        for option in self.options:
            option.hover = option.rect.collidepoint(mouse_pos)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.result = "quit"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for option in self.options:
                        if option.rect.collidepoint(event.pos):
                            if option.action:
                                option.action()
                            else:
                                self.result = option.text
                                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    self.result = "back"
    
    def draw(self) -> None:
        """Draw the menu screen."""
        self.screen.fill(self.bg_color)
        
        # Draw title
        title_surf = self.title_font.render(self.title, True, self.title_color)
        title_rect = title_surf.get_rect(center=(self.width // 2, 60))
        self.screen.blit(title_surf, title_rect)
        
        # Draw options
        for option in self.options:
            color = self.button_hover_color if option.hover else self.button_color
            pygame.draw.rect(self.screen, color, option.rect, border_radius=8)
            pygame.draw.rect(self.screen, (100, 150, 200), option.rect, width=2, border_radius=8)
            
            text_surf = self.font.render(option.text, True, self.text_color)
            text_rect = text_surf.get_rect(center=option.rect.center)
            self.screen.blit(text_surf, text_rect)
    
    def run(self) -> Optional[str]:
        """Run the menu loop and return the selected option."""
        while self.running:
            self.clock.tick(60)
            self.handle_events()
            self.draw()
            pygame.display.flip()
        return self.result


class MainMenu(MenuScreen):
    """Main menu with game mode selection."""
    
    def __init__(self):
        super().__init__(width=640, height=480, title="Chessington")
        self.add_option("Local Game", 150)
        self.add_option("Play Against Engine", 220)
        self.add_option("Theory Practice", 290)
        self.add_option("Exit", 380)


class ColorSelectionMenu(MenuScreen):
    """Menu for selecting which color to play as."""
    
    def __init__(self, title: str = "Select Your Color"):
        super().__init__(width=640, height=400, title=title)
        self.add_option("Play as White", 150)
        self.add_option("Play as Black", 220)
        self.add_option("Back", 310)


class DifficultyMenu(MenuScreen):
    """Menu for selecting difficulty level in theory practice."""
    
    def __init__(self):
        super().__init__(width=640, height=520, title="Select Difficulty")
        self.add_option("Infinite (∞ chances)", 130)
        self.add_option("Easy (10 chances)", 195)
        self.add_option("Medium (5 chances)", 260)
        self.add_option("Hard (3 chances)", 325)
        self.add_option("Insane (1 chance)", 390)
        self.add_option("Back", 470)


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def get_piece_dir() -> str:
    """Get the directory containing piece images."""
    base_dir = Path(__file__).resolve().parent
    assets_dir = base_dir / "assets"
    pieces_dir = assets_dir / "pieces"

    def has_piece_images(d: Path) -> bool:
        if not d.exists():
            return False

        for p in d.iterdir():
            if not p.is_file() or p.suffix.lower() != ".png":
                continue
            name = p.name.lower()
            if re.match(r"^[wb][pnbrqk]\.png$", name):
                return True
            if re.match(r"^chess_[pnbrqk][ld]t\d+\.png$", name):
                return True
            if re.match(r"^(white|black)[_-](pawn|knight|bishop|rook|queen|king)\.png$", name):
                return True
            if re.match(r"^(pawn|knight|bishop|rook|queen|king)[_-](white|black)\.png$", name):
                return True
            if re.match(r"^[wb][_-](pawn|knight|bishop|rook|queen|king)\.png$", name):
                return True
        return False

    if has_piece_images(assets_dir):
        return str(assets_dir)
    elif has_piece_images(pieces_dir):
        return str(pieces_dir)
    else:
        return str(assets_dir)


def get_book_path() -> str:
    """Get the path to the opening book."""
    base_dir = Path(__file__).resolve().parent
    return str(base_dir / "src" / "Book.bin")


def run_local_game(piece_dir: str) -> bool:
    """
    Run a local two-player game.
    
    Returns:
        True if user wants to return to main menu, False otherwise
    """
    board = Board.from_startpos()
    gui = ChessGUI(board, piece_dir)
    gui.play_vs_engine = False
    return gui.run()


def run_engine_game(piece_dir: str, player_color: Color) -> bool:
    """
    Run a game against the engine.
    
    Returns:
        True if user wants to return to main menu, False otherwise
    """
    board = Board.from_startpos()
    gui = ChessGUI(board, piece_dir)
    gui.play_vs_engine = True
    gui.engine_side = player_color.other()  # Engine plays the opposite color
    gui.engine_epoch += 1
    gui._reset_board_state()
    
    if gui._ensure_engine():
        gui._engine_send("ucinewgame")
        gui.status_message = f"Engine game started (depth {gui.engine_depth}). You play {'White' if player_color == Color.WHITE else 'Black'}."
        if board.side_to_move == gui.engine_side:
            gui._request_engine_move()
    else:
        gui.status_message = "Engine failed to start."
    
    return gui.run()


def run_theory_game(piece_dir: str, book_path: str, difficulty: DifficultyLevel, player_color: int) -> bool:
    """
    Run a theory practice game.
    
    Returns:
        True if user wants to return to main menu, False to quit entirely
    """
    return run_theory_practice(
        book_path=book_path,
        piece_dir=piece_dir,
        difficulty=difficulty,
        player_color=player_color
    )


def main():
    pygame.init()
    
    piece_dir = get_piece_dir()
    book_path = get_book_path()
    
    while True:
        # Show main menu
        main_menu = MainMenu()
        choice = main_menu.run()
        
        if choice in ("Exit", "quit", None):
            break
        
        if choice == "Local Game":
            return_to_menu = run_local_game(piece_dir)
            if not return_to_menu:
                break  # User quit entirely
        
        elif choice == "Play Against Engine":
            # Show color selection
            color_menu = ColorSelectionMenu(title="Play Against Engine")
            color_choice = color_menu.run()
            
            if color_choice == "Play as White":
                return_to_menu = run_engine_game(piece_dir, Color.WHITE)
                if not return_to_menu:
                    break
            elif color_choice == "Play as Black":
                return_to_menu = run_engine_game(piece_dir, Color.BLACK)
                if not return_to_menu:
                    break
            # "Back" or quit returns to main menu
        
        elif choice == "Theory Practice":
            # Show color selection first
            color_menu = ColorSelectionMenu(title="Theory Practice - Color")
            color_choice = color_menu.run()
            
            if color_choice in ("Back", "back", "quit", None):
                continue
            
            player_color = 0 if color_choice == "Play as White" else 1
            
            # Show difficulty selection
            diff_menu = DifficultyMenu()
            diff_choice = diff_menu.run()
            
            if diff_choice in ("Back", "back", "quit", None):
                continue
            
            # Map choice to difficulty level
            difficulty_map = {
                "Infinite (∞ chances)": DifficultyLevel.INFINITE,
                "Easy (10 chances)": DifficultyLevel.EASY,
                "Medium (5 chances)": DifficultyLevel.MEDIUM,
                "Hard (3 chances)": DifficultyLevel.HARD,
                "Insane (1 chance)": DifficultyLevel.INSANE,
            }
            
            difficulty = difficulty_map.get(diff_choice, DifficultyLevel.MEDIUM)
            return_to_menu = run_theory_game(piece_dir, book_path, difficulty, player_color)
            if not return_to_menu:
                break
    
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
