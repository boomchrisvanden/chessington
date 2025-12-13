import os
import sys
import pygame
import shutil
import subprocess
from pathlib import Path

# --------------------------------------------------------------------
# Import your engine core here
# --------------------------------------------------------------------
# Adjust paths/imports to match your project structure
# from chess_engine.core.board import Board
# from chess_engine.core.types import Color, PieceType, Move

# For now I’ll define tiny stand-ins; replace with your real ones.

from enum import IntEnum, auto
from dataclasses import dataclass
from typing import Optional, Tuple, List

class Color(IntEnum):
    WHITE = 0
    BLACK = 1

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
        self._setup_startpos()

    @staticmethod
    def from_startpos() -> "Board":
        return Board()

    def _setup_startpos(self):
        self.squares = [None] * 64

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

    def make_move(self, move: Move) -> None:
        piece = self.squares[move.from_sq]
        if piece is None:
            return

        color, pt = piece

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

        # Handle en-passant style capture (best-effort, no state tracking)
        if pt == PieceType.PAWN:
            from_file = move.from_sq % 8
            to_file = move.to_sq % 8
            if abs(to_file - from_file) == 1 and self.squares[move.to_sq] is None:
                captured_sq = move.to_sq - 8 if color == Color.WHITE else move.to_sq + 8
                if 0 <= captured_sq < 64:
                    self.squares[captured_sq] = None

        # Move the piece (captures are implicit)
        self.squares[move.from_sq] = None
        self.squares[move.to_sq] = (color, pt)

        # Handle promotion (if provided)
        if pt == PieceType.PAWN and move.promotion is not None:
            self.squares[move.to_sq] = (color, move.promotion)

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
        self.status_message = "Enter move in UCI format (e2e4, g8f6, e7e8q)"
        self.last_move: Optional[Move] = None

        self.light_color = (240, 217, 181)
        self.dark_color = (181, 136, 99)
        self.highlight_color = (186, 202, 68)

        # Load piece images
        self.pieces = self.load_piece_images(piece_dir)
        self.fallback_pieces = self.build_fallback_piece_surfaces()

    def load_piece_images(self, piece_dir: str):
        pieces = {}
        piece_dir_path = Path(piece_dir)
        for color, ccode in ((Color.WHITE, 'w'), (Color.BLACK, 'b')):
            for pt, pcode in (
                (PieceType.PAWN,   'P'),
                (PieceType.KNIGHT, 'N'),
                (PieceType.BISHOP, 'B'),
                (PieceType.ROOK,   'R'),
                (PieceType.QUEEN,  'Q'),
                (PieceType.KING,   'K'),
            ):
                png_name = f"{ccode}{pcode}.png"
                png_path = piece_dir_path / png_name

                if not png_path.exists():
                    # If only SVGs are present, try to convert on the fly.
                    svg_candidates = [
                        piece_dir_path / f"{ccode}{pcode}.svg",
                        piece_dir_path / f"{ccode}{pcode.lower()}.svg",
                    ]
                    svg_path = next((p for p in svg_candidates if p.exists()), None)
                    if svg_path is not None:
                        _convert_svg_to_png(svg_path, png_path, self.square_size)

                if not png_path.exists():
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

    def run(self):
        running = True
        while running:
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event)

            self.draw()
            pygame.display.flip()

        pygame.quit()
        sys.exit(0)

    def handle_key(self, event):
        if event.key == pygame.K_RETURN:
            self.handle_move_input()
        elif event.key == pygame.K_BACKSPACE:
            self.input_text = self.input_text[:-1]
        elif event.key == pygame.K_ESCAPE:
            self.input_text = ""
        else:
            ch = event.unicode
            if ch and ch.isprintable():
                self.input_text += ch

    def handle_move_input(self):
        text = self.input_text.strip()
        self.input_text = ""

        parsed = parse_uci_move(text)
        if parsed is None:
            self.status_message = f"Invalid input: '{text}'"
            return

        src, dst, promo_letter = parsed
        promo_piece = promo_letter_to_piece(promo_letter)

        piece = self.board.squares[src]
        if piece is None:
            self.status_message = f"No piece on {square_to_str(src)}"
            return

        move = Move(src, dst, promotion=promo_piece)
        self.board.make_move(move)
        self.last_move = move
        self.status_message = f"Played (unchecked): {move.uci()}"

        # ----------------------------------------------------------------
        # Hook engine reply here if you want:
        #   engine_move = engine_get_move(self.board)
        #   self.board.make_move(engine_move)
        #   self.last_move = engine_move
        #   self.status_message = f"Engine: {engine_move.uci()}"
        # ----------------------------------------------------------------

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

    def draw_info_panel(self):
        panel_y = self.board_size
        pygame.draw.rect(
            self.screen,
            (30, 30, 30),
            (0, panel_y, self.width, self.info_height),
        )

        # Current side to move
        stm_text = "White to move" if self.board.side_to_move == Color.WHITE else "Black to move"
        stm_surf = self.font.render(stm_text, True, (220, 220, 220))
        self.screen.blit(stm_surf, (10, panel_y + 10))

        # Input box
        input_label = self.small_font.render("Move (UCI):", True, (200, 200, 200))
        self.screen.blit(input_label, (10, panel_y + 50))

        # Draw input text
        display_text = self.input_text if self.input_text else ""
        input_surf = self.font.render(display_text + "▏", True, (255, 255, 255))
        self.screen.blit(input_surf, (130, panel_y + 45))

        # Status message
        status_surf = self.small_font.render(self.status_message, True, (200, 200, 0))
        self.screen.blit(status_surf, (10, panel_y + 80))


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    # Change this to wherever your PNGs live
    piece_dir = os.path.join(os.path.dirname(__file__), "assets", "pieces")
    board = Board.from_startpos()
    gui = ChessGUI(board, piece_dir)
    gui.run()

if __name__ == "__main__":
    main()
