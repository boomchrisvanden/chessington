import os
import sys
import pygame

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
        # super minimal: just kings for debugging; replace with real setup
        self.squares = [None] * 64
        self.squares[str_to_square("e1")] = (Color.WHITE, PieceType.KING)
        self.squares[str_to_square("e8")] = (Color.BLACK, PieceType.KING)
        self.side_to_move = Color.WHITE

    def generate_legal(self) -> List[Move]:
        # Replace with your real move generator.
        # For demo: let the white king move one square in any direction legally.
        moves = []
        for i, pc in enumerate(self.squares):
            if pc is None:
                continue
            color, pt = pc
            if color != self.side_to_move:
                continue
            if pt != PieceType.KING:
                continue
            rank = i // 8
            file = i % 8
            for dr in (-1, 0, 1):
                for df in (-1, 0, 1):
                    if dr == 0 and df == 0:
                        continue
                    r2, f2 = rank + dr, file + df
                    if 0 <= r2 < 8 and 0 <= f2 < 8:
                        j = r2 * 8 + f2
                        if self.squares[j] is None or self.squares[j][0] != color:
                            moves.append(Move(i, j))
        return moves

    def make_move(self, move: Move) -> None:
        piece = self.squares[move.from_sq]
        self.squares[move.from_sq] = None
        self.squares[move.to_sq] = piece
        self.side_to_move = Color.BLACK if self.side_to_move == Color.WHITE else Color.WHITE

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

        self.board = board
        self.input_text = ""
        self.status_message = "Enter move in UCI format (e2e4, g8f6, e7e8q)"
        self.last_move: Optional[Move] = None

        self.light_color = (240, 217, 181)
        self.dark_color = (181, 136, 99)
        self.highlight_color = (186, 202, 68)

        # Load piece images
        self.pieces = self.load_piece_images(piece_dir)

    def load_piece_images(self, piece_dir: str):
        pieces = {}
        for color, ccode in ((Color.WHITE, 'w'), (Color.BLACK, 'b')):
            for pt, pcode in (
                (PieceType.PAWN,   'P'),
                (PieceType.KNIGHT, 'N'),
                (PieceType.BISHOP, 'B'),
                (PieceType.ROOK,   'R'),
                (PieceType.QUEEN,  'Q'),
                (PieceType.KING,   'K'),
            ):
                filename = f"{ccode}{pcode}.png"
                path = os.path.join(piece_dir, filename)
                if not os.path.exists(path):
                    continue  # you can assert here instead if you want
                img = pygame.image.load(path).convert_alpha()
                img = pygame.transform.smoothscale(img, (self.square_size, self.square_size))
                pieces[(color, pt)] = img
        return pieces

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

        legal_moves = self.board.generate_legal()
        matching_move = None
        for m in legal_moves:
            if m.from_sq == src and m.to_sq == dst:
                # If this is a promotion move, promotion must match
                if promo_piece is not None:
                    if m.promotion == promo_piece:
                        matching_move = m
                        break
                    else:
                        continue
                # If no promo letter specified, accept only non-promotion moves
                if m.promotion is None:
                    matching_move = m
                    break

        if matching_move is None:
            self.status_message = f"Illegal move: '{text}'"
            return

        # Apply move
        self.board.make_move(matching_move)
        self.last_move = matching_move
        self.status_message = f"Played: {matching_move.uci()}"

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
