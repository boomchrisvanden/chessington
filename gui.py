import math
import os
import sys
import pygame
from pathlib import Path
from typing import Optional, Tuple, List, Callable

# Import theory practice module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from theory.practice import DifficultyLevel
from theory.gui import run_theory_practice

from src.core.types import (
    Color,
    Move,
    PieceType,
    move_from_uci,
    square_to_str,
)
from src.core.board import Board
from src.protocols.engine_client import EngineClient
from src.utils.assets import (
    build_fallback_piece_surfaces,
    get_piece_dir,
    load_piece_images,
)


# Piece values for material counting
PIECE_VALUES = {
    PieceType.PAWN: 1,
    PieceType.KNIGHT: 3,
    PieceType.BISHOP: 3,
    PieceType.ROOK: 5,
    PieceType.QUEEN: 9,
    PieceType.KING: 0,
}

# Arrow/highlight colors (RGBA)
HIGHLIGHT_RGBA = (235, 97, 80, 140)
ARROW_RGBA = (245, 171, 53, 200)


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
        self.engine_depth = 6
        self.engine_side = Color.WHITE
        self.move_history: List[str] = []
        self.engine = EngineClient()
        self.engine_thinking = False
        self.engine_button_rect = pygame.Rect(self.width - 220, self.board_size + 8, 210, 32)
        self.menu_button_rect = pygame.Rect(10, self.board_size + 8, 80, 32)

        # Flag to indicate returning to main menu
        self.return_to_menu = False

        # Board orientation
        self.flip_board = False

        # Right-click annotations (arrows and square highlights)
        self.arrows: List[Tuple[int, int]] = []
        self.highlights: set = set()
        self.right_click_from: Optional[int] = None
        self.right_dragging = False
        self.right_drag_pos = (0, 0)

        self.light_color = (240, 217, 181)
        self.dark_color = (181, 136, 99)
        self.highlight_color = (186, 202, 68)

        # Load piece images
        self.pieces = load_piece_images(piece_dir, self.square_size)
        self.fallback_pieces = build_fallback_piece_surfaces(
            self.square_size, self.piece_font
        )

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

            self._poll_engine()
            self.draw()
            pygame.display.flip()

        self.engine.shutdown()
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
        self.engine.epoch += 1
        self._reset_board_state()
        if not self.engine.ensure_running():
            self.play_vs_engine = False
            self.status_message = "Engine failed to start."
            return
        self.engine.send("ucinewgame")
        self.status_message = f"Engine game started (depth {self.engine_depth})."
        if self.board.side_to_move == self.engine_side:
            self._request_engine_move()

    def _request_engine_move(self) -> None:
        if self.engine_thinking or self.game_over:
            return
        if not self.engine.ensure_running():
            self.status_message = "Engine not available."
            return
        self.engine_thinking = True
        self.status_message = f"Engine thinking (depth {self.engine_depth})..."
        self.engine.search_async(self.move_history, self.engine_depth)

    def _poll_engine(self) -> None:
        for epoch, bestmove, error in self.engine.poll():
            self.engine_thinking = False
            if epoch != self.engine.epoch:
                continue
            if error is not None:
                self.status_message = error
                continue
            if bestmove is None:
                self.status_message = "Engine returned no move."
                continue
            self._apply_engine_move(bestmove)

    def square_from_pos(self, pos: Tuple[int, int]) -> Optional[int]:
        x, y = pos
        if x < 0 or y < 0 or x >= self.board_size or y >= self.board_size:
            return None
        visual_file = x // self.square_size
        visual_rank = y // self.square_size
        if self.flip_board:
            actual_file = 7 - visual_file
            actual_rank = visual_rank
        else:
            actual_file = visual_file
            actual_rank = 7 - visual_rank
        return actual_rank * 8 + actual_file

    def pos_from_square(self, square: int) -> Tuple[int, int]:
        actual_file = square % 8
        actual_rank = square // 8
        if self.flip_board:
            visual_file = 7 - actual_file
            visual_rank = actual_rank
        else:
            visual_file = actual_file
            visual_rank = 7 - actual_rank
        return visual_file * self.square_size, visual_rank * self.square_size

    def handle_mouse_down(self, event):
        if event.button == 3:
            sq = self.square_from_pos(event.pos)
            if sq is not None:
                self.right_click_from = sq
                self.right_dragging = True
                self.right_drag_pos = event.pos
            return
        if event.button != 1:
            return
        # Clear annotations on left click
        self.arrows.clear()
        self.highlights.clear()
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
        if self.right_dragging:
            self.right_drag_pos = event.pos
        if not self.dragging:
            return
        self.drag_pos = event.pos

    def handle_mouse_up(self, event):
        if event.button == 3:
            if self.right_dragging and self.right_click_from is not None:
                drop_sq = self.square_from_pos(event.pos)
                if drop_sq is not None:
                    if drop_sq == self.right_click_from:
                        if drop_sq in self.highlights:
                            self.highlights.discard(drop_sq)
                        else:
                            self.highlights.add(drop_sq)
                    else:
                        arrow = (self.right_click_from, drop_sq)
                        if arrow in self.arrows:
                            self.arrows.remove(arrow)
                        else:
                            self.arrows.append(arrow)
            self.right_click_from = None
            self.right_dragging = False
            return
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
            self.return_to_menu = True
            self.running = False
        elif event.key == pygame.K_m:
            self.return_to_menu = True
            self.running = False
        else:
            ch = event.unicode
            if ch and ch.isprintable():
                self.input_text += ch

    def handle_move_input(self):
        text = self.input_text.strip()
        self.input_text = ""
        self._apply_move_text(text)

    def _apply_move_text(self, text: str) -> None:
        if text.lower() == "exit":
            self.running = False
            return
        if text.lower() == "reset":
            self.engine.epoch += 1
            self._reset_board_state()
            self.status_message = "Reset to start position."
            if self.play_vs_engine:
                if self.engine_thinking:
                    self.status_message = "Reset to start position (engine busy)."
                    return
                if self.engine.ensure_running():
                    self.engine.send("ucinewgame")
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

        move = move_from_uci(text)
        if move is None:
            self.status_message = f"Invalid syntax: '{text}'"
            return

        ok, reason = self.board.validate_move(move)
        if not ok:
            self.status_message = f"Invalid move: {reason}"
            return

        self._apply_move(move, "Played")
        if self.play_vs_engine and not self.game_over and self.board.side_to_move == self.engine_side:
            self._request_engine_move()

    def _apply_move(self, move: Move, label: str) -> None:
        self.board.make_move(move)
        self.last_move = move
        self.move_history.append(move.uci())
        self.status_message = f"{label}: {move.uci()}"
        if self.board.is_checkmate():
            winner = "White" if self.board.side_to_move == Color.BLACK else "Black"
            self.status_message = f"Checkmate! {winner} wins. Type 'reset' to restart."
            self.game_over = True
            return
        if self.board.is_draw():
            reason = self.board.draw_reason()
            self.status_message = f"Draw by {reason}! Type 'reset' to restart."
            self.game_over = True
            return
        if self.board.in_check(self.board.side_to_move):
            self.status_message += " (check)"

    def _apply_engine_move(self, move_text: str) -> None:
        move = move_from_uci(move_text)
        if move is None:
            self.status_message = f"Engine returned invalid move: {move_text}"
            return
        ok, reason = self.board.validate_move(move)
        if not ok:
            self.status_message = f"Engine move invalid: {reason}"
            return
        self._apply_move(move, "Engine")

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.draw_board()
        self.draw_info_panel()

    def draw_board(self):
        for visual_rank in range(8):
            for visual_file in range(8):
                if self.flip_board:
                    actual_file = 7 - visual_file
                    actual_rank = visual_rank
                else:
                    actual_file = visual_file
                    actual_rank = 7 - visual_rank

                square_index = actual_rank * 8 + actual_file
                is_light = (visual_rank + visual_file) % 2 == 0
                color = self.light_color if is_light else self.dark_color

                if self.last_move is not None and square_index in (self.last_move.from_sq, self.last_move.to_sq):
                    color = self.highlight_color

                x = visual_file * self.square_size
                y = visual_rank * self.square_size
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

        for i in range(8):
            if self.flip_board:
                file_label = chr(ord('h') - i)
                rank_label = str(i + 1)
            else:
                file_label = chr(ord('a') + i)
                rank_label = str(8 - i)

            label = self.small_font.render(file_label, True, (0, 0, 0))
            x = i * self.square_size + 5
            y = self.board_size - 20
            self.screen.blit(label, (x, y))

            label = self.small_font.render(rank_label, True, (0, 0, 0))
            x = 5
            y = i * self.square_size + 5
            self.screen.blit(label, (x, y))

        self._draw_annotations()

        if self.dragging and self.drag_piece is not None:
            img = self.pieces.get(self.drag_piece)
            if img is None:
                img = self.fallback_pieces.get(self.drag_piece)
            if img is not None:
                x = self.drag_pos[0] - self.square_size // 2
                y = self.drag_pos[1] - self.square_size // 2
                self.screen.blit(img, (x, y))

    def _draw_annotations(self):
        for sq in self.highlights:
            x, y = self.pos_from_square(sq)
            surf = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            surf.fill(HIGHLIGHT_RGBA)
            self.screen.blit(surf, (x, y))
        for from_sq, to_sq in self.arrows:
            self._draw_arrow(from_sq, to_sq)
        if self.right_dragging and self.right_click_from is not None:
            temp_to = self.square_from_pos(self.right_drag_pos)
            if temp_to is not None and temp_to != self.right_click_from:
                self._draw_arrow(self.right_click_from, temp_to)

    def _draw_arrow(self, from_sq: int, to_sq: int, y_offset: int = 0):
        half = self.square_size // 2
        fx, fy = self.pos_from_square(from_sq)
        tx, ty = self.pos_from_square(to_sq)
        start = (fx + half, fy + half + y_offset)
        end = (tx + half, ty + half + y_offset)

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            return

        ux, uy = dx / length, dy / length
        px, py = -uy, ux

        shaft_w = max(10, self.square_size // 6)
        head_len = self.square_size * 0.4
        head_w = self.square_size * 0.5

        se_x = end[0] - ux * head_len
        se_y = end[1] - uy * head_len

        shaft = [
            (start[0] + px * shaft_w / 2, start[1] + py * shaft_w / 2),
            (start[0] - px * shaft_w / 2, start[1] - py * shaft_w / 2),
            (se_x - px * shaft_w / 2, se_y - py * shaft_w / 2),
            (se_x + px * shaft_w / 2, se_y + py * shaft_w / 2),
        ]
        head = [
            end,
            (se_x + px * head_w / 2, se_y + py * head_w / 2),
            (se_x - px * head_w / 2, se_y - py * head_w / 2),
        ]

        arrow_surf = pygame.Surface((self.board_size, self.board_size + y_offset), pygame.SRCALPHA)
        pygame.draw.polygon(arrow_surf, ARROW_RGBA, shaft)
        pygame.draw.polygon(arrow_surf, ARROW_RGBA, head)
        self.screen.blit(arrow_surf, (0, 0))

    def _get_material_balance(self) -> Tuple[int, int]:
        """Returns (white_material, black_material)."""
        white = 0
        black = 0
        for sq in range(64):
            piece = self.board.squares[sq]
            if piece is not None:
                color, pt = piece
                val = PIECE_VALUES.get(pt, 0)
                if color == Color.WHITE:
                    white += val
                else:
                    black += val
        return white, black

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

        # Current side to move + material
        stm_text = "White to move" if self.board.side_to_move == Color.WHITE else "Black to move"
        w_mat, b_mat = self._get_material_balance()
        diff = w_mat - b_mat
        if diff > 0:
            mat_text = f"  [+{diff}]"
            mat_color = (220, 220, 220)
        elif diff < 0:
            mat_text = f"  [{diff}]"
            mat_color = (180, 180, 180)
        else:
            mat_text = "  [=]"
            mat_color = (140, 140, 140)

        stm_surf = self.font.render(stm_text, True, (220, 220, 220))
        self.screen.blit(stm_surf, (100, panel_y + 10))
        mat_surf = self.small_font.render(mat_text, True, mat_color)
        stm_end_x = 100 + stm_surf.get_width()
        self.screen.blit(mat_surf, (stm_end_x, panel_y + 14))

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
        self.add_option("Local Game", 130)
        self.add_option("Play Against Engine", 200)
        self.add_option("Theory Practice", 270)
        self.add_option("Opening Search Practice", 340)
        self.add_option("Exit", 410)


class ColorSelectionMenu(MenuScreen):
    """Menu for selecting which color to play as."""

    def __init__(self, title: str = "Select Your Color"):
        super().__init__(width=640, height=400, title=title)
        self.add_option("Play as White", 150)
        self.add_option("Play as Black", 220)
        self.add_option("Back", 310)


class DepthSelectionMenu(MenuScreen):
    """Menu for selecting engine search depth."""

    def __init__(self):
        super().__init__(width=640, height=520, title="Select Engine Depth")
        for i, d in enumerate(range(1, 11)):
            col = i % 2
            row = i // 2
            x = 100 + col * 230
            y = 120 + row * 60
            rect = pygame.Rect(x, y, 200, 45)
            option = MenuOption(f"Depth {d}", rect)
            self.options.append(option)
        back_rect = pygame.Rect((self.width - 300) // 2, 440, 300, 50)
        self.options.append(MenuOption("Back", back_rect))


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


def run_engine_game(piece_dir: str, player_color: Color, depth: int = 6) -> bool:
    """
    Run a game against the engine.

    Returns:
        True if user wants to return to main menu, False otherwise
    """
    board = Board.from_startpos()
    gui = ChessGUI(board, piece_dir)
    gui.play_vs_engine = True
    gui.engine_depth = depth
    gui.engine_side = player_color.other()
    gui.flip_board = (player_color == Color.BLACK)
    gui.engine.epoch += 1
    gui._reset_board_state()

    if gui.engine.ensure_running():
        gui.engine.send("ucinewgame")
        gui.status_message = f"Engine game started (depth {gui.engine_depth}). You play {'White' if player_color == Color.WHITE else 'Black'}."
        if board.side_to_move == gui.engine_side:
            gui._request_engine_move()
    else:
        gui.status_message = "Engine failed to start."

    return gui.run()


def run_theory_game(
    piece_dir: str,
    book_path: str,
    difficulty: DifficultyLevel,
    player_color: int,
    search_mode: bool = False,
) -> bool:
    """
    Run a theory practice game.

    search_mode enables the opening search overlay at startup.

    Returns:
        True if user wants to return to main menu, False to quit entirely
    """
    return run_theory_practice(
        book_path=book_path,
        piece_dir=piece_dir,
        difficulty=difficulty,
        player_color=player_color,
        search_mode=search_mode,
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
                break

        elif choice == "Play Against Engine":
            # Show color selection
            color_menu = ColorSelectionMenu(title="Play Against Engine")
            color_choice = color_menu.run()

            if color_choice in ("Back", "back", "quit", None):
                continue

            player_color = Color.WHITE if color_choice == "Play as White" else Color.BLACK

            # Show depth selection
            depth_menu = DepthSelectionMenu()
            depth_choice = depth_menu.run()

            if depth_choice in ("Back", "back", "quit", None):
                continue

            depth = int(depth_choice.split()[-1])
            return_to_menu = run_engine_game(piece_dir, player_color, depth)
            if not return_to_menu:
                break

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

            difficulty_map = {
                "Infinite (∞ chances)": DifficultyLevel.INFINITE,
                "Easy (10 chances)": DifficultyLevel.EASY,
                "Medium (5 chances)": DifficultyLevel.MEDIUM,
                "Hard (3 chances)": DifficultyLevel.HARD,
                "Insane (1 chance)": DifficultyLevel.INSANE,
            }

            difficulty = difficulty_map.get(diff_choice, DifficultyLevel.MEDIUM)
            return_to_menu = run_theory_game(
                piece_dir,
                book_path,
                difficulty,
                player_color,
            )
            if not return_to_menu:
                break

        elif choice == "Opening Search Practice":
            color_menu = ColorSelectionMenu(title="Opening Search Practice - Color")
            color_choice = color_menu.run()

            if color_choice in ("Back", "back", "quit", None):
                continue

            player_color = 0 if color_choice == "Play as White" else 1

            diff_menu = DifficultyMenu()
            diff_choice = diff_menu.run()

            if diff_choice in ("Back", "back", "quit", None):
                continue

            difficulty_map = {
                "Infinite (∞ chances)": DifficultyLevel.INFINITE,
                "Easy (10 chances)": DifficultyLevel.EASY,
                "Medium (5 chances)": DifficultyLevel.MEDIUM,
                "Hard (3 chances)": DifficultyLevel.HARD,
                "Insane (1 chance)": DifficultyLevel.INSANE,
            }

            difficulty = difficulty_map.get(diff_choice, DifficultyLevel.MEDIUM)
            return_to_menu = run_theory_game(
                piece_dir,
                book_path,
                difficulty,
                player_color,
                search_mode=True,
            )
            if not return_to_menu:
                break

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
