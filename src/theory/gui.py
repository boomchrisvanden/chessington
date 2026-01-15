"""
Pygame GUI for opening theory practice.

Provides a graphical interface for practicing chess opening theory.
"""

import os
import sys
import pygame
from pathlib import Path
from typing import Optional, Tuple, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from theory.practice import (
    TheoryPracticeGame,
    DifficultyLevel,
    find_opening_name_matches,
)
from book.polyglot_zobrist import algebraic_to_square, square_to_algebraic


# Piece type mapping for display
PIECE_TYPE_TO_LETTER = {
    1: 'P',  # Pawn
    2: 'N',  # Knight
    3: 'B',  # Bishop
    4: 'R',  # Rook
    5: 'Q',  # Queen
    6: 'K',  # King
}


class TheoryPracticeGUI:
    """
    Pygame-based GUI for opening theory practice.
    """
    
    def __init__(
        self,
        book_path: str,
        piece_dir: str,
        difficulty: DifficultyLevel,
        player_color: int,
        search_mode: bool = False,
    ):
        pygame.init()
        pygame.display.set_caption("Opening Theory Practice")
        
        self.square_size = 80
        self.board_size = self.square_size * 8
        self.info_height = 120
        self.header_height = 50
        self.width = self.board_size
        self.height = self.header_height + self.board_size + self.info_height
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 24)
        self.small_font = pygame.font.SysFont("consolas", 18)
        self.header_font = pygame.font.SysFont("consolas", 20, bold=True)
        self.piece_font = pygame.font.SysFont("consolas", 46, bold=True)
        
        self.game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=difficulty,
            player_color=player_color
        )
        
        self.status_message = "Make a book move to continue the opening"
        self.running = True
        self.dragging = False
        self.drag_from: Optional[int] = None
        self.drag_piece: Optional[Tuple[int, int]] = None
        self.drag_pos = (0, 0)
        self.last_move: Optional[Tuple[int, int]] = None
        self.search_enabled = search_mode
        self.search_active = False
        self.search_query = ""
        self.search_results: List[str] = []
        self.search_result_rects: List[Tuple[pygame.Rect, int, str]] = []
        self.search_selected_index = 0
        self.search_previous_status = ""
        self.selected_opening_name: Optional[str] = None
        
        # Flip board if playing as black
        self.flip_board = (player_color == 1)
        
        # Colors
        self.light_color = (240, 217, 181)
        self.dark_color = (181, 136, 99)
        self.highlight_color = (186, 202, 68)
        self.error_color = (220, 80, 80)
        self.success_color = (80, 180, 80)
        
        # Button rectangles
        panel_y = self.header_height + self.board_size
        self.reset_button_rect = pygame.Rect(10, panel_y + 75, 100, 32)
        self.menu_button_rect = pygame.Rect(120, panel_y + 75, 100, 32)
        self.quit_button_rect = pygame.Rect(230, panel_y + 75, 100, 32)
        self.new_line_button_rect = pygame.Rect(340, panel_y + 75, 120, 32)
        self.search_button_rect = pygame.Rect(470, panel_y + 75, 120, 32)
        
        # Transition timer for showing success message before new line
        self.transition_timer: Optional[int] = None  # Timestamp when transition started
        self.transition_delay = 1500  # milliseconds to show success message
        
        # Flag to indicate returning to main menu
        self.return_to_menu = False
        
        # Load piece images
        self.pieces = self.load_piece_images(piece_dir)
        self.fallback_pieces = self.build_fallback_piece_surfaces()
        
        # Start the first line
        self.game.start_new_line()
        if self.search_enabled:
            self._open_search()
        else:
            self._check_opponent_turn()
    
    def load_piece_images(self, piece_dir: str):
        """Load piece images from directory."""
        pieces = {}
        piece_dir_path = Path(piece_dir)
        
        if not piece_dir_path.exists():
            return pieces
        
        # Build a mapping of lowercase filenames to paths
        png_by_name = {}
        for p in piece_dir_path.iterdir():
            if p.is_file() and p.suffix.lower() == ".png":
                png_by_name[p.name.lower()] = p
        
        # Color codes: 0 = white, 1 = black
        for color, ccode in ((0, 'w'), (1, 'b')):
            for pt, pcode in (
                (1, 'P'),  # Pawn
                (2, 'N'),  # Knight
                (3, 'B'),  # Bishop
                (4, 'R'),  # Rook
                (5, 'Q'),  # Queen
                (6, 'K'),  # King
            ):
                candidate_names = [
                    f"{ccode}{pcode}.png",
                    f"{ccode}{pcode.lower()}.png",
                ]
                
                png_path = None
                for name in candidate_names:
                    if name.lower() in png_by_name:
                        png_path = png_by_name[name.lower()]
                        break
                
                if png_path is None or not png_path.exists():
                    continue
                
                img = pygame.image.load(str(png_path)).convert_alpha()
                img = pygame.transform.smoothscale(img, (self.square_size, self.square_size))
                pieces[(color, pt)] = img
        
        return pieces
    
    def build_fallback_piece_surfaces(self):
        """Build fallback text-based piece surfaces."""
        fallback = {}
        for color in (0, 1):
            for pt, letter in PIECE_TYPE_TO_LETTER.items():
                fg = (245, 245, 245) if color == 0 else (20, 20, 20)
                outline = (20, 20, 20) if color == 0 else (245, 245, 245)
                base = self.piece_font.render(letter, True, fg)
                shadow = self.piece_font.render(letter, True, outline)
                
                surf = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
                rect = base.get_rect(center=(self.square_size // 2, self.square_size // 2))
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    surf.blit(shadow, rect.move(dx, dy))
                surf.blit(base, rect)
                fallback[(color, pt)] = surf
        return fallback
    
    def square_from_pos(self, pos: Tuple[int, int]) -> Optional[int]:
        """Convert screen position to board square index."""
        x, y = pos
        y = y - self.header_height  # Adjust for header
        
        if x < 0 or y < 0 or x >= self.board_size or y >= self.board_size:
            return None
        
        visual_file = x // self.square_size
        visual_rank = y // self.square_size
        
        # Convert visual coordinates to actual board coordinates
        # This must match the logic in draw_board
        if self.flip_board:
            actual_file = 7 - visual_file
            actual_rank = visual_rank
        else:
            actual_file = visual_file
            actual_rank = 7 - visual_rank
        
        return actual_rank * 8 + actual_file
    
    def pos_from_square(self, square: int) -> Tuple[int, int]:
        """Convert board square index to screen position."""
        actual_file = square % 8
        actual_rank = square // 8
        
        # Convert actual board coordinates to visual coordinates
        # This must be the inverse of square_from_pos
        if self.flip_board:
            visual_file = 7 - actual_file
            visual_rank = actual_rank
        else:
            visual_file = actual_file
            visual_rank = 7 - actual_rank
        
        x = visual_file * self.square_size
        y = visual_rank * self.square_size + self.header_height
        return x, y
    
    def _check_opponent_turn(self) -> None:
        """Check if it's the opponent's turn and make their move."""
        # Don't make moves during transition
        if self.transition_timer is not None:
            return
        
        if self.game.side_to_move != self.game.player_color and not self.game.game_over:
            if self.game.is_line_complete():
                # Line complete - will be handled by transition
                return
            
            move = self.game.make_opponent_move()
            if move is not None:
                # Update last move highlight
                from_sq = algebraic_to_square(move[0:2])
                to_sq = algebraic_to_square(move[2:4])
                self.last_move = (from_sq, to_sq)
                self.status_message = f"Opponent played: {move}"
                
                # Check if line is now complete after opponent's move
                if self.game.is_line_complete():
                    self._on_line_complete(player_completed=False)
    
    def _start_line_transition(self, player_completed: bool = False) -> None:
        """
        Start the transition to a new line after completing one.
        
        Args:
            player_completed: True if player made the final move (lines_completed already incremented)
        """
        if not player_completed:
            # Opponent made the final move, need to increment
            self.game.lines_completed += 1
        # Keep the success message from try_player_move if player completed,
        # otherwise set a completion message
        if not player_completed:
            self.status_message = f"Line complete! ({self.game.lines_completed} lines completed)"
        # Start the transition timer
        self.transition_timer = pygame.time.get_ticks()
    
    def _check_transition(self) -> None:
        """Check if transition timer has elapsed and start new line."""
        if self.transition_timer is None:
            return
        
        elapsed = pygame.time.get_ticks() - self.transition_timer
        if elapsed >= self.transition_delay:
            self.transition_timer = None
            self.game.start_new_line()
            self.last_move = None
            self.status_message = f"New line: {self.game.current_opening_name}"
            self._check_opponent_turn()
    
    def _can_skip_line(self) -> bool:
        """Check if the player can manually skip to a new line (only in infinite mode)."""
        return self.game.difficulty == DifficultyLevel.INFINITE and not self.game.game_over

    def _in_opening_practice(self) -> bool:
        """Check if we're practicing a selected opening name."""
        return self.selected_opening_name is not None

    def _refresh_search_results(self, reset_selection: bool = False) -> None:
        """Refresh fuzzy search results based on the current query."""
        self.search_results = find_opening_name_matches(self.search_query, limit=8)
        if reset_selection or self.search_selected_index >= len(self.search_results):
            self.search_selected_index = 0

    def _open_search(self) -> None:
        """Open the opening search overlay."""
        if not self.search_enabled:
            return
        if self.game.game_over:
            self.status_message = "Game over! Press Reset to try again."
            return
        self.search_active = True
        self.search_query = ""
        self._refresh_search_results(reset_selection=True)
        self.search_result_rects = []
        self.search_previous_status = self.status_message
        self.status_message = "Search for an opening to practice."
        self.transition_timer = None
        self.dragging = False
        self.drag_from = None
        self.drag_piece = None

    def _close_search(self) -> None:
        """Close the opening search overlay."""
        self.search_active = False
        self.search_result_rects = []
        if self.search_previous_status:
            self.status_message = self.search_previous_status
        self._check_opponent_turn()

    def _select_search_result(self, opening_name: str) -> None:
        """Start practicing the selected opening."""
        if not self.game.start_new_line_for_opening(opening_name):
            self.status_message = f"No book lines found for {opening_name}."
            return
        self.selected_opening_name = opening_name
        self.search_active = False
        self.last_move = None
        self.transition_timer = None
        self.status_message = f"New line: {self.game.current_opening_name}"
        self._check_opponent_turn()

    def _restart_selected_line(self) -> None:
        """Restart the current opening line."""
        if self.selected_opening_name is None:
            return
        if not self.game.start_new_line_for_opening(self.selected_opening_name):
            self.status_message = f"No book lines found for {self.selected_opening_name}."
            return
        self.last_move = None
        self.transition_timer = None
        self.status_message = f"New line: {self.game.current_opening_name}"
        self._check_opponent_turn()

    def _handle_search_key(self, event) -> None:
        """Handle keyboard input while search overlay is active."""
        if event.key == pygame.K_ESCAPE:
            self._close_search()
            return
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if self.search_results:
                index = min(self.search_selected_index, len(self.search_results) - 1)
                self._select_search_result(self.search_results[index])
            return
        if event.key in (pygame.K_UP, pygame.K_DOWN):
            if self.search_results:
                delta = -1 if event.key == pygame.K_UP else 1
                self.search_selected_index = (
                    self.search_selected_index + delta
                ) % len(self.search_results)
            return
        if event.key == pygame.K_BACKSPACE:
            self.search_query = self.search_query[:-1]
            self._refresh_search_results(reset_selection=True)
            return
        if event.unicode and event.unicode.isprintable():
            if len(self.search_query) < 40:
                self.search_query += event.unicode
                self._refresh_search_results(reset_selection=True)

    def _handle_search_click(self, pos: Tuple[int, int]) -> None:
        """Handle mouse clicks on search results."""
        for rect, index, opening_name in self.search_result_rects:
            if rect.collidepoint(pos):
                self.search_selected_index = index
                self._select_search_result(opening_name)
                return

    def _on_line_complete(self, player_completed: bool) -> None:
        """Handle line completion based on current practice mode."""
        if self._in_opening_practice():
            if not player_completed:
                self.game.lines_completed += 1
            self.status_message = (
                f"Line complete! ({self.game.lines_completed} lines) "
                "Restart the line or search again."
            )
            return
        self._start_line_transition(player_completed=player_completed)
    
    def handle_mouse_down(self, event) -> None:
        """Handle mouse button down event."""
        if event.button != 1:
            return

        if self.search_active:
            self._handle_search_click(event.pos)
            return
        
        # Don't process input during transition
        if self.transition_timer is not None:
            return
        
        # Check button clicks
        # Reset button - only works when game is over
        if self.reset_button_rect.collidepoint(event.pos):
            if self.game.game_over:
                if self._in_opening_practice():
                    self.game.reset_game(self.selected_opening_name)
                    self.status_message = f"New line: {self.game.current_opening_name}"
                else:
                    self.game.reset_game()
                    self.status_message = "Game reset. Make a book move!"
                self.last_move = None
                self._check_opponent_turn()
            return
        
        # Menu button - return to main menu
        if self.menu_button_rect.collidepoint(event.pos):
            self.return_to_menu = True
            self.running = False
            return
        
        if self.quit_button_rect.collidepoint(event.pos):
            self.running = False
            return

        if self.search_enabled and self.search_button_rect.collidepoint(event.pos):
            self._open_search()
            return
        
        if self.new_line_button_rect.collidepoint(event.pos):
            if self._in_opening_practice():
                if self.game.is_line_complete() and not self.game.game_over:
                    self._restart_selected_line()
                return
            if self._can_skip_line():
                self.game.start_new_line()
                self.last_move = None
                self.status_message = f"New line: {self.game.current_opening_name}"
                self._check_opponent_turn()
            return
        
        if self.game.game_over:
            return
        
        if self.dragging:
            return
        
        square = self.square_from_pos(event.pos)
        if square is None:
            return
        
        piece = self.game.squares[square]
        if piece is None:
            return
        
        color, _ = piece
        if color != self.game.player_color:
            return
        
        if self.game.side_to_move != self.game.player_color:
            return
        
        self.dragging = True
        self.drag_from = square
        self.drag_piece = piece
        self.drag_pos = event.pos
    
    def handle_mouse_motion(self, event) -> None:
        """Handle mouse motion event."""
        if self.search_active:
            return
        if not self.dragging:
            return
        self.drag_pos = event.pos
    
    def handle_mouse_up(self, event) -> None:
        """Handle mouse button up event."""
        if self.search_active:
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
        
        # Don't process moves during transition
        if self.transition_timer is not None:
            return
        
        # Build UCI move
        from_str = square_to_algebraic(from_sq)
        to_str = square_to_algebraic(drop_sq)
        move_text = from_str + to_str
        
        # Handle promotion
        color, piece_type = piece
        if piece_type == 1:  # Pawn
            to_rank = drop_sq // 8
            promotion_rank = 7 if color == 0 else 0
            if to_rank == promotion_rank:
                move_text += "q"  # Default to queen promotion
        
        # Try the move
        result = self.game.try_player_move(move_text)
        
        # None means illegal move - piece snaps back silently
        if result is None:
            return
        
        success, message = result
        self.status_message = message
        
        if success:
            # Update last move highlight
            self.last_move = (from_sq, drop_sq)
            
            # Check if line is complete
            if self.game.is_line_complete():
                # lines_completed already incremented by try_player_move
                self._on_line_complete(player_completed=True)
            else:
                # Continue with opponent's turn
                self._check_opponent_turn()
    
    def draw(self) -> None:
        """Draw the entire screen."""
        self.screen.fill((30, 30, 30))
        self.draw_header()
        self.draw_board()
        self.draw_info_panel()
        if self.search_active:
            self.draw_search_overlay()
    
    def draw_header(self) -> None:
        """Draw the header with opening name."""
        # Background
        pygame.draw.rect(
            self.screen,
            (40, 60, 80),
            (0, 0, self.width, self.header_height)
        )
        
        # Opening name
        opening_surf = self.header_font.render(
            self.game.current_opening_name,
            True,
            (255, 255, 255)
        )
        opening_rect = opening_surf.get_rect(center=(self.width // 2, self.header_height // 2))
        self.screen.blit(opening_surf, opening_rect)
        
        # Lives display (top right)
        lives_text = f"Lives: {self.game.get_lives_display()}"
        lives_surf = self.small_font.render(lives_text, True, (255, 200, 100))
        self.screen.blit(lives_surf, (self.width - 100, 10))
    
    def draw_board(self) -> None:
        """Draw the chess board and pieces."""
        # Draw squares
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
                
                # Highlight last move
                if self.last_move is not None and square_index in self.last_move:
                    color = self.highlight_color
                
                x = visual_file * self.square_size
                y = visual_rank * self.square_size + self.header_height
                pygame.draw.rect(self.screen, color, (x, y, self.square_size, self.square_size))
                
                # Draw piece
                piece = self.game.squares[square_index]
                if piece is not None:
                    if self.dragging and self.drag_from == square_index:
                        continue
                    img = self.pieces.get(piece)
                    if img is None:
                        img = self.fallback_pieces.get(piece)
                    if img is not None:
                        self.screen.blit(img, (x, y))
        
        # Draw file/rank labels
        for i in range(8):
            if self.flip_board:
                file_label = chr(ord('h') - i)
                rank_label = str(i + 1)
            else:
                file_label = chr(ord('a') + i)
                rank_label = str(8 - i)
            
            # File labels (bottom)
            label = self.small_font.render(file_label, True, (0, 0, 0))
            x = i * self.square_size + 5
            y = self.header_height + self.board_size - 20
            self.screen.blit(label, (x, y))
            
            # Rank labels (left)
            label = self.small_font.render(rank_label, True, (0, 0, 0))
            x = 5
            y = i * self.square_size + self.header_height + 5
            self.screen.blit(label, (x, y))
        
        # Draw dragged piece
        if self.dragging and self.drag_piece is not None:
            img = self.pieces.get(self.drag_piece)
            if img is None:
                img = self.fallback_pieces.get(self.drag_piece)
            if img is not None:
                x = self.drag_pos[0] - self.square_size // 2
                y = self.drag_pos[1] - self.square_size // 2
                self.screen.blit(img, (x, y))
    
    def draw_info_panel(self) -> None:
        """Draw the info panel at the bottom."""
        panel_y = self.header_height + self.board_size
        
        # Background
        pygame.draw.rect(
            self.screen,
            (30, 30, 30),
            (0, panel_y, self.width, self.info_height)
        )
        
        # Score
        score_surf = self.font.render(
            self.game.get_score_display(),
            True,
            (220, 220, 220)
        )
        self.screen.blit(score_surf, (10, panel_y + 10))
        
        # Status message
        status_color = (200, 200, 0)
        if "Wrong" in self.status_message or "over" in self.status_message.lower():
            status_color = self.error_color
        elif "Correct" in self.status_message:
            status_color = self.success_color
        
        status_surf = self.small_font.render(self.status_message, True, status_color)
        self.screen.blit(status_surf, (10, panel_y + 45))
        
        # Buttons
        # Reset button - only active when game is over
        if self.game.game_over:
            reset_color = (100, 100, 180)
        else:
            reset_color = (60, 60, 80)  # Grayed out
        pygame.draw.rect(self.screen, reset_color, self.reset_button_rect, border_radius=4)
        reset_label = self.small_font.render("Reset", True, (255, 255, 255) if self.game.game_over else (120, 120, 120))
        reset_label_rect = reset_label.get_rect(center=self.reset_button_rect.center)
        self.screen.blit(reset_label, reset_label_rect)
        
        # Menu button - always active
        menu_color = (100, 100, 140)
        pygame.draw.rect(self.screen, menu_color, self.menu_button_rect, border_radius=4)
        menu_label = self.small_font.render("Menu", True, (255, 255, 255))
        menu_label_rect = menu_label.get_rect(center=self.menu_button_rect.center)
        self.screen.blit(menu_label, menu_label_rect)
        
        # Quit button
        quit_color = (180, 80, 80)
        pygame.draw.rect(self.screen, quit_color, self.quit_button_rect, border_radius=4)
        quit_label = self.small_font.render("Quit", True, (255, 255, 255))
        quit_label_rect = quit_label.get_rect(center=self.quit_button_rect.center)
        self.screen.blit(quit_label, quit_label_rect)

        if self.search_enabled:
            search_active = not self.game.game_over
            if self._in_opening_practice() and self.game.is_line_complete():
                search_text = "Search Again"
            else:
                search_text = "Search"
            search_color = (100, 120, 160) if search_active else (60, 60, 80)
            pygame.draw.rect(self.screen, search_color, self.search_button_rect, border_radius=4)
            search_label = self.small_font.render(
                search_text,
                True,
                (255, 255, 255) if search_active else (120, 120, 120),
            )
            search_label_rect = search_label.get_rect(center=self.search_button_rect.center)
            self.screen.blit(search_label, search_label_rect)
        
        if self._in_opening_practice():
            restart_active = self.game.is_line_complete() and not self.game.game_over
            restart_color = (80, 140, 80) if restart_active else (60, 60, 80)
            pygame.draw.rect(self.screen, restart_color, self.new_line_button_rect, border_radius=4)
            restart_label = self.small_font.render(
                "Restart Line",
                True,
                (255, 255, 255) if restart_active else (120, 120, 120),
            )
            restart_label_rect = restart_label.get_rect(center=self.new_line_button_rect.center)
            self.screen.blit(restart_label, restart_label_rect)
        elif self._can_skip_line():
            new_line_color = (80, 140, 80)
            pygame.draw.rect(self.screen, new_line_color, self.new_line_button_rect, border_radius=4)
            new_line_label = self.small_font.render("Skip Line", True, (255, 255, 255))
            new_line_label_rect = new_line_label.get_rect(center=self.new_line_button_rect.center)
            self.screen.blit(new_line_label, new_line_label_rect)

    def draw_search_overlay(self) -> None:
        """Draw the opening search overlay."""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.screen.blit(overlay, (0, 0))

        panel_rect = pygame.Rect(60, 60, self.width - 120, self.height - 120)
        pygame.draw.rect(self.screen, (45, 50, 60), panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (90, 110, 140), panel_rect, 2, border_radius=8)

        title = self.header_font.render("Search Openings", True, (240, 240, 240))
        self.screen.blit(title, (panel_rect.x + 20, panel_rect.y + 15))

        query_label = self.small_font.render("Query:", True, (200, 200, 200))
        self.screen.blit(query_label, (panel_rect.x + 20, panel_rect.y + 55))

        display_query = self.search_query
        if len(display_query) > 30:
            display_query = "..." + display_query[-27:]
        query_text = display_query if display_query else "Type to search..."
        query_color = (255, 255, 255) if display_query else (150, 150, 150)
        query_surf = self.font.render(query_text, True, query_color)
        self.screen.blit(query_surf, (panel_rect.x + 20, panel_rect.y + 75))

        self.search_result_rects = []
        results_y = panel_rect.y + 120
        for idx, name in enumerate(self.search_results):
            rect = pygame.Rect(panel_rect.x + 20, results_y + idx * 32, panel_rect.w - 40, 28)
            is_selected = idx == self.search_selected_index
            color = (90, 120, 160) if is_selected else (55, 65, 80)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            if is_selected:
                pygame.draw.rect(self.screen, (160, 190, 220), rect, 2, border_radius=4)
            label = self.small_font.render(name, True, (230, 230, 230))
            self.screen.blit(label, (rect.x + 8, rect.y + 5))
            self.search_result_rects.append((rect, idx, name))

        if not self.search_results:
            empty_label = self.small_font.render("No matches found.", True, (200, 150, 150))
            self.screen.blit(empty_label, (panel_rect.x + 20, results_y))

        hint = self.small_font.render(
            "Up/Down: navigate | Enter: select | Esc: close",
            True,
            (180, 180, 180),
        )
        self.screen.blit(hint, (panel_rect.x + 20, panel_rect.bottom - 30))
    
    def run(self) -> bool:
        """
        Main game loop.
        
        Returns:
            True if user wants to return to main menu, False otherwise
        """
        while self.running:
            self.clock.tick(60)
            
            # Check for transition timer
            self._check_transition()
            
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
                    if self.search_active:
                        self._handle_search_key(event)
                        continue
                    if event.key == pygame.K_ESCAPE:
                        self.return_to_menu = True
                        self.running = False
                    elif event.key == pygame.K_r and self.game.game_over:
                        # Reset only works when game is over
                        if self._in_opening_practice():
                            self.game.reset_game(self.selected_opening_name)
                            self.status_message = f"New line: {self.game.current_opening_name}"
                        else:
                            self.game.reset_game()
                            self.status_message = "Game reset. Make a book move!"
                        self.last_move = None
                        self._check_opponent_turn()
                    elif event.key == pygame.K_n and self._can_skip_line():
                        # Skip line only in infinite/practice mode
                        self.game.start_new_line()
                        self.last_move = None
                        self.status_message = f"New line: {self.game.current_opening_name}"
                        self._check_opponent_turn()
                    elif event.key == pygame.K_f and self.search_enabled:
                        # 'f' for find
                        self._open_search()
                    elif event.key == pygame.K_m:
                        # 'm' for menu
                        self.return_to_menu = True
                        self.running = False
            
            self.draw()
            pygame.display.flip()
        
        return self.return_to_menu


def run_theory_practice(
    book_path: str,
    piece_dir: str,
    difficulty: DifficultyLevel,
    player_color: int,
    search_mode: bool = False,
) -> bool:
    """
    Launch the theory practice GUI.
    
    Args:
        book_path: Path to the Polyglot opening book (.bin file)
        piece_dir: Path to directory containing piece images
        difficulty: Selected difficulty level
        player_color: 0 for white, 1 for black
        search_mode: True to start in opening search mode
    
    Returns:
        True if user wants to return to main menu, False to quit entirely
    """
    gui = TheoryPracticeGUI(
        book_path=book_path,
        piece_dir=piece_dir,
        difficulty=difficulty,
        player_color=player_color,
        search_mode=search_mode,
    )
    return gui.run()
