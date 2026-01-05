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

from theory.practice import TheoryPracticeGame, DifficultyLevel
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
        player_color: int
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
                    self._start_line_transition(player_completed=False)
    
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
    
    def handle_mouse_down(self, event) -> None:
        """Handle mouse button down event."""
        if event.button != 1:
            return
        
        # Don't process input during transition
        if self.transition_timer is not None:
            return
        
        # Check button clicks
        # Reset button - only works when game is over
        if self.reset_button_rect.collidepoint(event.pos):
            if self.game.game_over:
                self.game.reset_game()
                self.last_move = None
                self.status_message = "Game reset. Make a book move!"
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
        
        # New Line button - only in infinite mode
        if self.new_line_button_rect.collidepoint(event.pos) and self._can_skip_line():
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
        if not self.dragging:
            return
        self.drag_pos = event.pos
    
    def handle_mouse_up(self, event) -> None:
        """Handle mouse button up event."""
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
                # Start transition to show success message
                # lines_completed already incremented by try_player_move
                self._start_line_transition(player_completed=True)
            else:
                # Continue with opponent's turn
                self._check_opponent_turn()
    
    def draw(self) -> None:
        """Draw the entire screen."""
        self.screen.fill((30, 30, 30))
        self.draw_header()
        self.draw_board()
        self.draw_info_panel()
    
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
        
        # New Line button - only in infinite mode (practice mode)
        if self._can_skip_line():
            new_line_color = (80, 140, 80)
            pygame.draw.rect(self.screen, new_line_color, self.new_line_button_rect, border_radius=4)
            new_line_label = self.small_font.render("Skip Line", True, (255, 255, 255))
            new_line_label_rect = new_line_label.get_rect(center=self.new_line_button_rect.center)
            self.screen.blit(new_line_label, new_line_label_rect)
    
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
                    if event.key == pygame.K_ESCAPE:
                        self.return_to_menu = True
                        self.running = False
                    elif event.key == pygame.K_r and self.game.game_over:
                        # Reset only works when game is over
                        self.game.reset_game()
                        self.last_move = None
                        self.status_message = "Game reset. Make a book move!"
                        self._check_opponent_turn()
                    elif event.key == pygame.K_n and self._can_skip_line():
                        # Skip line only in infinite/practice mode
                        self.game.start_new_line()
                        self.last_move = None
                        self.status_message = f"New line: {self.game.current_opening_name}"
                        self._check_opponent_turn()
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
    player_color: int
) -> bool:
    """
    Launch the theory practice GUI.
    
    Args:
        book_path: Path to the Polyglot opening book (.bin file)
        piece_dir: Path to directory containing piece images
        difficulty: Selected difficulty level
        player_color: 0 for white, 1 for black
    
    Returns:
        True if user wants to return to main menu, False to quit entirely
    """
    gui = TheoryPracticeGUI(
        book_path=book_path,
        piece_dir=piece_dir,
        difficulty=difficulty,
        player_color=player_color
    )
    return gui.run()
