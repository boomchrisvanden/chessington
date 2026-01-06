"""
Opening theory practice game logic.

Handles the game state, move validation against opening book,
and tracking lives/chances for the practice mode.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import random

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from book.polyglot_book import PolyglotBook, BookMove, BookEntry
from book.polyglot_zobrist import (
    compute_polyglot_hash,
    algebraic_to_square,
    square_from_file_rank,
)


class DifficultyLevel(Enum):
    """Difficulty levels for theory practice."""
    INFINITE = auto()  # Unlimited chances
    EASY = auto()      # 10 chances, 5 moves depth
    MEDIUM = auto()    # 5 chances, 10 moves depth
    HARD = auto()      # 3 chances, 15 moves depth
    INSANE = auto()    # 1 chance, full line


DIFFICULTY_LIVES = {
    DifficultyLevel.INFINITE: -1,  # -1 means infinite
    DifficultyLevel.EASY: 10,
    DifficultyLevel.MEDIUM: 5,
    DifficultyLevel.HARD: 3,
    DifficultyLevel.INSANE: 1,
}

# Move depth limits per difficulty (moves per side, so total = 2x; None means full line)
# Easy: 5 moves per side = 10 total moves
# Medium: 10 moves per side = 20 total moves
# Hard: 15 moves per side = 30 total moves
DIFFICULTY_MOVE_DEPTH = {
    DifficultyLevel.INFINITE: None,  # Full line
    DifficultyLevel.EASY: 10,        # 5 moves per side
    DifficultyLevel.MEDIUM: 20,      # 10 moves per side
    DifficultyLevel.HARD: 30,        # 15 moves per side
    DifficultyLevel.INSANE: None,    # Full line
}


# Common opening names based on starting moves
# Format: tuple of UCI moves -> opening name
OPENING_NAMES = {
    # King's Pawn openings
    ("e2e4",): "King's Pawn",
    ("e2e4", "e7e5"): "Open Game",
    ("e2e4", "e7e5", "g1f3"): "King's Knight Opening",
    ("e2e4", "e7e5", "g1f3", "b8c6"): "Four Knights Setup",
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"): "Ruy Lopez",
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4"): "Italian Game",
    ("e2e4", "e7e5", "g1f3", "b8c6", "d2d4"): "Scotch Game",
    ("e2e4", "e7e5", "g1f3", "g8f6"): "Petrov Defense",
    ("e2e4", "e7e5", "f1c4"): "Bishop's Opening",
    ("e2e4", "e7e5", "f2f4"): "King's Gambit",
    
    # Sicilian Defense
    ("e2e4", "c7c5"): "Sicilian Defense",
    ("e2e4", "c7c5", "g1f3"): "Open Sicilian",
    ("e2e4", "c7c5", "g1f3", "d7d6"): "Sicilian Najdorf/Dragon",
    ("e2e4", "c7c5", "g1f3", "b8c6"): "Sicilian Classical",
    ("e2e4", "c7c5", "g1f3", "e7e6"): "Sicilian Scheveningen",
    ("e2e4", "c7c5", "c2c3"): "Sicilian Alapin",
    
    # French Defense
    ("e2e4", "e7e6"): "French Defense",
    ("e2e4", "e7e6", "d2d4"): "French Defense",
    ("e2e4", "e7e6", "d2d4", "d7d5"): "French Defense Main Line",
    
    # Caro-Kann
    ("e2e4", "c7c6"): "Caro-Kann Defense",
    ("e2e4", "c7c6", "d2d4"): "Caro-Kann Defense",
    ("e2e4", "c7c6", "d2d4", "d7d5"): "Caro-Kann Main Line",
    
    # Scandinavian
    ("e2e4", "d7d5"): "Scandinavian Defense",
    
    # Pirc/Modern
    ("e2e4", "d7d6"): "Pirc Defense",
    ("e2e4", "g7g6"): "Modern Defense",
    
    # Queen's Pawn openings
    ("d2d4",): "Queen's Pawn",
    ("d2d4", "d7d5"): "Closed Game",
    ("d2d4", "d7d5", "c2c4"): "Queen's Gambit",
    ("d2d4", "d7d5", "c2c4", "e7e6"): "Queen's Gambit Declined",
    ("d2d4", "d7d5", "c2c4", "d5c4"): "Queen's Gambit Accepted",
    ("d2d4", "d7d5", "c2c4", "c7c6"): "Slav Defense",
    
    # Indian Defenses
    ("d2d4", "g8f6"): "Indian Defense",
    ("d2d4", "g8f6", "c2c4"): "Indian Game",
    ("d2d4", "g8f6", "c2c4", "e7e6"): "Nimzo/Queen's Indian",
    ("d2d4", "g8f6", "c2c4", "g7g6"): "King's Indian Defense",
    ("d2d4", "g8f6", "c2c4", "c7c5"): "Benoni Defense",
    ("d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"): "Nimzo-Indian Defense",
    ("d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"): "Queen's Indian Defense",
    
    # Dutch Defense
    ("d2d4", "f7f5"): "Dutch Defense",
    
    # English Opening
    ("c2c4",): "English Opening",
    ("c2c4", "e7e5"): "English Opening - Reversed Sicilian",
    ("c2c4", "c7c5"): "English Opening - Symmetrical",
    ("c2c4", "g8f6"): "English Opening",
    
    # Reti Opening
    ("g1f3",): "Reti Opening",
    ("g1f3", "d7d5"): "Reti Opening",
    ("g1f3", "d7d5", "c2c4"): "Reti Opening",
    
    # Other
    ("b2b3",): "Larsen's Opening",
    ("g2g3",): "Benko's Opening",
    ("f2f4",): "Bird's Opening",
}

def get_opening_name_with_prefix(move_history: List[str]) -> Tuple[str, int]:
    """
    Get the name of the opening based on move history and the matched prefix length.
    Returns (opening name, prefix length).
    """
    best_match = "Opening Theory"
    best_length = 0
    
    for length in range(1, len(move_history) + 1):
        prefix = tuple(move_history[:length])
        if prefix in OPENING_NAMES:
            best_match = OPENING_NAMES[prefix]
            best_length = length
    
    return best_match, best_length


def get_opening_name(move_history: List[str]) -> str:
    """
    Get the name of the opening based on move history.
    Returns the most specific matching opening name.
    """
    return get_opening_name_with_prefix(move_history)[0]


@dataclass
class TheoryPracticeGame:
    """
    Manages an opening theory practice session.
    
    The game pre-selects a random opening line from the book,
    displays the opening name, and tests the user's knowledge.
    Alternate book moves are allowed within the opening anchor.
    Incorrect moves result in losing a life.
    """
    book_path: str
    difficulty: DifficultyLevel
    player_color: int  # 0 = white, 1 = black
    
    # Internal state
    book: Optional[PolyglotBook] = field(default=None, init=False)
    squares: List[Optional[Tuple[int, int]]] = field(default_factory=list, init=False)
    side_to_move: int = field(default=0, init=False)
    castling_rights: int = field(default=0xF, init=False)
    ep_square: Optional[int] = field(default=None, init=False)
    move_history: List[str] = field(default_factory=list, init=False)
    lives: int = field(default=0, init=False)
    game_over: bool = field(default=False, init=False)
    current_opening_name: str = field(default="Opening Theory", init=False)
    opening_anchor_moves: List[str] = field(default_factory=list, init=False)
    opening_anchor_name: str = field(default="Opening Theory", init=False)
    lines_completed: int = field(default=0, init=False)
    total_correct_moves: int = field(default=0, init=False)
    rng: random.Random = field(default_factory=random.Random, init=False)
    
    # Pre-selected target line
    target_line: List[str] = field(default_factory=list, init=False)
    current_move_index: int = field(default=0, init=False)
    
    # Track played lines to avoid duplicates during a run
    played_lines: set = field(default_factory=set, init=False)
    
    def __post_init__(self):
        """Initialize the game state."""
        self.book = PolyglotBook(self.book_path)
        self.lives = DIFFICULTY_LIVES[self.difficulty]
        self.played_lines = set()
        self._reset_position()
    
    def _reset_position(self) -> None:
        """Reset to starting position."""
        self.squares = [None] * 64
        self._setup_startpos()
        self.side_to_move = 0  # White
        self.castling_rights = 0xF  # All castling rights
        self.ep_square = None
        self.move_history = []
        self.target_line = []
        self.current_move_index = 0
        self.current_opening_name = "Opening Theory"
        self.opening_anchor_moves = []
        self.opening_anchor_name = "Opening Theory"
    
    def _generate_random_line(self) -> List[str]:
        """
        Generate a complete random opening line by traversing the book.
        Uses weighted random selection at each position.
        Applies move depth limit based on difficulty.
        Avoids generating duplicate lines within the same run.
        Returns a list of UCI moves.
        """
        max_attempts = 100  # Prevent infinite loop if all lines have been played
        
        for _ in range(max_attempts):
            line = self._generate_single_line()
            
            # Convert to tuple for hashing
            line_tuple = tuple(line)
            
            # Check if this line has been played before
            if line_tuple not in self.played_lines:
                self.played_lines.add(line_tuple)
                return line
        
        # If we've exhausted attempts (very unlikely), clear history and return last generated
        self.played_lines.clear()
        line = self._generate_single_line()
        self.played_lines.add(tuple(line))
        return line

    def _generate_continuation_from_state(self, remaining_depth: int) -> List[str]:
        """
        Generate a random continuation line from the current position.
        """
        if remaining_depth <= 0 or self.book is None:
            return []
        
        line = []
        temp_squares = list(self.squares)
        temp_side = self.side_to_move
        temp_castling = self.castling_rights
        temp_ep = self.ep_square
        
        for _ in range(remaining_depth):
            key = compute_polyglot_hash(temp_squares, temp_side, temp_castling, temp_ep)
            move = self.book.get_weighted_random_move(key, self.rng)
            if move is None:
                break
            uci = move.to_uci()
            line.append(uci)
            temp_squares, temp_side, temp_castling, temp_ep = self._apply_move_to_state(
                uci, temp_squares, temp_side, temp_castling, temp_ep
            )
        
        return line

    def _refresh_target_line_from_current_position(self) -> None:
        """Rebuild the target line from the current position."""
        max_depth = self._get_line_depth_limit()
        remaining_depth = max(0, max_depth - self.current_move_index)
        continuation = self._generate_continuation_from_state(remaining_depth)
        self.target_line = self.move_history + continuation

    def _get_line_depth_limit(self) -> int:
        move_depth = DIFFICULTY_MOVE_DEPTH.get(self.difficulty)
        return move_depth if move_depth is not None else 40
    
    def _generate_single_line(self) -> List[str]:
        """
        Generate a single random opening line by traversing the book.
        Applies move depth limit based on difficulty.
        """
        line = []
        
        # Temporarily store state
        temp_squares = [None] * 64
        self._setup_startpos_into(temp_squares)
        temp_side = 0
        temp_castling = 0xF
        temp_ep = None
        
        max_depth = self._get_line_depth_limit()
        
        for _ in range(max_depth):
            # Compute hash for current position
            key = compute_polyglot_hash(temp_squares, temp_side, temp_castling, temp_ep)
            
            # Get book moves
            if self.book is None:
                break
            move = self.book.get_weighted_random_move(key, self.rng)
            if move is None:
                break
            
            uci = move.to_uci()
            line.append(uci)
            
            # Apply move to temp state
            temp_squares, temp_side, temp_castling, temp_ep = self._apply_move_to_state(
                uci, temp_squares, temp_side, temp_castling, temp_ep
            )
        
        return line
    
    def _setup_startpos_into(self, squares: List[Optional[Tuple[int, int]]]) -> None:
        """Set up the starting position into the given squares list."""
        for i in range(64):
            squares[i] = None
        
        # White pieces (color=0)
        squares[algebraic_to_square("a1")] = (0, 4)  # Rook
        squares[algebraic_to_square("b1")] = (0, 2)  # Knight
        squares[algebraic_to_square("c1")] = (0, 3)  # Bishop
        squares[algebraic_to_square("d1")] = (0, 5)  # Queen
        squares[algebraic_to_square("e1")] = (0, 6)  # King
        squares[algebraic_to_square("f1")] = (0, 3)  # Bishop
        squares[algebraic_to_square("g1")] = (0, 2)  # Knight
        squares[algebraic_to_square("h1")] = (0, 4)  # Rook
        for f in range(8):
            squares[square_from_file_rank(f, 1)] = (0, 1)  # Pawns
        
        # Black pieces (color=1)
        squares[algebraic_to_square("a8")] = (1, 4)  # Rook
        squares[algebraic_to_square("b8")] = (1, 2)  # Knight
        squares[algebraic_to_square("c8")] = (1, 3)  # Bishop
        squares[algebraic_to_square("d8")] = (1, 5)  # Queen
        squares[algebraic_to_square("e8")] = (1, 6)  # King
        squares[algebraic_to_square("f8")] = (1, 3)  # Bishop
        squares[algebraic_to_square("g8")] = (1, 2)  # Knight
        squares[algebraic_to_square("h8")] = (1, 4)  # Rook
        for f in range(8):
            squares[square_from_file_rank(f, 6)] = (1, 1)  # Pawns
    
    def _apply_move_to_state(
        self,
        uci_move: str,
        squares: List[Optional[Tuple[int, int]]],
        side_to_move: int,
        castling_rights: int,
        ep_square: Optional[int]
    ) -> Tuple[List[Optional[Tuple[int, int]]], int, int, Optional[int]]:
        """Apply a move to the given state and return the new state."""
        from_sq = algebraic_to_square(uci_move[0:2])
        to_sq = algebraic_to_square(uci_move[2:4])
        promotion = None
        if len(uci_move) == 5:
            promo_map = {'n': 2, 'b': 3, 'r': 4, 'q': 5}
            promotion = promo_map.get(uci_move[4].lower())
        
        piece = squares[from_sq]
        if piece is None:
            return squares, 1 - side_to_move, castling_rights, None
        
        color, piece_type = piece
        from_rank = from_sq // 8
        from_file = from_sq % 8
        to_rank = to_sq // 8
        to_file = to_sq % 8
        
        # Handle castling
        if piece_type == 6:  # King
            if from_file == 4 and to_file == 6:
                rook_from = from_rank * 8 + 7
                rook_to = from_rank * 8 + 5
                squares[rook_to] = squares[rook_from]
                squares[rook_from] = None
            elif from_file == 4 and to_file == 2:
                rook_from = from_rank * 8 + 0
                rook_to = from_rank * 8 + 3
                squares[rook_to] = squares[rook_from]
                squares[rook_from] = None
            
            if color == 0:
                castling_rights &= ~0x3
            else:
                castling_rights &= ~0xC
        
        # Handle rook moves
        if piece_type == 4:  # Rook
            if color == 0:
                if from_sq == algebraic_to_square("a1"):
                    castling_rights &= ~0x2
                elif from_sq == algebraic_to_square("h1"):
                    castling_rights &= ~0x1
            else:
                if from_sq == algebraic_to_square("a8"):
                    castling_rights &= ~0x8
                elif from_sq == algebraic_to_square("h8"):
                    castling_rights &= ~0x4
        
        # Handle en passant capture
        if piece_type == 1 and ep_square is not None:
            if to_sq == ep_square:
                captured_rank = from_rank
                captured_sq = captured_rank * 8 + to_file
                squares[captured_sq] = None
        
        # Update en passant square
        new_ep = None
        if piece_type == 1:
            if abs(to_rank - from_rank) == 2:
                ep_rank = (from_rank + to_rank) // 2
                new_ep = ep_rank * 8 + from_file
        
        # Make the move
        squares[from_sq] = None
        if promotion is not None:
            squares[to_sq] = (color, promotion)
        else:
            squares[to_sq] = piece
        
        return squares, 1 - side_to_move, castling_rights, new_ep

    def _setup_startpos(self) -> None:
        """Set up the starting position."""
        self._setup_startpos_into(self.squares)
    
    def get_current_hash(self) -> int:
        """Get the Polyglot hash for the current position."""
        return compute_polyglot_hash(
            self.squares,
            self.side_to_move,
            self.castling_rights,
            self.ep_square
        )
    
    def get_book_moves(self) -> List[Tuple[BookMove, int]]:
        """Get all book moves for the current position with weights."""
        if self.book is None:
            return []
        key = self.get_current_hash()
        return self.book.get_all_moves(key)
    
    def is_in_book(self) -> bool:
        """Check if the current position is in the opening book."""
        if self.book is None:
            return False
        key = self.get_current_hash()
        return self.book.contains(key)

    def _is_book_move(self, uci_move: str) -> bool:
        """Check if the given move exists in the opening book from the current position."""
        for move, _ in self.get_book_moves():
            if move.to_uci() == uci_move:
                return True
        return False

    def _matches_opening_anchor(self, move_history: List[str]) -> bool:
        """Check if the move history stays within the opening anchor prefix."""
        if not self.opening_anchor_moves:
            return True
        anchor_len = len(self.opening_anchor_moves)
        if len(move_history) <= anchor_len:
            return move_history == self.opening_anchor_moves[:len(move_history)]
        return move_history[:anchor_len] == self.opening_anchor_moves

    def _update_opening_name_from_history(self) -> None:
        """Update the displayed opening name from the current move history."""
        if self.opening_anchor_moves and len(self.move_history) < len(self.opening_anchor_moves):
            self.current_opening_name = self.opening_anchor_name
            return
        self.current_opening_name = get_opening_name(self.move_history)
    
    def is_correct_move(self, uci_move: str) -> bool:
        """Check if a move matches the expected move in the target line."""
        if self.current_move_index >= len(self.target_line):
            return False
        expected_move = self.target_line[self.current_move_index]
        return uci_move == expected_move
    
    def get_expected_move(self) -> Optional[str]:
        """Get the expected move at the current position in the target line."""
        if self.current_move_index >= len(self.target_line):
            return None
        return self.target_line[self.current_move_index]
    
    def is_line_complete(self) -> bool:
        """Check if we've reached the end of the target line."""
        return self.current_move_index >= len(self.target_line)
    
    def is_legal_move(self, uci_move: str) -> bool:
        """
        Check if a move is legal (valid chess move, not necessarily the book move).
        This is a basic validation to detect obviously illegal moves.
        """
        if len(uci_move) < 4:
            return False
        
        try:
            from_sq = algebraic_to_square(uci_move[0:2])
            to_sq = algebraic_to_square(uci_move[2:4])
        except (ValueError, IndexError):
            return False
        
        if from_sq < 0 or from_sq >= 64 or to_sq < 0 or to_sq >= 64:
            return False
        
        if from_sq == to_sq:
            return False
        
        piece = self.squares[from_sq]
        if piece is None:
            return False
        
        color, piece_type = piece
        
        # Check if it's the right color's turn
        if color != self.side_to_move:
            return False
        
        # Check destination isn't occupied by own piece
        dest_piece = self.squares[to_sq]
        if dest_piece is not None and dest_piece[0] == color:
            return False
        
        from_rank = from_sq // 8
        from_file = from_sq % 8
        to_rank = to_sq // 8
        to_file = to_sq % 8
        rank_diff = to_rank - from_rank
        file_diff = to_file - from_file
        
        # Basic piece movement validation
        if piece_type == 1:  # Pawn
            direction = 1 if color == 0 else -1
            start_rank = 1 if color == 0 else 6
            
            # Forward moves
            if file_diff == 0:
                if rank_diff == direction:
                    if dest_piece is not None:
                        return False  # Can't capture forward
                elif rank_diff == 2 * direction:
                    if from_rank != start_rank:
                        return False  # Double push only from start
                    # Check path is clear
                    between_sq = from_sq + 8 * direction
                    if self.squares[between_sq] is not None:
                        return False
                    if dest_piece is not None:
                        return False
                else:
                    return False
            # Captures (including en passant)
            elif abs(file_diff) == 1 and rank_diff == direction:
                if dest_piece is None and to_sq != self.ep_square:
                    return False  # Must capture something or be en passant
            else:
                return False
                
        elif piece_type == 2:  # Knight
            if not ((abs(rank_diff) == 2 and abs(file_diff) == 1) or
                    (abs(rank_diff) == 1 and abs(file_diff) == 2)):
                return False
                
        elif piece_type == 3:  # Bishop
            if abs(rank_diff) != abs(file_diff) or rank_diff == 0:
                return False
            # Check path is clear
            if not self._is_path_clear(from_sq, to_sq):
                return False
                
        elif piece_type == 4:  # Rook
            if rank_diff != 0 and file_diff != 0:
                return False
            if rank_diff == 0 and file_diff == 0:
                return False
            # Check path is clear
            if not self._is_path_clear(from_sq, to_sq):
                return False
                
        elif piece_type == 5:  # Queen
            is_diagonal = abs(rank_diff) == abs(file_diff) and rank_diff != 0
            is_straight = (rank_diff == 0) != (file_diff == 0)
            if not (is_diagonal or is_straight):
                return False
            # Check path is clear
            if not self._is_path_clear(from_sq, to_sq):
                return False
                
        elif piece_type == 6:  # King
            # Normal move
            if max(abs(rank_diff), abs(file_diff)) == 1:
                pass  # Valid
            # Castling
            elif from_file == 4 and abs(file_diff) == 2 and rank_diff == 0:
                # Basic castling check (not full validation)
                pass
            else:
                return False
        
        # Check if move would leave king in check (simplified)
        # We'll do a basic check by simulating the move
        if not self._move_leaves_king_safe(from_sq, to_sq, piece):
            return False
        
        return True
    
    def _is_path_clear(self, from_sq: int, to_sq: int) -> bool:
        """Check if the path between two squares is clear (for sliding pieces)."""
        from_rank = from_sq // 8
        from_file = from_sq % 8
        to_rank = to_sq // 8
        to_file = to_sq % 8
        
        rank_step = 0 if to_rank == from_rank else (1 if to_rank > from_rank else -1)
        file_step = 0 if to_file == from_file else (1 if to_file > from_file else -1)
        
        current_rank = from_rank + rank_step
        current_file = from_file + file_step
        
        while current_rank != to_rank or current_file != to_file:
            sq = current_rank * 8 + current_file
            if self.squares[sq] is not None:
                return False
            current_rank += rank_step
            current_file += file_step
        
        return True
    
    def _move_leaves_king_safe(self, from_sq: int, to_sq: int, piece: Tuple[int, int]) -> bool:
        """Check if making a move would leave the king in check (basic check)."""
        color = piece[0]
        
        # Temporarily make the move
        original_from = self.squares[from_sq]
        original_to = self.squares[to_sq]
        self.squares[from_sq] = None
        self.squares[to_sq] = piece
        
        # Find king
        king_sq = None
        for sq in range(64):
            p = self.squares[sq]
            if p is not None and p[0] == color and p[1] == 6:
                king_sq = sq
                break
        
        if king_sq is None:
            # Restore and return True (shouldn't happen)
            self.squares[from_sq] = original_from
            self.squares[to_sq] = original_to
            return True
        
        # Check if king is attacked
        is_safe = not self._is_square_attacked(king_sq, 1 - color)
        
        # Restore position
        self.squares[from_sq] = original_from
        self.squares[to_sq] = original_to
        
        return is_safe
    
    def _is_square_attacked(self, square: int, by_color: int) -> bool:
        """Check if a square is attacked by the given color."""
        rank = square // 8
        file = square % 8
        
        # Check pawn attacks
        pawn_dir = -1 if by_color == 0 else 1  # Direction pawns attack FROM
        for df in [-1, 1]:
            attack_rank = rank + pawn_dir
            attack_file = file + df
            if 0 <= attack_rank < 8 and 0 <= attack_file < 8:
                sq = attack_rank * 8 + attack_file
                p = self.squares[sq]
                if p is not None and p[0] == by_color and p[1] == 1:
                    return True
        
        # Check knight attacks
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        for dr, df in knight_moves:
            r, f = rank + dr, file + df
            if 0 <= r < 8 and 0 <= f < 8:
                sq = r * 8 + f
                p = self.squares[sq]
                if p is not None and p[0] == by_color and p[1] == 2:
                    return True
        
        # Check king attacks
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                if dr == 0 and df == 0:
                    continue
                r, f = rank + dr, file + df
                if 0 <= r < 8 and 0 <= f < 8:
                    sq = r * 8 + f
                    p = self.squares[sq]
                    if p is not None and p[0] == by_color and p[1] == 6:
                        return True
        
        # Check sliding piece attacks (rook, bishop, queen)
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # Rook/Queen
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # Bishop/Queen
        ]
        for i, (dr, df) in enumerate(directions):
            r, f = rank + dr, file + df
            while 0 <= r < 8 and 0 <= f < 8:
                sq = r * 8 + f
                p = self.squares[sq]
                if p is not None:
                    if p[0] == by_color:
                        # Rook or Queen for straight, Bishop or Queen for diagonal
                        if i < 4:  # Straight
                            if p[1] in (4, 5):  # Rook or Queen
                                return True
                        else:  # Diagonal
                            if p[1] in (3, 5):  # Bishop or Queen
                                return True
                    break
                r += dr
                f += df
        
        return False
    
    def make_move(self, uci_move: str) -> None:
        """
        Make a move on the board (without validation).
        Updates position, castling rights, en passant, etc.
        """
        from_sq = algebraic_to_square(uci_move[0:2])
        to_sq = algebraic_to_square(uci_move[2:4])
        promotion = None
        if len(uci_move) == 5:
            promo_map = {'n': 2, 'b': 3, 'r': 4, 'q': 5}
            promotion = promo_map.get(uci_move[4].lower())
        
        piece = self.squares[from_sq]
        if piece is None:
            return
        
        color, piece_type = piece
        from_rank = from_sq // 8
        from_file = from_sq % 8
        to_rank = to_sq // 8
        to_file = to_sq % 8
        
        # Handle castling
        if piece_type == 6:  # King
            # Kingside castling
            if from_file == 4 and to_file == 6:
                rook_from = from_rank * 8 + 7
                rook_to = from_rank * 8 + 5
                self.squares[rook_to] = self.squares[rook_from]
                self.squares[rook_from] = None
            # Queenside castling
            elif from_file == 4 and to_file == 2:
                rook_from = from_rank * 8 + 0
                rook_to = from_rank * 8 + 3
                self.squares[rook_to] = self.squares[rook_from]
                self.squares[rook_from] = None
            
            # Update castling rights
            if color == 0:  # White
                self.castling_rights &= ~0x3  # Remove white castling
            else:  # Black
                self.castling_rights &= ~0xC  # Remove black castling
        
        # Handle rook moves (update castling rights)
        if piece_type == 4:  # Rook
            if color == 0:  # White
                if from_sq == algebraic_to_square("a1"):
                    self.castling_rights &= ~0x2  # Remove white queenside
                elif from_sq == algebraic_to_square("h1"):
                    self.castling_rights &= ~0x1  # Remove white kingside
            else:  # Black
                if from_sq == algebraic_to_square("a8"):
                    self.castling_rights &= ~0x8  # Remove black queenside
                elif from_sq == algebraic_to_square("h8"):
                    self.castling_rights &= ~0x4  # Remove black kingside
        
        # Handle en passant capture
        if piece_type == 1 and self.ep_square is not None:  # Pawn
            if to_sq == self.ep_square:
                # Remove captured pawn
                captured_rank = from_rank  # Same rank as capturing pawn
                captured_sq = captured_rank * 8 + to_file
                self.squares[captured_sq] = None
        
        # Update en passant square
        self.ep_square = None
        if piece_type == 1:  # Pawn
            if abs(to_rank - from_rank) == 2:
                # Double pawn push - set EP square
                ep_rank = (from_rank + to_rank) // 2
                self.ep_square = ep_rank * 8 + from_file
        
        # Make the move
        self.squares[from_sq] = None
        if promotion is not None:
            self.squares[to_sq] = (color, promotion)
        else:
            self.squares[to_sq] = piece
        
        # Update side to move
        self.side_to_move = 1 - self.side_to_move
        
        # Update move history
        self.move_history.append(uci_move)
        self._update_opening_name_from_history()
    
    def try_player_move(self, uci_move: str) -> Optional[Tuple[bool, str]]:
        """
        Try a player move.
        
        Returns:
            None if move is illegal (piece should snap back silently)
            (success, message) tuple otherwise:
            - success: True if move was correct (matches target line or allowed alternate)
            - message: Feedback message
        """
        if self.game_over:
            return False, "Game over! Press Reset to try again."
        
        if self.side_to_move != self.player_color:
            return None  # Not your turn - illegal
        
        if self.is_line_complete():
            return None  # Line complete - no more moves
        
        # First check if the move is legal at all
        if not self.is_legal_move(uci_move):
            return None  # Illegal move - snap back silently
        
        # Check if this matches the expected move in the target line
        if self.is_correct_move(uci_move):
            self.make_move(uci_move)
            self.current_move_index += 1
            self.total_correct_moves += 1
            
            # Check if we've reached the end of the line
            if self.is_line_complete():
                self.lines_completed += 1
                return True, f"Correct! Line complete. ({self.lines_completed} lines)"
            
            return True, "Correct!"
        else:
            prospective_history = self.move_history + [uci_move]
            if self._is_book_move(uci_move) and self._matches_opening_anchor(prospective_history):
                self.make_move(uci_move)
                self.current_move_index += 1
                self.total_correct_moves += 1
                self._refresh_target_line_from_current_position()
                
                if self.is_line_complete():
                    self.lines_completed += 1
                    return True, f"Correct! Line complete. ({self.lines_completed} lines)"
                
                return True, "Correct!"
            
            # Wrong book move (but legal) - lose a life
            if self.lives > 0:
                self.lives -= 1
            
            if self.lives == 0 and self.difficulty != DifficultyLevel.INFINITE:
                self.game_over = True
                return False, f"Wrong! Game over. Completed {self.lines_completed} lines."
            
            # Show the expected move as a hint
            expected = self.get_expected_move()
            if expected:
                if self.lives > 0:
                    return False, f"Wrong! Expected: {expected}. Lives: {self.lives}"
                else:
                    return False, f"Wrong! Expected: {expected}. (Infinite mode)"
            else:
                return False, "Wrong! Line complete."
    
    def make_opponent_move(self) -> Optional[str]:
        """
        Make the opponent's move from the target line.
        
        Returns:
            The move made (UCI format), or None if end of line or player's turn.
        """
        if self.game_over:
            return None
        
        if self.side_to_move == self.player_color:
            return None
        
        if self.is_line_complete():
            return None
        
        move = self.get_expected_move()
        if move is None:
            return None
        
        self.make_move(move)
        self.current_move_index += 1
        return move
    
    def start_new_line(self) -> None:
        """Start a new opening line from the beginning."""
        if self.game_over:
            return
        
        self._reset_position()
        
        # Generate a random complete opening line
        self.target_line = self._generate_random_line()
        self.current_move_index = 0
        
        opening_name, opening_length = get_opening_name_with_prefix(self.target_line)
        self.opening_anchor_moves = self.target_line[:opening_length]
        self.opening_anchor_name = opening_name
        self.current_opening_name = opening_name
        
        # If player is black, make white's first move(s)
        while self.side_to_move != self.player_color and not self.is_line_complete():
            self.make_opponent_move()
    
    def reset_game(self) -> None:
        """Reset the entire game (including lives and played lines history)."""
        self.lives = DIFFICULTY_LIVES[self.difficulty]
        self.game_over = False
        self.lines_completed = 0
        self.total_correct_moves = 0
        self.played_lines.clear()  # Clear played lines for fresh start
        self.start_new_line()
    
    def get_lives_display(self) -> str:
        """Get a display string for remaining lives."""
        if self.difficulty == DifficultyLevel.INFINITE:
            return "∞"
        return str(self.lives)
    
    def get_score_display(self) -> str:
        """Get a display string for the current score."""
        return f"Lines: {self.lines_completed} | Moves: {self.total_correct_moves}"
