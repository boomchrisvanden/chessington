"""
Tests for the opening theory practice module.

Verifies:
1. Game initialization
2. Opening name detection
3. Book move validation
4. Lives/difficulty system
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from theory.practice import (
    TheoryPracticeGame,
    DifficultyLevel,
    DIFFICULTY_LIVES,
    DIFFICULTY_MOVE_DEPTH,
    find_opening_name_matches,
    get_opening_name,
    get_opening_names,
    normalize_opening_name,
    resolve_opening_name,
)


class TestOpeningNames:
    """Test opening name detection."""
    
    def test_kings_pawn(self):
        assert get_opening_name(["e2e4"]) == "King's Pawn"
    
    def test_open_game(self):
        assert get_opening_name(["e2e4", "e7e5"]) == "Open Game"
    
    def test_ruy_lopez(self):
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
        assert get_opening_name(moves) == "Ruy Lopez"
    
    def test_italian_game(self):
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]
        assert get_opening_name(moves) == "Italian Game"
    
    def test_sicilian_defense(self):
        assert get_opening_name(["e2e4", "c7c5"]) == "Sicilian Defense"
    
    def test_queens_gambit(self):
        moves = ["d2d4", "d7d5", "c2c4"]
        assert get_opening_name(moves) == "Queen's Gambit"
    
    def test_unknown_opening(self):
        # Unusual moves should return default
        assert get_opening_name([]) == "Opening Theory"
        assert get_opening_name(["a2a3"]) == "Opening Theory"


class TestDifficultyLives:
    """Test difficulty level lives mapping."""
    
    def test_infinite_lives(self):
        assert DIFFICULTY_LIVES[DifficultyLevel.INFINITE] == -1
    
    def test_easy_lives(self):
        assert DIFFICULTY_LIVES[DifficultyLevel.EASY] == 10
    
    def test_medium_lives(self):
        assert DIFFICULTY_LIVES[DifficultyLevel.MEDIUM] == 5
    
    def test_hard_lives(self):
        assert DIFFICULTY_LIVES[DifficultyLevel.HARD] == 3
    
    def test_insane_lives(self):
        assert DIFFICULTY_LIVES[DifficultyLevel.INSANE] == 1


class TestOpeningSearch:
    """Test opening search helpers."""
    
    def test_opening_names_contains_known(self):
        names = get_opening_names()
        assert "Slav Defense" in names
        assert "Sicilian Defense" in names
    
    def test_normalize_opening_name(self):
        assert normalize_opening_name("King's Pawn") == "kings pawn"
    
    def test_find_opening_name_matches(self):
        matches = find_opening_name_matches("slav")
        assert "Slav Defense" in matches
        matches = find_opening_name_matches("sicillian")
        assert "Sicilian Defense" in matches
    
    def test_resolve_opening_name(self):
        assert resolve_opening_name("slav defense") == "Slav Defense"


class TestTheoryPracticeGame:
    """Test the TheoryPracticeGame class."""
    
    @pytest.fixture
    def book_path(self):
        """Get path to the main opening book."""
        return os.path.join(os.path.dirname(__file__), '..', 'src', 'Book.bin')
    
    @pytest.fixture
    def test_book_path(self):
        """Get path to the test book."""
        return os.path.join(os.path.dirname(__file__), 'test_book.bin')
    
    def test_game_initialization(self, book_path):
        """Test that game initializes correctly."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.MEDIUM,
            player_color=0
        )
        
        assert game.lives == 5
        assert game.player_color == 0
        assert game.game_over is False
        assert game.lines_completed == 0
    
    def test_starting_position_in_book(self, book_path):
        """Test that starting position is in the book."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.EASY,
            player_color=0
        )
        
        assert game.is_in_book()
        book_moves = game.get_book_moves()
        assert len(book_moves) > 0
    
    def test_target_line_generated(self, book_path):
        """Test that a target line is generated when starting a new line."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.EASY,
            player_color=0
        )
        
        game.start_new_line()
        assert len(game.target_line) > 0
        assert game.current_opening_name != "Opening Theory"

    def test_start_new_line_for_opening(self, book_path):
        """Test that a target line is generated for a specific opening name."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.EASY,
            player_color=0
        )
        
        started = game.start_new_line_for_opening("King's Pawn")
        assert started
        assert game.opening_anchor_name == "King's Pawn"
        assert game.opening_anchor_moves[:1] == ["e2e4"]
    
    def test_correct_move_increments_score(self, book_path):
        """Test that correct moves increment the score."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.EASY,
            player_color=0
        )
        
        game.start_new_line()
        initial_moves = game.total_correct_moves
        
        # Get the expected move and play it
        expected_move = game.get_expected_move()
        assert expected_move is not None
        
        success, _ = game.try_player_move(expected_move)
        
        assert success
        assert game.total_correct_moves == initial_moves + 1
    
    def test_wrong_move_decrements_lives(self, book_path):
        """Test that wrong moves decrement lives."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.MEDIUM,
            player_color=0
        )
        
        game.start_new_line()
        initial_lives = game.lives
        
        # Make a move that is legal but not in the book from the starting position
        expected_move = game.get_expected_move()
        book_moves = {move.to_uci() for move, _ in game.get_book_moves()}
        candidates = [
            "a2a3", "a2a4", "b2b3", "b2b4", "c2c3", "c2c4",
            "d2d3", "d2d4", "e2e3", "e2e4", "f2f3", "f2f4",
            "g2g3", "g2g4", "h2h3", "h2h4", "b1a3", "b1c3",
            "g1f3", "g1h3",
        ]
        wrong_move = None
        for candidate in candidates:
            if candidate != expected_move and candidate not in book_moves:
                wrong_move = candidate
                break
        if wrong_move is None:
            pytest.skip("No non-book legal move available to test wrong-move handling")
        
        success, _ = game.try_player_move(wrong_move)
        
        assert not success
        assert game.lives == initial_lives - 1
    
    def test_infinite_mode_never_game_over(self, book_path):
        """Test that infinite mode never ends from wrong moves."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.INFINITE,
            player_color=0
        )
        
        # Make several wrong moves
        for _ in range(20):
            game.try_player_move("h2h3")  # Unlikely book move
        
        # Game should never be over in infinite mode
        assert game.game_over is False
    
    def test_game_reset(self, book_path):
        """Test that reset restores initial state."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.EASY,
            player_color=0
        )
        
        # Make some moves and modify state
        game.try_player_move("e2e4")
        game.lines_completed = 5
        
        # Reset
        game.reset_game()
        
        assert game.lives == 10
        assert game.lines_completed == 0
        assert game.total_correct_moves == 0
        assert game.game_over is False
    
    def test_player_color_black(self, book_path):
        """Test game with player as black."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.MEDIUM,
            player_color=1
        )
        
        assert game.player_color == 1
        # After starting a new line, white should have moved first
        game.start_new_line()
        # Side to move should now be black (player's turn)
        assert game.side_to_move == 1
    
    def test_move_depth_easy(self, book_path):
        """Test that Easy difficulty limits lines to 10 moves (5 per side)."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.EASY,
            player_color=0
        )
        
        game.start_new_line()
        assert len(game.target_line) <= 10
    
    def test_move_depth_medium(self, book_path):
        """Test that Medium difficulty limits lines to 20 moves (10 per side)."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.MEDIUM,
            player_color=0
        )
        
        game.start_new_line()
        assert len(game.target_line) <= 20
    
    def test_move_depth_hard(self, book_path):
        """Test that Hard difficulty limits lines to 30 moves (15 per side)."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.HARD,
            player_color=0
        )
        
        game.start_new_line()
        assert len(game.target_line) <= 30
    
    def test_no_duplicate_lines(self, book_path):
        """Test that lines are not repeated during a run."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.EASY,  # Short lines = more likely collision
            player_color=0
        )
        
        lines_seen = set()
        for _ in range(15):
            game.start_new_line()
            line_tuple = tuple(game.target_line)
            assert line_tuple not in lines_seen, "Duplicate line generated"
            lines_seen.add(line_tuple)
    
    def test_illegal_move_returns_none(self, book_path):
        """Test that illegal moves return None (no life loss)."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.MEDIUM,
            player_color=0
        )
        
        game.start_new_line()
        initial_lives = game.lives
        
        # Try an obviously illegal move (pawn 3 squares)
        result = game.try_player_move("e2e5")
        
        assert result is None
        assert game.lives == initial_lives  # No life lost
    
    def test_legal_wrong_move_costs_life(self, book_path):
        """Test that legal but wrong moves cost a life."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.MEDIUM,
            player_color=0
        )
        
        game.start_new_line()
        initial_lives = game.lives
        
        # Find a legal move that's not the expected one
        expected = game.get_expected_move()
        wrong_move = "a2a3" if expected != "a2a3" else "h2h3"
        
        result = game.try_player_move(wrong_move)
        
        assert result is not None
        assert result[0] is False  # Not successful
        assert game.lives == initial_lives - 1  # Life lost
    
    def test_reset_clears_played_lines(self, book_path):
        """Test that reset clears the played lines history."""
        if not os.path.exists(book_path):
            pytest.skip("Main book file not found")
        
        game = TheoryPracticeGame(
            book_path=book_path,
            difficulty=DifficultyLevel.EASY,
            player_color=0
        )
        
        # Generate some lines
        for _ in range(5):
            game.start_new_line()
        
        assert len(game.played_lines) == 5
        
        # Reset should clear history
        game.reset_game()
        
        # After reset, played_lines should have 1 entry (the new line)
        assert len(game.played_lines) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
