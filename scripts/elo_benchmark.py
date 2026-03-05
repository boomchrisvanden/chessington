#!/usr/bin/env python3
"""
Elo benchmark: play Chessington against Stockfish at various strength levels.

Uses python-chess for UCI communication. Requires stockfish to be installed
and accessible via PATH (or pass --stockfish-path).

Usage:
    python3 scripts/elo_benchmark.py                        # defaults
    python3 scripts/elo_benchmark.py --elo 1000 1200 1500   # specific levels
    python3 scripts/elo_benchmark.py --rounds 20 --tc 30    # 20 games, 30s/side
    python3 scripts/elo_benchmark.py --stockfish-path /usr/local/bin/stockfish
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import chess
import chess.engine
import chess.pgn

# ── Elo maths ────────────────────────────────────────────────────────────────

def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_diff_from_score(score: float) -> Optional[float]:
    """Elo difference implied by an actual score (0-1). None if 0 or 1."""
    if score <= 0.0 or score >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_error_margin(wins: int, draws: int, losses: int, confidence: float = 0.95) -> Optional[float]:
    """Approximate 95% confidence interval for Elo difference."""
    n = wins + draws + losses
    if n == 0:
        return None
    w = wins / n
    d = draws / n
    l = losses / n  # noqa: E741
    score = w + 0.5 * d
    if score <= 0.0 or score >= 1.0:
        return None
    # Variance of score
    var = (w * (1.0 - score) ** 2 + d * (0.5 - score) ** 2 + l * (0.0 - score) ** 2) / n
    if var <= 0:
        return None
    std = math.sqrt(var)
    # z for 95% CI
    z = 1.96
    score_lo = max(0.001, score - z * std)
    score_hi = min(0.999, score + z * std)
    elo_lo = elo_diff_from_score(score_lo)
    elo_hi = elo_diff_from_score(score_hi)
    if elo_lo is None or elo_hi is None:
        return None
    return (elo_hi - elo_lo) / 2.0


# ── Opening positions ────────────────────────────────────────────────────────

# A small set of diverse openings to reduce variance. Each is (name, FEN after moves).
OPENINGS = [
    ("Startpos",
     chess.STARTING_FEN),
    ("Italian Game",
     "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("Sicilian Najdorf",
     "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ("Queen's Gambit Declined",
     "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4"),
    ("Ruy Lopez",
     "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"),
    ("French Defence",
     "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ("Caro-Kann",
     "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ("King's Indian",
     "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"),
    ("English Opening",
     "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1"),
    ("Scandinavian",
     "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"),
]


# ── Match runner ─────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    sf_elo: int
    wins: int = 0
    draws: int = 0
    losses: int = 0
    games: list = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        n = self.total
        if n == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / n

    @property
    def elo_estimate(self) -> Optional[float]:
        diff = elo_diff_from_score(self.score)
        if diff is None:
            return None
        return self.sf_elo + diff

    def summary(self) -> str:
        est = self.elo_estimate
        margin = elo_error_margin(self.wins, self.draws, self.losses)
        est_str = f"{est:.0f}" if est is not None else "N/A"
        margin_str = f" +/- {margin:.0f}" if margin is not None else ""
        return (
            f"vs SF {self.sf_elo:>4d}: "
            f"+{self.wins} ={self.draws} -{self.losses}  "
            f"score {self.score:.1%}  "
            f"=> Elo ~{est_str}{margin_str}"
        )


def play_game(
    engine_ours: chess.engine.SimpleEngine,
    engine_sf: chess.engine.SimpleEngine,
    fen: str,
    time_limit: float,
    our_color: chess.Color,
) -> tuple[str, Optional[chess.pgn.Game]]:
    """
    Play one game. Returns ("1-0", "0-1", or "1/2-1/2", pgn_game).
    """
    board = chess.Board(fen)
    game = chess.pgn.Game()
    game.setup(board)
    node = game

    move_count = 0
    max_moves = 300  # safety limit

    while not board.is_game_over(claim_draw=True) and move_count < max_moves:
        if board.turn == our_color:
            engine = engine_ours
        else:
            engine = engine_sf

        result = engine.play(board, chess.engine.Limit(time=time_limit))
        if result.move is None:
            break
        node = node.add_variation(result.move)
        board.push(result.move)
        move_count += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result_str = "1/2-1/2"
    else:
        result_str = outcome.result()

    game.headers["Result"] = result_str
    return result_str, game


def run_match(
    engine_cmd: List[str],
    sf_path: str,
    sf_elo: int,
    rounds: int,
    time_per_move: float,
    pgn_dir: Optional[Path],
) -> MatchResult:
    """Run a match of *rounds* games at the given Stockfish Elo."""
    result = MatchResult(sf_elo=sf_elo)

    engine_ours = chess.engine.SimpleEngine.popen_uci(engine_cmd)
    engine_sf = chess.engine.SimpleEngine.popen_uci(sf_path)

    # Configure Stockfish strength.
    engine_sf.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
    # Disable own book on our side so search is always exercised.
    engine_ours.configure({"OwnBook": False})

    all_pgn_games: list = []

    try:
        for i in range(rounds):
            opening_name, fen = OPENINGS[i % len(OPENINGS)]
            # Alternate colors each round.
            our_color = chess.WHITE if i % 2 == 0 else chess.BLACK

            color_str = "W" if our_color == chess.WHITE else "B"
            print(f"  Game {i + 1}/{rounds} ({color_str}) [{opening_name}] ... ", end="", flush=True)

            game_result, pgn_game = play_game(
                engine_ours, engine_sf, fen, time_per_move, our_color,
            )

            if pgn_game is not None:
                pgn_game.headers["White"] = "Chessington" if our_color == chess.WHITE else f"Stockfish_{sf_elo}"
                pgn_game.headers["Black"] = f"Stockfish_{sf_elo}" if our_color == chess.WHITE else "Chessington"
                pgn_game.headers["Event"] = f"Elo Benchmark vs SF {sf_elo}"
                all_pgn_games.append(pgn_game)

            # Score from our perspective.
            if our_color == chess.WHITE:
                if game_result == "1-0":
                    result.wins += 1
                    print("WIN")
                elif game_result == "0-1":
                    result.losses += 1
                    print("LOSS")
                else:
                    result.draws += 1
                    print("DRAW")
            else:
                if game_result == "0-1":
                    result.wins += 1
                    print("WIN")
                elif game_result == "1-0":
                    result.losses += 1
                    print("LOSS")
                else:
                    result.draws += 1
                    print("DRAW")
    finally:
        engine_ours.quit()
        engine_sf.quit()

    # Save PGN.
    if pgn_dir and all_pgn_games:
        pgn_dir.mkdir(parents=True, exist_ok=True)
        pgn_file = pgn_dir / f"vs_sf_{sf_elo}.pgn"
        with open(pgn_file, "w") as f:
            for g in all_pgn_games:
                print(g, file=f)
                print(file=f)
        print(f"  PGN saved to {pgn_file}")

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate Chessington's Elo by playing against Stockfish.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example:
              python3 scripts/elo_benchmark.py --elo 1000 1200 1500 --rounds 20
        """),
    )
    parser.add_argument(
        "--elo", type=int, nargs="+",
        default=[800, 1000, 1200, 1400, 1600],
        help="Stockfish Elo levels to test against (default: 800 1000 1200 1400 1600)",
    )
    parser.add_argument(
        "--rounds", type=int, default=10,
        help="Number of games per Elo level (default: 10)",
    )
    parser.add_argument(
        "--tc", type=float, default=5.0,
        help="Time per move in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--stockfish-path", type=str, default="stockfish",
        help="Path to Stockfish binary (default: 'stockfish')",
    )
    parser.add_argument(
        "--pgn-dir", type=str, default="benchmark_pgns",
        help="Directory to save PGN files (default: benchmark_pgns/)",
    )

    args = parser.parse_args()

    # Resolve our engine command.
    project_root = Path(__file__).resolve().parents[1]
    engine_cmd = [sys.executable, "-m", "src.protocols.uci"]

    print(f"Engine command  : {' '.join(engine_cmd)}")
    print(f"Working dir     : {project_root}")
    print(f"Stockfish       : {args.stockfish_path}")
    print(f"Elo levels      : {args.elo}")
    print(f"Rounds per level: {args.rounds}")
    print(f"Time per move   : {args.tc}s")
    print()

    results: List[MatchResult] = []
    os.chdir(project_root)

    for sf_elo in sorted(args.elo):
        print(f"--- Match vs Stockfish {sf_elo} ---")
        try:
            mr = run_match(
                engine_cmd=engine_cmd,
                sf_path=args.stockfish_path,
                sf_elo=sf_elo,
                rounds=args.rounds,
                time_per_move=args.tc,
                pgn_dir=Path(args.pgn_dir),
            )
            results.append(mr)
            print(f"  {mr.summary()}")
        except chess.engine.EngineTerminatedError as e:
            print(f"  Engine crashed: {e}")
        except FileNotFoundError:
            print(f"  ERROR: Could not find engine binary. Is Stockfish installed?")
            print(f"         Install with: sudo apt install stockfish")
            sys.exit(1)
        print()

    # ── Final summary ────────────────────────────────────────────────────
    if not results:
        print("No results collected.")
        return

    print("=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    for mr in results:
        print(f"  {mr.summary()}")

    # Weighted average Elo estimate (weight by number of decisive games).
    estimates = [(mr.elo_estimate, mr.total) for mr in results if mr.elo_estimate is not None]
    if estimates:
        total_weight = sum(w for _, w in estimates)
        if total_weight > 0:
            avg_elo = sum(e * w for e, w in estimates) / total_weight
            print(f"\n  Weighted average Elo estimate: {avg_elo:.0f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
