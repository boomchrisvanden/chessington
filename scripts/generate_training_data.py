#!/usr/bin/env python3
"""Generate NNUE training data via self-play.

Strategy:
  1. Start from startpos.
  2. Play 8 random moves to diversify the opening.
  3. Continue with a mix of depth-1 (PeSTO) and random moves.
  4. At each non-check, non-extreme position, record features + eval.
  5. Output .npz with sparse feature arrays and scores.

Usage:
  python scripts/generate_training_data.py --games 10000 --output data/train.npz
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.board import Board
from src.core.types import Color, PieceType
from src.search.eval import evaluate

# ── Feature encoding (mirrors src/search/nnue.py) ────────────────────────────

_FEAT_BASE: dict[tuple[int, int], tuple[int, int]] = {}
for _c in range(2):
    for _pt in range(1, 7):
        _FEAT_BASE[(_c, _pt)] = (
            _c * 384 + (_pt - 1) * 64,           # white-perspective base
            (1 - _c) * 384 + (_pt - 1) * 64,     # black-perspective base
        )


def _extract_features(board: Board):
    """Return (white_indices, black_indices) lists of active feature indices."""
    w_feats: list[int] = []
    b_feats: list[int] = []
    for sq in range(64):
        piece = board.squares[sq]
        if piece is None:
            continue
        wb, bb = _FEAT_BASE[piece]
        w_feats.append(wb + sq)
        b_feats.append(bb + (sq ^ 56))
    return w_feats, b_feats


# ── Self-play game ────────────────────────────────────────────────────────────

MAX_PIECES_PER_SIDE = 16   # max active features per perspective
PADDING = 65535             # sentinel for unused feature slots
SCORE_CUTOFF = 3000         # skip extreme positions (cp)


def _play_game(
    random_opening: int,
    random_prob: float,
    max_plies: int,
) -> list[tuple[list[int], list[int], int, int]]:
    """Play one self-play game, return [(w_feats, b_feats, stm, score_cp), ...]."""
    board = Board.from_startpos()
    positions: list[tuple[list[int], list[int], int, int]] = []
    seen_hashes: dict[int, int] = {}

    for ply in range(max_plies):
        # ── Termination checks ────────────────────────────────────────
        h = board.hash
        seen_hashes[h] = seen_hashes.get(h, 0) + 1
        if seen_hashes[h] >= 3:
            break  # threefold repetition
        if board.halfmove_clock >= 100:
            break  # 50-move rule

        pseudo_moves = board.generate_pseudo_legal()
        random.shuffle(pseudo_moves)

        # Find legal moves (and optionally their scores for depth-1)
        use_random = ply < random_opening or random.random() < random_prob
        best_move = None
        best_score = -999999
        legal_count = 0

        for m in pseudo_moves:
            undo = board.make_move(m)
            if undo is None:
                continue
            legal_count += 1

            if use_random:
                # Take first legal move (already shuffled)
                board.unmake_move(undo)
                best_move = m
                break
            else:
                # Depth-1: pick highest-scoring move
                score = -evaluate(board)
                board.unmake_move(undo)
                if score > best_score:
                    best_score = score
                    best_move = m

        if best_move is None:
            break  # no legal moves (checkmate or stalemate)

        # ── Record position (before making the chosen move) ───────────
        if ply >= random_opening and not board.in_check(board.side_to_move):
            score_cp = evaluate(board)  # from stm perspective
            if abs(score_cp) < SCORE_CUTOFF:
                w_feats, b_feats = _extract_features(board)
                positions.append((
                    w_feats, b_feats,
                    int(board.side_to_move),
                    score_cp,
                ))

        # ── Make the move ─────────────────────────────────────────────
        undo = board.make_move(best_move)
        if undo is None:
            break  # shouldn't happen

    return positions


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NNUE training data")
    parser.add_argument("--games", type=int, default=10000,
                        help="Number of self-play games")
    parser.add_argument("--output", type=str, default="data/training_data.npz",
                        help="Output .npz path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-opening", type=int, default=8,
                        help="Random moves at start of each game")
    parser.add_argument("--random-prob", type=float, default=0.20,
                        help="Probability of random move after opening")
    parser.add_argument("--max-plies", type=int, default=300,
                        help="Max half-moves per game")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    all_w: list[list[int]] = []
    all_b: list[list[int]] = []
    all_stm: list[int] = []
    all_scores: list[int] = []

    start = time.time()
    for g in range(args.games):
        positions = _play_game(args.random_opening, args.random_prob, args.max_plies)
        for w_feats, b_feats, stm, score_cp in positions:
            # Pad to fixed size
            w_padded = (w_feats + [PADDING] * MAX_PIECES_PER_SIDE)[:MAX_PIECES_PER_SIDE]
            b_padded = (b_feats + [PADDING] * MAX_PIECES_PER_SIDE)[:MAX_PIECES_PER_SIDE]
            all_w.append(w_padded)
            all_b.append(b_padded)
            all_stm.append(stm)
            all_scores.append(score_cp)

        if (g + 1) % 1000 == 0:
            elapsed = time.time() - start
            rate = (g + 1) / elapsed
            print(f"  {g+1:>6}/{args.games} games | "
                  f"{len(all_scores):>8} positions | "
                  f"{rate:.0f} games/s")

    # ── Save ──────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        white_features=np.array(all_w, dtype=np.uint16),
        black_features=np.array(all_b, dtype=np.uint16),
        stm=np.array(all_stm, dtype=np.uint8),
        scores=np.array(all_scores, dtype=np.int16),
    )

    elapsed = time.time() - start
    n = len(all_scores)
    scores_arr = np.array(all_scores)
    print(f"\nDone: {n:,} positions from {args.games:,} games in {elapsed:.1f}s")
    print(f"Score stats: mean={scores_arr.mean():.0f}  std={scores_arr.std():.0f}  "
          f"min={scores_arr.min()}  max={scores_arr.max()}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
