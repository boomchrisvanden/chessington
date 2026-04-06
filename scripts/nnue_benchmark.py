#!/usr/bin/env python3
"""
Benchmark NNUE inference vs PeSTO eval.

Measures raw evals/sec for:
  1. NNUE forward pass (full accumulator recompute each call)
  2. Current PeSTO tapered eval
  3. Full iterative-deepening search at depth 5/6/7 with each eval

Usage:
    python3 scripts/nnue_benchmark.py              # raw eval + search benchmarks
    python3 scripts/nnue_benchmark.py --eval-only   # skip search benchmarks
    python3 scripts/nnue_benchmark.py --depth 5 6   # choose search depths
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.board import Board
from src.search.eval import evaluate as pst_evaluate
from src.search.nnue import NNUENetwork, nnue_evaluate
from src.search.iterative import iterative_deepening
from src.search.tt import TranspositionTable
import src.search.alphabeta as _ab_mod
import src.search.quiescence as _qs_mod

# ── Test positions ───────────────────────────────────────────────────────────

POSITIONS = {
    "startpos":   "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "midgame":    "r1bqkb1r/pppppppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "complex":    "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 6 8",
    "endgame":    "8/5pk1/6p1/8/3K4/8/5PPP/8 w - - 0 40",
}


def bench_raw_eval(board: Board, n_iters: int) -> dict:
    """Benchmark raw eval functions on a single position."""
    # --- PeSTO ---
    t0 = time.perf_counter()
    for _ in range(n_iters):
        pst_evaluate(board)
    pst_time = time.perf_counter() - t0

    # --- NNUE ---
    net = NNUENetwork()
    net.init_random()
    # Warm up numpy
    for _ in range(100):
        nnue_evaluate(net, board)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        nnue_evaluate(net, board)
    nnue_time = time.perf_counter() - t0

    return {
        "pst_eps": n_iters / pst_time,
        "nnue_eps": n_iters / nnue_time,
        "pst_us": pst_time / n_iters * 1e6,
        "nnue_us": nnue_time / n_iters * 1e6,
        "ratio": nnue_time / pst_time,
    }


def _swap_eval(eval_fn):
    """Monkey-patch the evaluate function used by search modules."""
    _ab_mod.evaluate = eval_fn
    _qs_mod.evaluate = eval_fn


def bench_search(board: Board, depth: int, use_nnue: bool) -> dict:
    """Run iterative deepening to `depth` and report stats."""
    if use_nnue:
        net = NNUENetwork()
        net.init_random()
        _swap_eval(lambda b: nnue_evaluate(net, b))
    else:
        _swap_eval(pst_evaluate)

    tt = TranspositionTable()

    t0 = time.perf_counter()
    result = iterative_deepening(board, max_depth=depth, tt=tt)
    elapsed = time.perf_counter() - t0

    # Always restore PeSTO after benchmark
    _swap_eval(pst_evaluate)

    return {
        "depth": depth,
        "nodes": result.nodes,
        "time": elapsed,
        "nps": result.nodes / elapsed if elapsed > 0 else 0,
        "score": result.score_cp,
        "move": result.best_move.uci() if result.best_move else "-",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NNUE inference benchmark")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip search benchmarks")
    parser.add_argument("--iters", type=int, default=50_000,
                        help="Iterations for raw eval benchmark (default 50000)")
    parser.add_argument("--depth", type=int, nargs="+", default=[5, 6, 7],
                        help="Search depths to benchmark (default 5 6 7)")
    args = parser.parse_args()

    print("=" * 68)
    print("  NNUE Inference Benchmark (int8/int16 numpy)")
    print("=" * 68)

    # ── Raw eval throughput ──────────────────────────────────────────────
    print(f"\n--- Raw eval throughput ({args.iters:,} iterations per position) ---\n")
    print(f"{'Position':<12} {'PeSTO evals/s':>15} {'NNUE evals/s':>15} "
          f"{'PeSTO us':>10} {'NNUE us':>10} {'Ratio':>7}")
    print("-" * 68)

    for name, fen in POSITIONS.items():
        board = Board.from_fen(fen)
        r = bench_raw_eval(board, args.iters)
        print(f"{name:<12} {r['pst_eps']:>15,.0f} {r['nnue_eps']:>15,.0f} "
              f"{r['pst_us']:>9.1f}µ {r['nnue_us']:>9.1f}µ {r['ratio']:>6.1f}x")

    if args.eval_only:
        return

    # ── Search benchmarks ────────────────────────────────────────────────
    for eval_name, use_nnue in [("PeSTO", False), ("NNUE (random weights)", True)]:
        print(f"\n--- Search benchmark ({eval_name}, iterative deepening) ---\n")
        print(f"{'Position':<12} {'Depth':>5} {'Nodes':>10} {'Time (s)':>10} "
              f"{'NPS':>12} {'Score':>7} {'Move':>6}")
        print("-" * 68)

        for name, fen in POSITIONS.items():
            board = Board.from_fen(fen)
            for d in args.depth:
                r = bench_search(board, d, use_nnue=use_nnue)
                print(f"{name:<12} {r['depth']:>5} {r['nodes']:>10,} "
                      f"{r['time']:>10.3f} {r['nps']:>12,.0f} "
                      f"{r['score']:>7} {r['move']:>6}")


if __name__ == "__main__":
    main()
