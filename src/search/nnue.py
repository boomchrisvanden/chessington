"""
NNUE evaluation: HalfKA-style 768 -> 256 -> 32 -> 32 -> 1

Quantized inference with int16 feature-transformer weights/accumulators
and int8 hidden-layer weights.  All arithmetic uses integer numpy ops
so the benchmark reflects realistic FPGA-portable inference cost.

Feature encoding (768 per perspective):
  index = piece_color * 384 + (piece_type - 1) * 64 + square
  White perspective uses raw squares; black perspective flips color
  and mirrors the square vertically (sq ^ 56).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from src.core.board import Board
from src.core.types import Color, PieceType

# ── Architecture constants ───────────────────────────────────────────────────

N_FEATURES = 768        # 2 colors * 6 piece types * 64 squares
FT_OUT     = 256        # feature-transformer output (per perspective)
L1_OUT     = 32
L2_OUT     = 32
OUT_DIM    = 1

# Quantization parameters (Stockfish-style).
# FT weights/accumulators live in int16 space.
# After ClippedReLU the activations are clamped to [0, QA] and stored as int8.
QA = 127                # activation quantization range
QB = 64                 # weight quantization factor for hidden layers
OUTPUT_SCALE = 400      # final rescaling to centipawns


# ── Network ──────────────────────────────────────────────────────────────────

class NNUENetwork:
    """Quantized NNUE net.  Weights can be loaded from .npz or randomised."""

    __slots__ = (
        "ft_weight", "ft_bias",
        "l1_weight", "l1_bias",
        "l2_weight", "l2_bias",
        "out_weight", "out_bias",
        "_l1_w32", "_l2_w32", "_out_w32",
    )

    def __init__(self) -> None:
        # Feature transformer (768 -> 256): int16
        self.ft_weight  = np.zeros((N_FEATURES, FT_OUT), dtype=np.int16)
        self.ft_bias    = np.zeros(FT_OUT, dtype=np.int16)
        # Hidden 1 (512 -> 32): int8 weights, int32 bias
        self.l1_weight  = np.zeros((FT_OUT * 2, L1_OUT), dtype=np.int8)
        self.l1_bias    = np.zeros(L1_OUT, dtype=np.int32)
        # Hidden 2 (32 -> 32): int8 weights, int32 bias
        self.l2_weight  = np.zeros((L1_OUT, L2_OUT), dtype=np.int8)
        self.l2_bias    = np.zeros(L2_OUT, dtype=np.int32)
        # Output  (32 -> 1):  int8 weights, int32 bias
        self.out_weight = np.zeros((L2_OUT, OUT_DIM), dtype=np.int8)
        self.out_bias   = np.zeros(OUT_DIM, dtype=np.int32)
        # Pre-cast hidden weights for fast matmul (avoid per-eval .astype)
        self._l1_w32  = np.zeros((FT_OUT * 2, L1_OUT), dtype=np.int32)
        self._l2_w32  = np.zeros((L1_OUT, L2_OUT), dtype=np.int32)
        self._out_w32 = np.zeros((L2_OUT, OUT_DIM), dtype=np.int32)

    def _cache_int32(self) -> None:
        """Cache int32 copies of hidden-layer weights for fast matmul."""
        self._l1_w32  = self.l1_weight.astype(np.int32)
        self._l2_w32  = self.l2_weight.astype(np.int32)
        self._out_w32 = self.out_weight.astype(np.int32)

    # ── Serialisation ────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            ft_weight=self.ft_weight, ft_bias=self.ft_bias,
            l1_weight=self.l1_weight, l1_bias=self.l1_bias,
            l2_weight=self.l2_weight, l2_bias=self.l2_bias,
            out_weight=self.out_weight, out_bias=self.out_bias,
        )

    def load(self, path: str | Path) -> None:
        d = np.load(path)
        self.ft_weight  = d["ft_weight"]
        self.ft_bias    = d["ft_bias"]
        self.l1_weight  = d["l1_weight"]
        self.l1_bias    = d["l1_bias"]
        self.l2_weight  = d["l2_weight"]
        self.l2_bias    = d["l2_bias"]
        self.out_weight = d["out_weight"]
        self.out_bias   = d["out_bias"]
        self._cache_int32()

    def init_random(self, seed: int = 42) -> None:
        """Fill all layers with small random ints for benchmarking."""
        rng = np.random.default_rng(seed)
        self.ft_weight  = rng.integers(-30, 31, (N_FEATURES, FT_OUT), np.int16)
        self.ft_bias    = rng.integers(-30, 31, FT_OUT, np.int16)
        self.l1_weight  = rng.integers(-64, 64, (FT_OUT * 2, L1_OUT), np.int8)
        self.l1_bias    = rng.integers(-500, 500, L1_OUT, np.int32)
        self.l2_weight  = rng.integers(-64, 64, (L1_OUT, L2_OUT), np.int8)
        self.l2_bias    = rng.integers(-500, 500, L2_OUT, np.int32)
        self.out_weight = rng.integers(-64, 64, (L2_OUT, OUT_DIM), np.int8)
        self.out_bias   = rng.integers(-500, 500, OUT_DIM, np.int32)
        self._cache_int32()



# ── Accumulator (full recompute) ─────────────────────────────────────────────

# Pre-built lookup: (color, piece_type) -> (white_feat_base, black_feat_base).
# White feat index = base_w + sq.  Black feat index = base_b + (sq ^ 56).
_FEAT_BASE: dict[tuple[int, int], tuple[int, int]] = {}
for _c in range(2):
    for _pt in range(1, 7):
        _FEAT_BASE[(_c, _pt)] = (
            _c * 384 + (_pt - 1) * 64,
            (1 - _c) * 384 + (_pt - 1) * 64,
        )


def _compute_accumulator(
    net: NNUENetwork,
    board: Board,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build both perspective accumulators from scratch.

    Returns (white_acc, black_acc) as arrays of shape (FT_OUT,).
    Per-piece in-place addition is faster than fancy-index-sum for
    the typical ~32 active features (avoids gather allocation overhead).
    """
    ft = net.ft_weight
    w_acc = net.ft_bias.copy()
    b_acc = net.ft_bias.copy()
    fb = _FEAT_BASE
    squares = board.squares

    for sq in range(64):
        piece = squares[sq]
        if piece is None:
            continue
        wb, bb = fb[piece]
        w_acc += ft[wb + sq]
        b_acc += ft[bb + (sq ^ 56)]

    return w_acc, b_acc


# ── ClippedReLU ──────────────────────────────────────────────────────────────

def _crelu(x: np.ndarray) -> np.ndarray:
    """Clamp to [0, QA] and return int8."""
    return np.clip(x, 0, QA).astype(np.int8)


# ── Forward pass ─────────────────────────────────────────────────────────────

def nnue_evaluate(net: NNUENetwork, board: Board) -> int:
    """
    Full NNUE forward pass.

    Returns score in centipawns from the side-to-move's perspective.
    """
    w_acc, b_acc = _compute_accumulator(net, board)

    # Perspective ordering: side-to-move accumulator first.
    if board.side_to_move == Color.WHITE:
        combined = np.concatenate([w_acc, b_acc])
    else:
        combined = np.concatenate([b_acc, w_acc])

    # FT ClippedReLU: int16 -> int8  (values are already in QA-scale)
    x = _crelu(combined)

    # Hidden layer 1:  (512,) int8  @  (512, 32) int32  ->  int32
    h = x.astype(np.int32) @ net._l1_w32 + net.l1_bias
    x = _crelu(h // QA)

    # Hidden layer 2:  (32,) int8  @  (32, 32) int32  ->  int32
    h = x.astype(np.int32) @ net._l2_w32 + net.l2_bias
    x = _crelu(h // QA)

    # Output layer:  (32,) int8  @  (32, 1) int32  ->  int32
    out = x.astype(np.int32) @ net._out_w32 + net.out_bias

    # Rescale to centipawns.
    score = int(out[0]) * OUTPUT_SCALE // (QA * QA)

    return score
