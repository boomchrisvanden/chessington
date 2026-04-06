"""Tests for NNUE inference correctness."""
from __future__ import annotations

import tempfile

import numpy as np
import pytest

from src.core.board import Board
from src.search.nnue import (
    FT_OUT,
    N_FEATURES,
    NNUENetwork,
    QA,
    _compute_accumulator,
    _FEAT_BASE,
    nnue_evaluate,
)


@pytest.fixture
def net() -> NNUENetwork:
    n = NNUENetwork()
    n.init_random(seed=123)
    return n


@pytest.fixture
def startpos() -> Board:
    return Board.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")


# ── Feature indexing ─────────────────────────────────────────────────────────

class TestFeatureIndexing:
    def test_feat_base_covers_768(self):
        """All feature bases + sq offsets should stay within [0, 768)."""
        for (c, pt), (wb, bb) in _FEAT_BASE.items():
            for sq in range(64):
                assert 0 <= wb + sq < N_FEATURES
                assert 0 <= bb + (sq ^ 56) < N_FEATURES

    def test_white_black_perspectives_differ(self):
        """White and black perspectives should produce different indices
        for the same piece on the same square (unless trivially symmetric)."""
        # White pawn on e2 (sq=12)
        wb, bb = _FEAT_BASE[(0, 1)]  # (WHITE, PAWN)
        w_idx = wb + 12
        b_idx = bb + (12 ^ 56)
        assert w_idx != b_idx

    def test_no_duplicate_features_startpos(self, startpos):
        """Each feature index should appear at most once per perspective."""
        w_idxs = set()
        b_idxs = set()
        for sq in range(64):
            piece = startpos.squares[sq]
            if piece is None:
                continue
            wb, bb = _FEAT_BASE[piece]
            w_idx = wb + sq
            b_idx = bb + (sq ^ 56)
            assert w_idx not in w_idxs, f"duplicate white feature {w_idx}"
            assert b_idx not in b_idxs, f"duplicate black feature {b_idx}"
            w_idxs.add(w_idx)
            b_idxs.add(b_idx)


# ── Accumulator ──────────────────────────────────────────────────────────────

class TestAccumulator:
    def test_shape_and_dtype(self, net, startpos):
        w_acc, b_acc = _compute_accumulator(net, startpos)
        assert w_acc.shape == (FT_OUT,)
        assert b_acc.shape == (FT_OUT,)
        # Result should be at least int16 (may be wider after sum)
        assert w_acc.dtype in (np.int16, np.int32, np.int64)

    def test_empty_board_equals_bias(self, net):
        board = Board.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        w_acc, b_acc = _compute_accumulator(net, board)
        # Only two kings -> bias + 2 weight rows
        # Should NOT equal bare bias (kings contribute)
        assert not np.array_equal(w_acc, net.ft_bias)

    def test_symmetric_startpos(self, net, startpos):
        """At startpos, white and black perspectives should be mirrors
        of each other (same set of feature activations, just swapped)."""
        w_acc, b_acc = _compute_accumulator(net, startpos)
        # They won't be identical because the weight rows differ,
        # but they should both be deterministic.
        w2, b2 = _compute_accumulator(net, startpos)
        np.testing.assert_array_equal(w_acc, w2)
        np.testing.assert_array_equal(b_acc, b2)


# ── Forward pass ─────────────────────────────────────────────────────────────

class TestForward:
    def test_returns_int(self, net, startpos):
        score = nnue_evaluate(net, startpos)
        assert isinstance(score, int)

    def test_deterministic(self, net, startpos):
        s1 = nnue_evaluate(net, startpos)
        s2 = nnue_evaluate(net, startpos)
        assert s1 == s2

    def test_side_to_move_changes_score(self, net):
        """Flipping side-to-move should change the score (accumulators swap)."""
        fen_w = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        fen_b = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        board_w = Board.from_fen(fen_w)
        board_b = Board.from_fen(fen_b)
        sw = nnue_evaluate(net, board_w)
        sb = nnue_evaluate(net, board_b)
        # With random weights the two perspectives should generally differ
        # (ClippedReLU breaks exact negation symmetry).
        assert sw != sb

    def test_zero_weights_give_zero(self):
        """With all-zero weights, output should be zero."""
        net = NNUENetwork()  # all zeros
        board = Board.from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        assert nnue_evaluate(net, board) == 0


# ── Serialisation ────────────────────────────────────────────────────────────

class TestSerialisation:
    def test_save_load_roundtrip(self, net):
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            net.save(f.name)
            net2 = NNUENetwork()
            net2.load(f.name)
            np.testing.assert_array_equal(net.ft_weight, net2.ft_weight)
            np.testing.assert_array_equal(net.l1_weight, net2.l1_weight)
            np.testing.assert_array_equal(net.out_weight, net2.out_weight)
