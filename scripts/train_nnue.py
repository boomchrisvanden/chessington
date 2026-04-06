#!/usr/bin/env python3
"""Train NNUE (768→256→32→32→1) from generated training data.

Float training with sigmoid-scaled MSE loss, then quantize to int8/int16
for the inference engine.  Architecture matches src/search/nnue.py exactly.

Usage:
  python scripts/train_nnue.py --data data/training_data.npz --output data/nnue_trained.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("PyTorch is required.  Install with:  pip install torch")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.board import Board
from src.core.types import Color
from src.search.eval import evaluate as pesto_evaluate
from src.search.nnue import (
    N_FEATURES, FT_OUT, L1_OUT, L2_OUT, OUT_DIM,
    QA, OUTPUT_SCALE, NNUENetwork, nnue_evaluate,
)

# ── PyTorch Model ─────────────────────────────────────────────────────────────

class NNUEModel(nn.Module):
    """Float NNUE for training.  Mirrors quantised inference exactly."""

    def __init__(self):
        super().__init__()
        self.ft  = nn.Linear(N_FEATURES, FT_OUT)       # 768 → 256
        self.l1  = nn.Linear(FT_OUT * 2, L1_OUT)       # 512 → 32
        self.l2  = nn.Linear(L1_OUT, L2_OUT)            # 32 → 32
        self.out = nn.Linear(L2_OUT, OUT_DIM)           # 32 → 1

        # Small init — keeps quantised weights within int8/int16 range
        nn.init.uniform_(self.ft.weight, -0.05, 0.05)
        nn.init.zeros_(self.ft.bias)
        nn.init.kaiming_normal_(self.l1.weight, nonlinearity='relu')
        nn.init.zeros_(self.l1.bias)
        nn.init.kaiming_normal_(self.l2.weight, nonlinearity='relu')
        nn.init.zeros_(self.l2.bias)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, w_feat, b_feat, stm):
        """
        Args:
            w_feat: (B, 768) float binary features — white perspective
            b_feat: (B, 768) float binary features — black perspective
            stm:    (B,)     float 0.0 = white, 1.0 = black
        Returns:
            (B, 1) float — raw score (centipawns = output × OUTPUT_SCALE)
        """
        w_acc = self.ft(w_feat)                         # (B, 256)
        b_acc = self.ft(b_feat)                         # (B, 256)

        # Side-to-move perspective: stm accumulator first
        mask = stm.unsqueeze(1)                         # (B, 1)
        first  = w_acc * (1.0 - mask) + b_acc * mask
        second = b_acc * (1.0 - mask) + w_acc * mask
        x = torch.cat([first, second], dim=1)           # (B, 512)

        x = torch.clamp(x, 0.0, 1.0)                   # ClippedReLU
        x = torch.clamp(self.l1(x), 0.0, 1.0)
        x = torch.clamp(self.l2(x), 0.0, 1.0)
        return self.out(x)                              # (B, 1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class NNUEDataset(Dataset):
    """Loads .npz from generate_training_data.py."""

    def __init__(self, path: str):
        data = np.load(path)
        w_idx  = data["white_features"]                 # (N, 16) uint16
        b_idx  = data["black_features"]                 # (N, 16) uint16
        self.stm    = torch.from_numpy(data["stm"].astype(np.float32))
        self.scores = torch.from_numpy(
            data["scores"].astype(np.float32) / OUTPUT_SCALE
        )
        self.n = len(self.scores)

        # Vectorised sparse→dense conversion
        print(f"  Building dense features for {self.n:,} positions …")
        rows = np.repeat(np.arange(self.n, dtype=np.int64), 16)

        cols_w = w_idx.ravel().astype(np.int64)
        mask_w = cols_w < N_FEATURES
        self.w_dense = torch.zeros(self.n, N_FEATURES, dtype=torch.float32)
        self.w_dense[rows[mask_w], cols_w[mask_w]] = 1.0

        cols_b = b_idx.ravel().astype(np.int64)
        mask_b = cols_b < N_FEATURES
        self.b_dense = torch.zeros(self.n, N_FEATURES, dtype=torch.float32)
        self.b_dense[rows[mask_b], cols_b[mask_b]] = 1.0
        print("  Done.")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (
            self.w_dense[idx],
            self.b_dense[idx],
            self.stm[idx],
            self.scores[idx],
        )


# ── Loss ──────────────────────────────────────────────────────────────────────

def sigmoid_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE in win-probability space.

    Both pred and target are in normalised score space (cp / OUTPUT_SCALE).
    sigmoid(x) maps them to estimated win probability.
    """
    pred_wp   = torch.sigmoid(pred.squeeze(1))
    target_wp = torch.sigmoid(target)
    return torch.mean((pred_wp - target_wp) ** 2)


# ── Quantisation ──────────────────────────────────────────────────────────────

def quantize(model: NNUEModel) -> NNUENetwork:
    """Convert float model → quantised int inference network.

    Mapping:
      FT  — weight × QA → int16,  bias × QA → int16
      L1/L2/Out — weight × QA → int8,  bias × QA² → int32
    """
    net = NNUENetwork()

    with torch.no_grad():
        # Feature transformer
        ft_w = model.ft.weight.T.cpu().numpy()          # (768, 256)
        ft_b = model.ft.bias.cpu().numpy()               # (256,)
        net.ft_weight = np.clip(np.round(ft_w * QA), -32768, 32767).astype(np.int16)
        net.ft_bias   = np.clip(np.round(ft_b * QA), -32768, 32767).astype(np.int16)

        # Hidden + output layers
        for attr, layer in [("l1", model.l1), ("l2", model.l2), ("out", model.out)]:
            w = layer.weight.T.cpu().numpy()
            b = layer.bias.cpu().numpy()
            w_q = np.round(w * QA)
            b_q = np.round(b * QA * QA)

            # Warn if weights overflow int8
            overflow = np.sum(np.abs(w_q) > 127)
            if overflow > 0:
                pct = 100.0 * overflow / w_q.size
                print(f"  [quantize] {attr}: {overflow}/{w_q.size} weights "
                      f"clipped ({pct:.1f}%)")

            setattr(net, f"{attr}_weight",
                    np.clip(w_q, -128, 127).astype(np.int8))
            setattr(net, f"{attr}_bias", b_q.astype(np.int32))

    net._cache_int32()
    return net


# ── Validation helpers ────────────────────────────────────────────────────────

def _validate_quantization(model: NNUEModel, net: NNUENetwork, n: int = 50):
    """Compare float model vs quantised inference on random boards."""
    rng = np.random.default_rng(99)
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    ]
    model.eval()
    errors = []
    with torch.no_grad():
        for fen in fens:
            board = Board.from_fen(fen)
            # Quantised inference
            q_score = nnue_evaluate(net, board)
            # Float inference
            from scripts.generate_training_data import _extract_features
            w_f, b_f = _extract_features(board)
            w_t = torch.zeros(1, N_FEATURES)
            b_t = torch.zeros(1, N_FEATURES)
            for fi in w_f:
                w_t[0, fi] = 1.0
            for fi in b_f:
                b_t[0, fi] = 1.0
            stm_t = torch.tensor([float(board.side_to_move)])
            f_score = float(model(w_t, b_t, stm_t)[0, 0]) * OUTPUT_SCALE
            errors.append(abs(f_score - q_score))

    mean_err = sum(errors) / len(errors) if errors else 0
    print(f"  Quantisation check: mean |float − quant| = {mean_err:.1f} cp "
          f"(over {len(fens)} positions)")


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    dataset = NNUEDataset(args.data)
    print(f"Loaded {len(dataset):,} positions")

    n_val   = max(1000, len(dataset) // 20)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model     = NNUEModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        n_batches  = 0
        for w, b, stm, score in train_loader:
            w, b   = w.to(device), b.to(device)
            stm    = stm.to(device)
            score  = score.to(device)

            pred = model(w, b, stm)
            loss = sigmoid_mse(pred, score)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping keeps weights in quantisable range
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        val_batches = 0
        val_mae_cp  = 0.0
        with torch.no_grad():
            for w, b, stm, score in val_loader:
                w, b   = w.to(device), b.to(device)
                stm    = stm.to(device)
                score  = score.to(device)

                pred = model(w, b, stm)
                val_loss += sigmoid_mse(pred, score).item()
                # MAE in centipawns
                val_mae_cp += torch.mean(
                    torch.abs(pred.squeeze(1) - score) * OUTPUT_SCALE
                ).item()
                val_batches += 1

        avg_train = train_loss / max(n_batches, 1)
        avg_val   = val_loss / max(val_batches, 1)
        avg_mae   = val_mae_cp / max(val_batches, 1)
        lr        = optimizer.param_groups[0]["lr"]
        dt        = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={avg_train:.6f}  val={avg_val:.6f}  "
              f"mae={avg_mae:.0f}cp  lr={lr:.2e}  ({dt:.1f}s)")

        # ── Checkpointing ─────────────────────────────────────────────
        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), args.checkpoint)
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement")
            break

    # ── Export ─────────────────────────────────────────────────────────
    print(f"\nBest validation loss: {best_val:.6f}")
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.eval()

    print("Quantising weights …")
    net = quantize(model)

    print("Validating quantisation …")
    _validate_quantization(model, net)

    net.save(args.output)
    print(f"Saved quantised weights to {args.output}")

    return net


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train NNUE network")
    parser.add_argument("--data", default="data/training_data.npz",
                        help="Training data .npz")
    parser.add_argument("--output", default="data/nnue_trained.npz",
                        help="Quantised weight output .npz")
    parser.add_argument("--checkpoint", default="data/nnue_best.pt",
                        help="Best float model checkpoint .pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--wd", type=float, default=1e-5,
                        help="Weight decay (L2)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early-stop patience (0 = disabled)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
