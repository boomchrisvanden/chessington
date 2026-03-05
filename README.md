# Chessington

A chess engine with alpha-beta search, opening book integration, and an interactive Pygame GUI. The long-term goal is NNUE evaluation with hardware inference on FPGA via SystemVerilog.

## Features

**Engine**
- Alpha-beta search with iterative deepening and aspiration windows
- Principal Variation Search (PVS) with zero-window re-search
- Null move pruning, late move reductions, reverse futility pruning, futility pruning
- Quiescence search with delta pruning
- Move ordering: MVV-LVA, killer moves, history heuristic
- Transposition table (Zobrist hashing)
- PeSTO piece-square table evaluation (tapered middlegame/endgame)
- UCI protocol support

**GUI**
- Pygame board with drag-and-drop and UCI text input
- Local two-player and play-against-engine modes
- Move highlighting, check/checkmate/draw detection

**Opening Theory Practice**
- Practice openings against a Polyglot opening book (`Book.bin`)
- Five difficulty levels: Infinite, Easy, Medium, Hard, Insane
- Fuzzy search to find and drill specific openings by name

## Requirements

- Python 3.10+
- [pygame](https://pypi.org/project/pygame/)
- [pytest](https://pypi.org/project/pytest/) (for running tests)
- Optional: [cairosvg](https://pypi.org/project/CairoSVG/), `rsvg-convert`, or `inkscape` for SVG-to-PNG piece conversion

## Getting Started

### Piece images

Place PNG piece images in `assets/` (or `assets/pieces/`). Files should follow the naming convention `wP.png`, `bN.png`, etc. A set of PNGs is included by default.

If you only have SVGs, convert them:

```
python3 scripts/convert_svgs_to_pngs.py assets
```

### Running the GUI

```
python3 gui.py
```

The main menu offers four modes:

| Mode | Description |
|------|-------------|
| **Local Game** | Two players on one screen. Type UCI moves or drag pieces. |
| **Play Against Engine** | Choose your color; the engine plays the other side at depth 6. |
| **Theory Practice** | The book picks a random opening line. Match the book moves to continue; wrong moves cost lives. |
| **Opening Search Practice** | Same as theory practice, but lets you search for a specific opening by name first. |

**Controls**: drag-and-drop pieces, type UCI moves (e.g. `e2e4`, `e7e8q`) + Enter, `reset` to restart, `Esc` or `m` to return to menu.

### Running the engine standalone (UCI)

```
python3 cli.py
```

This starts a UCI loop. You can connect it to any UCI-compatible GUI (Arena, CuteChess, etc.).

### Running tests

```
python3 -m pytest tests/
```

## Project Structure

```
chessington/
├── cli.py                        # UCI engine entry point
├── gui.py                        # Pygame GUI (main menu, game modes)
├── assets/                       # Piece image PNGs
├── scripts/
│   └── convert_svgs_to_pngs.py   # SVG → PNG conversion utility
├── src/
│   ├── core/
│   │   ├── types.py              # Color, PieceType, Move, CastlingRights
│   │   ├── board.py              # Bitboard + array hybrid board
│   │   └── zobrist.py            # Zobrist hash tables
│   ├── search/
│   │   ├── alphabeta.py          # PVS + NMP + LMR + RFP + FP
│   │   ├── eval.py               # PeSTO PST evaluation (tapered)
│   │   ├── iterative.py          # Iterative deepening + aspiration windows
│   │   ├── ordering.py           # MVV-LVA, killers, history heuristic
│   │   ├── quiescence.py         # Capture search with delta pruning
│   │   └── tt.py                 # Transposition table
│   ├── protocols/
│   │   ├── uci.py                # UCI protocol loop
│   │   └── engine_client.py      # Engine subprocess management
│   ├── book/
│   │   ├── polyglot_book.py      # Polyglot .bin reader
│   │   └── polyglot_zobrist.py   # Zobrist hashing for book lookups
│   ├── theory/
│   │   ├── practice.py           # Theory practice game logic
│   │   └── gui.py                # Theory practice GUI
│   └── utils/
│       └── assets.py             # Piece image loading (shared)
├── tests/                        # pytest test suite
└── CLAUDE.md                     # Agent instructions
```

## Roadmap

### Phase 1: Search (complete)
Piece-square tables, null move pruning, late move reductions, principal variation search, aspiration windows, reverse futility pruning, futility pruning.

### Phase 2: NNUE
- HalfKA architecture (768 &rarr; 256 &rarr; 32 &rarr; 32 &rarr; 1)
- int8 weights / int16 accumulators, incremental accumulator updates
- Training data generation and PyTorch training

### Phase 3: FPGA Hardware Inference
- Quantization-aware training and weight export
- SystemVerilog MAC array, ClippedReLU, BRAM weight storage
- FPGA pipeline with host interface (UART/SPI)
