from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, TextIO

from src.core.board import Board
from src.core.types import Color, move_from_uci
from src.search.iterative import iterative_deepening
from src.search.tt import TranspositionTable


@dataclass(slots=True)
class GoLimits:
    depth: Optional[int] = None
    nodes: Optional[int] = None
    movetime_ms: Optional[int] = None
    wtime_ms: Optional[int] = None
    btime_ms: Optional[int] = None
    winc_ms: Optional[int] = None
    binc_ms: Optional[int] = None
    movestogo: Optional[int] = None
    infinite: bool = False
    ponder: bool = False


def _parse_go(tokens: List[str]) -> GoLimits:
    limits = GoLimits()
    i = 0
    while i < len(tokens):
        t = tokens[i]

        if t in ("infinite", "ponder"):
            setattr(limits, t, True)
            i += 1
            continue

        if i + 1 >= len(tokens):
            break

        v = tokens[i + 1]
        try:
            iv = int(v)
        except ValueError:
            i += 1
            continue

        if t == "depth":
            limits.depth = iv
        elif t == "nodes":
            limits.nodes = iv
        elif t == "movetime":
            limits.movetime_ms = iv
        elif t == "wtime":
            limits.wtime_ms = iv
        elif t == "btime":
            limits.btime_ms = iv
        elif t == "winc":
            limits.winc_ms = iv
        elif t == "binc":
            limits.binc_ms = iv
        elif t == "movestogo":
            limits.movestogo = iv

        i += 2

    return limits


def _handle_setoption(tokens: List[str], options: Dict[str, Optional[str]]) -> None:
    """
    UCI: setoption name <name> [value <value>]
    """
    if not tokens or tokens[0] != "name":
        return

    i = 1
    name_parts: List[str] = []
    while i < len(tokens) and tokens[i] != "value":
        name_parts.append(tokens[i])
        i += 1
    name = " ".join(name_parts).strip()
    if not name:
        return

    value: Optional[str] = None
    if i < len(tokens) and tokens[i] == "value":
        value = " ".join(tokens[i + 1 :]).strip() or None

    options[name] = value


def _handle_position(tokens: List[str], current: Optional[Board]) -> Optional[Board]:
    """
    UCI:
      position startpos [moves ...]
      position fen <fen> [moves ...]
    """
    if not tokens:
        return current

    board: Optional[Board]
    i = 0

    if tokens[0] == "startpos":
        board = Board.from_startpos()
        i = 1
    elif tokens[0] == "fen":
        i = 1
        fen_parts: List[str] = []
        while i < len(tokens) and tokens[i] != "moves":
            fen_parts.append(tokens[i])
            i += 1
        try:
            board = Board.from_fen(" ".join(fen_parts))
        except ValueError:
            return current
    else:
        return current

    if i < len(tokens) and tokens[i] == "moves":
        i += 1
        for mv_text in tokens[i:]:
            mv = move_from_uci(mv_text)
            if mv is None:
                continue
            board.make_move(mv)

    return board


def _compute_time_ms(board: Board, limits: GoLimits) -> int:
    if limits.movetime_ms is not None:
        return max(0, limits.movetime_ms)

    if limits.wtime_ms is None or limits.btime_ms is None:
        return 0

    remaining = limits.wtime_ms if board.side_to_move == Color.WHITE else limits.btime_ms
    inc = limits.winc_ms if board.side_to_move == Color.WHITE else limits.binc_ms
    movestogo = limits.movestogo or 30
    if remaining <= 0:
        return 0

    return max(0, (remaining // max(1, movestogo)) + (inc or 0))


def uci_loop(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
    def send(line: str) -> None:
        stdout.write(line + "\n")
        stdout.flush()

    board: Optional[Board] = None
    tt = TranspositionTable(size_mb=32)
    options: Dict[str, Optional[str]] = {}

    for raw in stdin:
        line = raw.strip()
        if not line:
            continue

        tokens = line.split()
        cmd, args = tokens[0], tokens[1:]

        if cmd == "uci":
            send("id name chessington")
            send("id author Chris Vanden Boom")
            send("option name Hash type spin default 32 min 1 max 2048")
            send("uciok")

        elif cmd == "isready":
            send("readyok")

        elif cmd == "ucinewgame":
            tt.clear()

        elif cmd == "setoption":
            _handle_setoption(args, options)
            if options.get("Hash"):
                try:
                    tt = TranspositionTable(size_mb=int(options["Hash"] or "32"))
                except ValueError:
                    pass

        elif cmd == "position":
            board = _handle_position(args, board)

        elif cmd == "go":
            if board is None:
                send("bestmove 0000")
                continue
            limits = _parse_go(args)
            max_depth = limits.depth or 1
            time_ms = _compute_time_ms(board, limits)

            result = iterative_deepening(board, max_depth=max_depth, time_ms=time_ms, tt=tt)
            if result.best_move is None:
                send("bestmove 0000")
            else:
                send(f"bestmove {result.best_move.uci()}")

        elif cmd == "stop":
            # Current search is synchronous, so there's nothing to stop yet.
            pass

        elif cmd == "quit":
            break

        else:
            # Ignore unknown commands for compatibility.
            continue


def main() -> None:
    uci_loop()


if __name__ == "__main__":
    main()
