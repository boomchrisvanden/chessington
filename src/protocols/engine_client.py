"""
UCI engine client — manages subprocess, I/O threading, and async search.

Extracted from gui.py to allow reuse across GUI frontends.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional


class EngineClient:
    """Manages a UCI engine subprocess."""

    def __init__(self, engine_path: Optional[Path] = None) -> None:
        if engine_path is None:
            # Default: cli.py next to the project root
            engine_path = Path(__file__).resolve().parents[2] / "cli.py"
        self.engine_path = engine_path
        self.proc: Optional[subprocess.Popen] = None
        self.queue: queue.Queue = queue.Queue()
        self.epoch: int = 0

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def ensure_running(self) -> bool:
        if self.proc is not None and self.proc.poll() is None:
            return True
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", str(self.engine_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError:
            self.proc = None
            return False
        if not self._handshake():
            self.shutdown()
            return False
        return True

    def _handshake(self) -> bool:
        if not self.send("uci"):
            return False
        if not self.read_until("uciok"):
            return False
        if not self.send("isready"):
            return False
        if not self.read_until("readyok"):
            return False
        return True

    def shutdown(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.proc.terminate()
        except OSError:
            pass
        self.proc = None

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def send(self, cmd: str) -> bool:
        if self.proc is None or self.proc.stdin is None:
            return False
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return False
        return True

    def read_line(self) -> Optional[str]:
        if self.proc is None or self.proc.stdout is None:
            return None
        line = self.proc.stdout.readline()
        if not line:
            return None
        return line.strip()

    def read_until(self, token: str) -> bool:
        while True:
            line = self.read_line()
            if line is None:
                return False
            if line == token or line.startswith(token):
                return True

    # ------------------------------------------------------------------
    # Position / search
    # ------------------------------------------------------------------

    @staticmethod
    def position_command(move_history: List[str]) -> str:
        if not move_history:
            return "position startpos"
        return "position startpos moves " + " ".join(move_history)

    def search_async(self, move_history: List[str], depth: int) -> None:
        """Launch a background thread that sends ``go depth`` and enqueues the result."""
        epoch = self.epoch
        threading.Thread(
            target=self._search_thread,
            args=(epoch, move_history, depth),
            daemon=True,
        ).start()

    def _search_thread(self, epoch: int, move_history: List[str], depth: int) -> None:
        if not self.ensure_running():
            self.queue.put((epoch, None, "Engine not available."))
            return
        if not self.send(self.position_command(move_history)):
            self.queue.put((epoch, None, "Engine command failed."))
            return
        if not self.send(f"go depth {depth}"):
            self.queue.put((epoch, None, "Engine command failed."))
            return
        bestmove: Optional[str] = None
        while True:
            line = self.read_line()
            if line is None:
                break
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    bestmove = parts[1]
                break
        if bestmove in (None, "0000", "(none)"):
            self.queue.put((epoch, None, "Engine returned no move."))
        else:
            self.queue.put((epoch, bestmove, None))

    def poll(self):
        """Drain the result queue.  Yields ``(epoch, bestmove, error)`` tuples."""
        while True:
            try:
                yield self.queue.get_nowait()
            except queue.Empty:
                break
