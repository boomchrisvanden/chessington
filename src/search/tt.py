from __future__ import annotations


class TranspositionTable:
    """
    Minimal placeholder TT.

    The engine is still WIP; this exists so the UCI layer can keep a stable API.
    """

    def __init__(self, size_mb: int = 32) -> None:
        self.size_mb = int(size_mb)
        self._table: dict[int, object] = {}

    def clear(self) -> None:
        self._table.clear()

