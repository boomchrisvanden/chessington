from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from src.core.types import Move


class Bound(IntEnum):
    EXACT = 0
    LOWER = 1
    UPPER = 2


@dataclass(slots=True)
class TTEntry:
    key: int
    depth: int
    score: int
    bound: Bound
    best_move: Optional[Move]


class TranspositionTable:
    def __init__(self, size_mb: int = 32) -> None:
        self.size_mb = int(size_mb)
        self._table: dict[int, TTEntry] = {}
        self._max_entries = max(1, (self.size_mb * 1024 * 1024) // 64)

    def clear(self) -> None:
        self._table.clear()

    def get(self, key: int) -> Optional[TTEntry]:
        return self._table.get(key)

    def store(self, entry: TTEntry) -> None:
        existing = self._table.get(entry.key)
        if existing is not None and existing.depth > entry.depth:
            return
        if len(self._table) >= self._max_entries:
            self._table.pop(next(iter(self._table)))
        self._table[entry.key] = entry
