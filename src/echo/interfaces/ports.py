"""Interface layer: SeriesLoader port."""

from __future__ import annotations

from typing import Protocol


class SeriesLoader(Protocol):
    def load_column(self, file_path: str, column: str) -> tuple[float, ...]: ...
