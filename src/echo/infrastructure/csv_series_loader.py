"""Infrastructure layer: CsvSeriesLoader."""

from __future__ import annotations

import csv
import math


class ColumnNotFoundError(ValueError):
    pass


class NonNumericValueError(ValueError):
    pass


class EmptyColumnError(ValueError):
    pass


class CsvSeriesLoader:
    def load_column(self, file_path: str, column: str) -> tuple[float, ...]:
        with open(file_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise ColumnNotFoundError(
                    f"Column '{column}' not found. Available: {reader.fieldnames}"
                )
            values = []
            for row_num, row in enumerate(reader, start=2):
                values.append(_parse_finite(row[column], row_num, column))
        if not values:
            raise EmptyColumnError(f"Column '{column}' has no data rows.")
        return tuple(values)


def _parse_finite(raw: str | None, row_num: int, column: str) -> float:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise NonNumericValueError(
            f"Row {row_num}: '{raw}' in column '{column}' is not numeric."
        ) from exc
    if not math.isfinite(value):
        raise NonNumericValueError(
            f"Row {row_num}: '{raw}' in column '{column}' is not finite."
        )
    return value
