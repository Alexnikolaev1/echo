"""Entity layer: TimeSeries and pure numeric primitives.

Z-normalized Euclidean distance is the standard distance measure in time
series analysis precisely because it makes shape comparison invariant to
offset and scale: a pattern shifted up by 10 units or stretched 2x in
amplitude is still recognised as "the same shape" — which is essential for
comparing, say, a sensor reading during a hot day against a cold day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class TooFewPointsError(ValueError):
    pass


class InvalidSubsequenceLengthError(ValueError):
    pass


class NonFiniteValueError(ValueError):
    pass


class LengthMismatchError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TimeSeries:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) < 4:
            raise TooFewPointsError(
                "TimeSeries needs at least 4 points to be meaningful."
            )
        if any(not math.isfinite(v) for v in self.values):
            raise NonFiniteValueError(
                "TimeSeries values must be finite (no NaN or Inf)."
            )

    def __len__(self) -> int:
        return len(self.values)

    def subsequence(self, start: int, length: int) -> tuple[float, ...]:
        return self.values[start : start + length]

    def num_subsequences(self, subsequence_length: int) -> int:
        if subsequence_length < 3 or subsequence_length > len(self.values) // 2:
            raise InvalidSubsequenceLengthError(
                f"subsequence_length must be between 3 and {len(self.values) // 2} "
                f"for a series of length {len(self.values)}."
            )
        return len(self.values) - subsequence_length + 1


def z_normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    """Rescales a subsequence to zero mean and unit variance. A constant
    (zero-variance) subsequence normalizes to all zeros rather than raising
    a division error — a flat line is a valid, if boring, shape."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    if std < 1e-10:
        return tuple(0.0 for _ in values)
    return tuple((v - mean) / std for v in values)


def z_normalized_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b):
        raise LengthMismatchError(
            f"Subsequences must have equal length, got {len(a)} and {len(b)}."
        )
    za, zb = z_normalize(a), z_normalize(b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(za, zb)))
