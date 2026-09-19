"""Entity layer: MatrixProfile computation.

For every subsequence in the series, the matrix profile records the
distance to its nearest neighbour ELSEWHERE in the series, and which index
that neighbour is at. A HIGH distance means "nothing else in this series
looks like this" — an anomaly (a "discord"). A LOW distance means "this
shape repeats" — a normal, recurring pattern (a "motif").

The exclusion zone is not an optimisation detail — it is a correctness
requirement. Without it, every subsequence's "nearest neighbour" would
trivially be itself (or a near-identical overlapping window one step away),
making every distance ~0 and the whole profile meaningless. Excluding a
window of half the subsequence length around each point is the standard
fix from the original Matrix Profile paper (Yeh et al., 2016).
"""

from __future__ import annotations

from dataclasses import dataclass

from echo.entities.time_series import TimeSeries, z_normalized_distance


@dataclass(frozen=True, slots=True)
class MatrixProfileEntry:
    index: int
    nearest_neighbor_distance: float
    nearest_neighbor_index: int


@dataclass(frozen=True, slots=True)
class MatrixProfile:
    entries: tuple[MatrixProfileEntry, ...]
    subsequence_length: int

    def sorted_by_distance_desc(self) -> list[MatrixProfileEntry]:
        """Discords (anomalies) first — highest distance means least like
        anything else in the series."""
        return sorted(
            self.entries, key=lambda e: e.nearest_neighbor_distance, reverse=True
        )

    def sorted_by_distance_asc(self) -> list[MatrixProfileEntry]:
        """Motifs (recurring patterns) first — lowest distance means most
        similar to something else in the series."""
        return sorted(self.entries, key=lambda e: e.nearest_neighbor_distance)


def _nearest_neighbor(series: TimeSeries, i: int, n: int, m: int) -> MatrixProfileEntry:
    exclusion_radius = m // 2
    sub_i = series.subsequence(i, m)
    best_distance = float("inf")
    best_j = -1
    for j in range(n):
        if abs(i - j) <= exclusion_radius:
            continue  # skip trivial self/near-self matches
        distance = z_normalized_distance(sub_i, series.subsequence(j, m))
        if distance < best_distance:
            best_distance = distance
            best_j = j
    return MatrixProfileEntry(
        index=i,
        nearest_neighbor_distance=best_distance,
        nearest_neighbor_index=best_j,
    )


def compute_matrix_profile(
    series: TimeSeries, subsequence_length: int
) -> MatrixProfile:
    m = subsequence_length
    n = series.num_subsequences(m)
    entries = tuple(_nearest_neighbor(series, i, n, m) for i in range(n))
    return MatrixProfile(entries=entries, subsequence_length=m)
