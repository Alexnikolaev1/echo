"""Entity layer: Discord and Motif — the human-facing findings.

A Discord carries not just "this is anomalous" but its nearest neighbour's
location and the distance to it — the explanation is structural, not a
black-box score. A Motif is the mirror image: a pattern with an unusually
close match elsewhere, evidence of genuine repetition rather than chance.
"""

from __future__ import annotations

from dataclasses import dataclass

from echo.entities.matrix_profile import MatrixProfile, MatrixProfileEntry
from echo.entities.time_series import TimeSeries

_PREVIEW_EDGE = 4
_MIN_DISCORD_DISTANCE = 1e-12


def _preview(values: tuple[float, ...]) -> list[float | str]:
    rounded: list[float | str] = [round(v, 2) for v in values]
    if len(rounded) <= 8:
        return rounded
    return [*rounded[:_PREVIEW_EDGE], "...", *rounded[-_PREVIEW_EDGE:]]


@dataclass(frozen=True, slots=True)
class Discord:
    start_index: int
    length: int
    distance_to_nearest_neighbor: float
    nearest_neighbor_index: int
    values: tuple[float, ...]
    nearest_neighbor_values: tuple[float, ...]

    def explanation(self) -> str:
        span = f"[{self.start_index}:{self.start_index + self.length}]"
        dist = self.distance_to_nearest_neighbor
        if self.nearest_neighbor_index < 0:
            return (
                f"Subsequence at {span} has no valid neighbour outside the "
                f"exclusion zone (distance={dist:.2f}). Values here are "
                f"{_preview(self.values)}."
            )
        return (
            f"Subsequence at {span} doesn't resemble anything else in the "
            f"series (distance={dist:.2f}). Its closest match anywhere else "
            f"is at index {self.nearest_neighbor_index}, but even that match "
            f"is far off — values here are {_preview(self.values)}, versus "
            f"{_preview(self.nearest_neighbor_values)} at the nearest match."
        )


@dataclass(frozen=True, slots=True)
class Motif:
    start_index: int
    length: int
    distance_to_match: float
    match_index: int

    def explanation(self) -> str:
        return (
            f"Subsequence at [{self.start_index}:{self.start_index + self.length}] "
            f"closely repeats the pattern at index {self.match_index} "
            f"(distance={self.distance_to_match:.2f}, near-zero means an almost exact shape match)."
        )


def _to_discord(entry: MatrixProfileEntry, series: TimeSeries, m: int) -> Discord:
    neighbor = entry.nearest_neighbor_index
    neighbor_values: tuple[float, ...] = ()
    if neighbor >= 0:
        neighbor_values = series.subsequence(neighbor, m)
    return Discord(
        start_index=entry.index,
        length=m,
        distance_to_nearest_neighbor=entry.nearest_neighbor_distance,
        nearest_neighbor_index=neighbor,
        values=series.subsequence(entry.index, m),
        nearest_neighbor_values=neighbor_values,
    )


def top_discords(
    profile: MatrixProfile,
    series: TimeSeries,
    top_k: int,
    min_separation: int | None = None,
) -> list[Discord]:
    """Returns up to top_k most anomalous, non-overlapping subsequences.
    Without de-duplication by separation, the top-k list would just be
    k adjacent windows all covering the same single anomaly.
    Stops early when remaining profile distances are ~0 — those are exact
    shape matches (motifs), not anomalies, so padding top_k with them
    would invent fake discords."""
    separation = (
        min_separation if min_separation is not None else profile.subsequence_length
    )
    selected: list[Discord] = []
    for entry in profile.sorted_by_distance_desc():
        if entry.nearest_neighbor_distance <= _MIN_DISCORD_DISTANCE:
            break
        if any(abs(entry.index - d.start_index) < separation for d in selected):
            continue
        selected.append(_to_discord(entry, series, profile.subsequence_length))
        if len(selected) >= top_k:
            break
    return selected


def top_motifs(
    profile: MatrixProfile, top_k: int, min_separation: int | None = None
) -> list[Motif]:
    separation = (
        min_separation if min_separation is not None else profile.subsequence_length
    )
    selected: list[Motif] = []
    for entry in profile.sorted_by_distance_asc():
        if any(abs(entry.index - m.start_index) < separation for m in selected):
            continue
        selected.append(
            Motif(
                start_index=entry.index,
                length=profile.subsequence_length,
                distance_to_match=entry.nearest_neighbor_distance,
                match_index=entry.nearest_neighbor_index,
            )
        )
        if len(selected) >= top_k:
            break
    return selected
