"""Golden README sensor demo and explanation edge cases."""

from __future__ import annotations

import math

from echo.entities.findings import Discord, top_discords, top_motifs
from echo.entities.matrix_profile import (
    MatrixProfile,
    MatrixProfileEntry,
    compute_matrix_profile,
)
from echo.entities.time_series import TimeSeries
from echo.use_cases.analyze_time_series import AnalyzeRequest, AnalyzeTimeSeriesUseCase

STUCK_START = 120
STUCK_END = 131
WINDOW = 24


def _stuck_sensor_series() -> tuple[float, ...]:
    values: list[float] = []
    for hour in range(240):
        phase = 2 * math.pi * (hour % 24) / 24.0
        values.append(15.0 + 8.0 * math.sin(phase - math.pi / 2))
    for index in range(STUCK_START, STUCK_END + 1):
        values[index] = 23.0
    return tuple(values)


def _overlaps_stuck(start_index: int, length: int) -> bool:
    return start_index <= STUCK_END and start_index + length > STUCK_START


def test_stuck_sensor_discords_overlap_injected_region():
    report = AnalyzeTimeSeriesUseCase().execute(
        AnalyzeRequest(
            values=_stuck_sensor_series(), subsequence_length=WINDOW, top_k=3
        )
    )
    # top_k=3 but only two real anomalies exist after separation + dist>0 gate
    assert len(report.discords) == 2
    assert all(d.distance_to_nearest_neighbor > 0 for d in report.discords)
    assert all(_overlaps_stuck(d.start_index, d.length) for d in report.discords)
    assert [d.start_index for d in report.discords] == [100, 124]


def test_stuck_sensor_motifs_have_low_distance():
    series = TimeSeries(values=_stuck_sensor_series())
    profile = compute_matrix_profile(series, subsequence_length=WINDOW)
    motifs = top_motifs(profile, top_k=2)
    assert motifs
    assert all(m.distance_to_match < 1.0 for m in motifs)
    assert motifs[0].start_index % WINDOW == 0


def test_discord_preview_truncates_long_windows():
    discord = Discord(
        start_index=0,
        length=12,
        distance_to_nearest_neighbor=3.5,
        nearest_neighbor_index=18,
        values=tuple(float(i) for i in range(12)),
        nearest_neighbor_values=tuple(float(i) for i in range(100, 112)),
    )
    explanation = discord.explanation()
    assert "..." in explanation
    assert "values here are" in explanation


def test_orphan_discord_explanation_avoids_bogus_neighbor():
    discord = Discord(
        start_index=0,
        length=6,
        distance_to_nearest_neighbor=float("inf"),
        nearest_neighbor_index=-1,
        values=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        nearest_neighbor_values=(),
    )
    text = discord.explanation()
    assert "no valid neighbour" in text
    assert "exclusion zone" in text


def test_to_discord_skips_negative_neighbor_slice():
    series = TimeSeries(values=tuple(float(i) for i in range(20)))
    profile = MatrixProfile(
        entries=(
            MatrixProfileEntry(
                index=0,
                nearest_neighbor_distance=float("inf"),
                nearest_neighbor_index=-1,
            ),
        ),
        subsequence_length=4,
    )
    discords = top_discords(profile, series, top_k=1, min_separation=1)
    assert discords[0].nearest_neighbor_values == ()
    assert "no valid neighbour" in discords[0].explanation()
