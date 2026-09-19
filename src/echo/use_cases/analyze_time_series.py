"""Use case: AnalyzeTimeSeries."""

from __future__ import annotations

from dataclasses import dataclass

from echo.entities.findings import Discord, Motif, top_discords, top_motifs
from echo.entities.matrix_profile import compute_matrix_profile
from echo.entities.time_series import TimeSeries


class InvalidTopKError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AnalyzeRequest:
    values: tuple[float, ...]
    subsequence_length: int
    top_k: int = 3


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    discords: list[Discord]
    motifs: list[Motif]
    total_subsequences: int


class AnalyzeTimeSeriesUseCase:
    def execute(self, request: AnalyzeRequest) -> AnalysisReport:
        if request.top_k < 1:
            raise InvalidTopKError("top_k must be at least 1.")
        series = TimeSeries(values=request.values)
        profile = compute_matrix_profile(series, request.subsequence_length)
        return AnalysisReport(
            discords=top_discords(profile, series, request.top_k),
            motifs=top_motifs(profile, request.top_k),
            total_subsequences=len(profile.entries),
        )
