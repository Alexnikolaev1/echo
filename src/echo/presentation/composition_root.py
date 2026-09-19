"""Composition root for Echo."""

from __future__ import annotations

from dataclasses import dataclass

from echo.infrastructure.csv_series_loader import CsvSeriesLoader
from echo.interfaces.ports import SeriesLoader
from echo.use_cases.analyze_time_series import AnalyzeTimeSeriesUseCase


@dataclass(frozen=True, slots=True)
class Application:
    analyze: AnalyzeTimeSeriesUseCase
    loader: SeriesLoader


def build_application() -> Application:
    return Application(analyze=AnalyzeTimeSeriesUseCase(), loader=CsvSeriesLoader())
