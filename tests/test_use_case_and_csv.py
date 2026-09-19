import tempfile
from pathlib import Path

import pytest

from echo.entities.time_series import InvalidSubsequenceLengthError
from echo.infrastructure.csv_series_loader import (
    ColumnNotFoundError,
    CsvSeriesLoader,
    EmptyColumnError,
    NonNumericValueError,
)
from echo.use_cases.analyze_time_series import (
    AnalyzeRequest,
    AnalyzeTimeSeriesUseCase,
    InvalidTopKError,
)


def test_use_case_returns_discords_and_motifs():
    import random

    random.seed(3)
    base_pattern = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1] * 8
    values = [v + random.gauss(0, 0.05) for v in base_pattern]
    for k in range(12):
        values[50 + k] = 5.0 if k % 2 == 0 else -5.0

    use_case = AnalyzeTimeSeriesUseCase()
    report = use_case.execute(
        AnalyzeRequest(values=tuple(values), subsequence_length=12, top_k=2)
    )

    assert len(report.discords) == 2
    assert len(report.motifs) == 2
    assert (
        report.discords[0].distance_to_nearest_neighbor
        > report.motifs[0].distance_to_match
    )


def test_use_case_propagates_invalid_subsequence_length():
    use_case = AnalyzeTimeSeriesUseCase()
    with pytest.raises(InvalidSubsequenceLengthError):
        use_case.execute(AnalyzeRequest(values=tuple(range(10)), subsequence_length=2))


def test_use_case_rejects_non_positive_top_k():
    use_case = AnalyzeTimeSeriesUseCase()
    with pytest.raises(InvalidTopKError):
        use_case.execute(
            AnalyzeRequest(values=tuple(range(20)), subsequence_length=4, top_k=0)
        )


def _write_csv(dir_path: Path, rows: list[dict]) -> str:
    import csv

    path = dir_path / "data.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def test_csv_loader_reads_numeric_column():
    tmp = Path(tempfile.mkdtemp())
    csv_path = _write_csv(tmp, [{"time": i, "temp": 20.0 + i * 0.1} for i in range(10)])

    loader = CsvSeriesLoader()
    values = loader.load_column(csv_path, "temp")

    assert len(values) == 10
    assert values[0] == 20.0
    assert abs(values[-1] - 20.9) < 1e-9


def test_csv_loader_rejects_unknown_column():
    tmp = Path(tempfile.mkdtemp())
    csv_path = _write_csv(tmp, [{"time": 1, "temp": 20.0}])

    loader = CsvSeriesLoader()
    with pytest.raises(ColumnNotFoundError):
        loader.load_column(csv_path, "pressure")


def test_csv_loader_rejects_non_numeric_value():
    tmp = Path(tempfile.mkdtemp())
    csv_path = _write_csv(tmp, [{"time": 1, "temp": "not-a-number"}])

    loader = CsvSeriesLoader()
    with pytest.raises(NonNumericValueError):
        loader.load_column(csv_path, "temp")


def test_csv_loader_rejects_empty_column():
    tmp = Path(tempfile.mkdtemp())
    path = tmp / "empty.csv"
    path.write_text("temp\n", encoding="utf-8")
    with pytest.raises(EmptyColumnError):
        CsvSeriesLoader().load_column(str(path), "temp")


def test_csv_loader_rejects_nan_and_inf():
    tmp = Path(tempfile.mkdtemp())
    for bad in ("nan", "inf", "-inf"):
        csv_path = _write_csv(tmp, [{"temp": bad}])
        with pytest.raises(NonNumericValueError):
            CsvSeriesLoader().load_column(csv_path, "temp")
