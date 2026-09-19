import tempfile
from pathlib import Path

from echo.presentation.composition_root import build_application
from echo.presentation.formatting import format_report
from echo.presentation.mcp_server import analyze_csv, analyze_series
from echo.use_cases.analyze_time_series import AnalysisReport


def _repeating_with_spike() -> list[float]:
    import random

    random.seed(3)
    base = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1] * 8
    values = [v + random.gauss(0, 0.05) for v in base]
    for k in range(12):
        values[50 + k] = 5.0 if k % 2 == 0 else -5.0
    return values


def _write_csv(rows: list[dict]) -> str:
    import csv

    path = Path(tempfile.mkdtemp()) / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def test_build_application_wires_analyze_and_loader():
    app = build_application()
    assert app.analyze is not None
    assert app.loader is not None


def test_analyze_series_reports_anomalies_and_motifs():
    text = analyze_series(_repeating_with_spike(), subsequence_length=12, top_k=2)
    assert "ANOMALIES (discords)" in text
    assert "RECURRING PATTERNS (motifs)" in text
    assert "Subsequence at" in text


def test_analyze_series_rejects_too_few_points():
    text = analyze_series([1.0, 2.0], subsequence_length=3)
    assert text.startswith("Invalid input:")


def test_analyze_series_rejects_bad_window():
    text = analyze_series(list(range(20)), subsequence_length=2)
    assert text.startswith("Invalid input:")


def test_analyze_series_rejects_nan():
    text = analyze_series([1.0, 2.0, 3.0, float("nan")], subsequence_length=3)
    assert text.startswith("Invalid input:")


def test_analyze_series_rejects_bad_top_k():
    text = analyze_series(list(range(20)), subsequence_length=4, top_k=0)
    assert text.startswith("Invalid input:")


def test_analyze_csv_loads_and_analyzes():
    values = _repeating_with_spike()
    csv_path = _write_csv([{"temp": v} for v in values])
    text = analyze_csv(csv_path, "temp", subsequence_length=12, top_k=2)
    assert "ANOMALIES (discords)" in text


def test_analyze_csv_reports_load_errors():
    text = analyze_csv("no_such_file.csv", "temp", subsequence_length=6)
    assert text.startswith("Could not load data:")


def test_analyze_csv_reports_unknown_column():
    csv_path = _write_csv([{"temp": 1.0} for _ in range(10)])
    text = analyze_csv(csv_path, "pressure", subsequence_length=3)
    assert text.startswith("Could not load data:")


def test_analyze_csv_reports_non_numeric_column():
    csv_path = _write_csv([{"temp": "bad"} for _ in range(10)])
    text = analyze_csv(csv_path, "temp", subsequence_length=3)
    assert text.startswith("Could not load data:")


def test_analyze_csv_reports_empty_column():
    path = Path(tempfile.mkdtemp()) / "empty.csv"
    path.write_text("temp\n", encoding="utf-8")
    text = analyze_csv(str(path), "temp", subsequence_length=3)
    assert text.startswith("Could not load data:")


def test_format_report_handles_empty_findings():
    text = format_report(
        AnalysisReport(discords=[], motifs=[], total_subsequences=0),
        subsequence_length=6,
    )
    assert "none found" in text
