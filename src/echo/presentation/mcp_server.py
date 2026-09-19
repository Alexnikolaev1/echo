"""Presentation layer: MCP server for Echo."""

from mcp.server.fastmcp import FastMCP

from echo.entities.time_series import (
    InvalidSubsequenceLengthError,
    NonFiniteValueError,
    TooFewPointsError,
)
from echo.infrastructure.csv_series_loader import (
    ColumnNotFoundError,
    EmptyColumnError,
    NonNumericValueError,
)
from echo.presentation.composition_root import build_application
from echo.presentation.formatting import format_report
from echo.use_cases.analyze_time_series import AnalyzeRequest, InvalidTopKError

mcp = FastMCP("echo")
app = build_application()

_INPUT_ERRORS = (
    TooFewPointsError,
    InvalidSubsequenceLengthError,
    NonFiniteValueError,
    InvalidTopKError,
)
_LOAD_ERRORS = (
    ColumnNotFoundError,
    NonNumericValueError,
    EmptyColumnError,
    FileNotFoundError,
)


def _analyze_values(
    values: tuple[float, ...], subsequence_length: int, top_k: int
) -> str:
    try:
        report = app.analyze.execute(
            AnalyzeRequest(
                values=values, subsequence_length=subsequence_length, top_k=top_k
            )
        )
    except _INPUT_ERRORS as exc:
        return f"Invalid input: {exc}"
    return format_report(report, subsequence_length)


@mcp.tool()
def analyze_series(values: list[float], subsequence_length: int, top_k: int = 3) -> str:
    """Find explainable anomalies and recurring patterns in a time series
    using the Matrix Profile algorithm. Each anomaly is explained by
    showing its nearest match elsewhere in the series (or the lack of one),
    not just flagged as a bare score.

    Args:
        values: The time series as a flat list of numbers, in order.
        subsequence_length: The length of the pattern window to compare
            (e.g. 12 for a daily cycle sampled hourly). Must be at least 3
            and at most half the series length.
        top_k: How many top anomalies/patterns to report.
    """
    return _analyze_values(tuple(values), subsequence_length, top_k)


@mcp.tool()
def analyze_csv(
    file_path: str, column: str, subsequence_length: int, top_k: int = 3
) -> str:
    """Same as analyze_series, but loads the values from a numeric column
    of a CSV file.

    Args:
        file_path: Absolute path to the CSV file.
        column: Name of the numeric column to analyze.
        subsequence_length: Pattern window length (see analyze_series).
        top_k: How many top anomalies/patterns to report.
    """
    try:
        values = app.loader.load_column(file_path, column)
    except _LOAD_ERRORS as exc:
        return f"Could not load data: {exc}"
    return _analyze_values(values, subsequence_length, top_k)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
