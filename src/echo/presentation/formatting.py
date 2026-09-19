"""Presentation helpers: human-readable analysis reports."""

from __future__ import annotations

from echo.use_cases.analyze_time_series import AnalysisReport


def format_report(report: AnalysisReport, subsequence_length: int) -> str:
    lines = [
        f"Analyzed {report.total_subsequences} subsequences of length {subsequence_length}.",
        "",
        "ANOMALIES (discords) — patterns unlike anything else in the series:",
    ]
    if not report.discords:
        lines.append("  none found")
    for discord in report.discords:
        lines.append(f"  - {discord.explanation()}")

    lines.append("")
    lines.append(
        "RECURRING PATTERNS (motifs) — patterns that repeat closely elsewhere:"
    )
    if not report.motifs:
        lines.append("  none found")
    for motif in report.motifs:
        lines.append(f"  - {motif.explanation()}")

    return "\n".join(lines)
