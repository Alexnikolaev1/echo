from itertools import pairwise

from echo.entities.findings import top_discords, top_motifs
from echo.entities.matrix_profile import compute_matrix_profile
from echo.entities.time_series import TimeSeries


def test_top_discords_deduplicates_adjacent_windows_of_same_anomaly():
    """The exact problem observed in development: without de-duplication,
    the top-k list would just be k adjacent windows all covering the SAME
    single anomaly, which is useless — a user asking for 'top 3 anomalies'
    wants 3 DIFFERENT anomalies, not the same one three times."""
    import random

    random.seed(7)
    base_pattern = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1] * 8
    series_values = [v + random.gauss(0, 0.05) for v in base_pattern]
    anomaly_start = 50
    for k in range(12):
        series_values[anomaly_start + k] = 5.0 if k % 2 == 0 else -5.0

    series = TimeSeries(values=tuple(series_values))
    profile = compute_matrix_profile(series, subsequence_length=12)

    discords = top_discords(profile, series, top_k=3)

    # the 3 reported discords must be spread apart, not all clustered
    # within a couple of indices of each other
    indices = sorted(d.start_index for d in discords)
    for a, b in pairwise(indices):
        assert b - a >= 12  # at least one subsequence_length apart


def test_discord_explanation_includes_index_and_both_value_sets():
    series = TimeSeries(values=tuple(float(i % 7) for i in range(40)))
    series_values = list(series.values)
    series_values[20:26] = [100.0, -100.0, 100.0, -100.0, 100.0, -100.0]
    series = TimeSeries(values=tuple(series_values))
    profile = compute_matrix_profile(series, subsequence_length=6)

    discords = top_discords(profile, series, top_k=1)
    explanation = discords[0].explanation()

    assert "index" in explanation
    assert "values here are" in explanation


def test_motif_explanation_mentions_match_index():
    pattern = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0] * 6
    series = TimeSeries(values=tuple(pattern))
    profile = compute_matrix_profile(series, subsequence_length=6)
    motifs = top_motifs(profile, top_k=1)
    explanation = motifs[0].explanation()
    assert "closely repeats the pattern at index" in explanation
    assert "distance=" in explanation


def test_top_motifs_finds_the_lowest_distance_entries():
    pattern = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0] * 6
    series = TimeSeries(values=tuple(pattern))
    profile = compute_matrix_profile(series, subsequence_length=6)

    motifs = top_motifs(profile, top_k=2)

    assert len(motifs) == 2
    assert all(m.distance_to_match < 0.5 for m in motifs)


def test_top_motifs_also_deduplicates_by_separation():
    pattern = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0] * 6
    series = TimeSeries(values=tuple(pattern))
    profile = compute_matrix_profile(series, subsequence_length=6)

    motifs = top_motifs(profile, top_k=3)
    indices = sorted(m.start_index for m in motifs)
    for a, b in pairwise(indices):
        assert b - a >= 6


def test_top_discords_does_not_pad_with_zero_distance_matches():
    """Perfectly repeating series: every window matches — zero discords, not
    top_k fake anomalies with distance 0."""
    pattern = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0] * 6
    series = TimeSeries(values=tuple(pattern))
    profile = compute_matrix_profile(series, subsequence_length=6)
    assert top_discords(profile, series, top_k=3) == []
