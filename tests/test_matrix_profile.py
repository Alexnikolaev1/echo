import math
import random

from echo.entities.matrix_profile import compute_matrix_profile
from echo.entities.time_series import TimeSeries

ANOMALY_START = 50
ANOMALY_LENGTH = 12
WINDOW = 12


def _series_with_injected_anomaly() -> TimeSeries:
    random.seed(42)
    base = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1] * 8
    values = [v + random.gauss(0, 0.05) for v in base]
    for k in range(ANOMALY_LENGTH):
        values[ANOMALY_START + k] = 5.0 if k % 2 == 0 else -5.0
    return TimeSeries(values=tuple(values))


def _overlaps_anomaly(index: int) -> bool:
    window_end = index + WINDOW
    anomaly_end = ANOMALY_START + ANOMALY_LENGTH
    return index < anomaly_end and window_end > ANOMALY_START


def test_profile_has_one_entry_per_subsequence():
    series = TimeSeries(values=tuple(range(20)))
    profile = compute_matrix_profile(series, subsequence_length=4)
    assert len(profile.entries) == series.num_subsequences(4)


def test_without_exclusion_zone_distances_would_trivially_be_near_zero():
    """Sanity check on the exclusion zone's necessity: a monotonically
    increasing series has near-identical ADJACENT subsequences (e.g.
    [0,1,2,3] and [1,2,3,4] are almost the same shape). If the exclusion
    zone didn't exclude near-self matches, every entry's nearest neighbour
    would be its immediate neighbour with a tiny distance, making the
    profile useless for finding real structure. We verify the exclusion
    zone actually pushes matches away from trivially adjacent indices."""
    series = TimeSeries(values=tuple(float(i) for i in range(30)))
    profile = compute_matrix_profile(series, subsequence_length=6)
    exclusion_radius = 6 // 2
    for entry in profile.entries:
        assert abs(entry.index - entry.nearest_neighbor_index) > exclusion_radius


def test_repeating_pattern_produces_low_distance_motifs():
    """A perfectly repeating pattern should have near-zero distance
    everywhere — every window looks like some other window."""
    pattern = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0] * 6
    series = TimeSeries(values=tuple(pattern))
    profile = compute_matrix_profile(series, subsequence_length=6)
    max_distance = max(e.nearest_neighbor_distance for e in profile.entries)
    assert max_distance < 0.5  # very low — everything matches something


def test_injected_anomaly_produces_high_distance_cluster_around_it():
    """Highest-distance entries must overlap the true injected anomaly."""
    profile = compute_matrix_profile(
        _series_with_injected_anomaly(), subsequence_length=WINDOW
    )
    top_5 = profile.sorted_by_distance_desc()[:5]
    for entry in top_5:
        assert _overlaps_anomaly(
            entry.index
        ), f"index {entry.index} does not overlap the true anomaly region"


def test_sorted_by_distance_asc_and_desc_are_consistent():
    """Both orderings share the same entries and are monotonic."""
    random.seed(11)
    values = tuple(math.sin(i * 0.37) + random.gauss(0, 0.01) for i in range(40))
    profile = compute_matrix_profile(TimeSeries(values=values), subsequence_length=5)
    asc = profile.sorted_by_distance_asc()
    desc = profile.sorted_by_distance_desc()
    assert {e.index for e in asc} == {e.index for e in desc}
    asc_distances = [e.nearest_neighbor_distance for e in asc]
    desc_distances = [e.nearest_neighbor_distance for e in desc]
    assert asc_distances == sorted(asc_distances)
    assert desc_distances == sorted(desc_distances, reverse=True)
