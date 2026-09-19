import pytest

from echo.entities.time_series import (
    InvalidSubsequenceLengthError,
    LengthMismatchError,
    NonFiniteValueError,
    TimeSeries,
    TooFewPointsError,
    z_normalize,
    z_normalized_distance,
)


def test_time_series_rejects_too_few_points():
    with pytest.raises(TooFewPointsError):
        TimeSeries(values=(1.0, 2.0))


def test_time_series_rejects_nan():
    with pytest.raises(NonFiniteValueError):
        TimeSeries(values=(1.0, 2.0, 3.0, float("nan")))


def test_time_series_rejects_inf():
    with pytest.raises(NonFiniteValueError):
        TimeSeries(values=(1.0, 2.0, 3.0, float("inf")))


def test_time_series_len_matches_values():
    series = TimeSeries(values=tuple(range(10)))
    assert len(series) == 10


def test_z_normalize_produces_zero_mean_unit_variance():
    normalized = z_normalize((1.0, 2.0, 3.0, 4.0, 5.0))
    mean = sum(normalized) / len(normalized)
    variance = sum((v - mean) ** 2 for v in normalized) / len(normalized)
    assert abs(mean) < 1e-9
    assert abs(variance - 1.0) < 1e-9


def test_z_normalize_constant_series_is_all_zeros_not_a_crash():
    normalized = z_normalize((5.0, 5.0, 5.0, 5.0))
    assert normalized == (0.0, 0.0, 0.0, 0.0)


def test_distance_is_zero_for_identical_shapes_even_with_different_offset_and_scale():
    """The whole point of z-normalization: [1,2,3,4] and [10,20,30,40] are
    the SAME shape (just scaled), so their distance should be ~0."""
    a = (1.0, 2.0, 3.0, 4.0)
    b = (10.0, 20.0, 30.0, 40.0)
    assert z_normalized_distance(a, b) < 1e-9


def test_distance_is_zero_for_shifted_shape():
    a = (1.0, 2.0, 3.0, 4.0)
    b = (101.0, 102.0, 103.0, 104.0)  # same shape, offset by +100
    assert z_normalized_distance(a, b) < 1e-9


def test_distance_is_large_for_genuinely_different_shapes():
    a = (1.0, 2.0, 3.0, 4.0)  # rising
    b = (4.0, 3.0, 2.0, 1.0)  # falling — opposite shape
    assert z_normalized_distance(a, b) > 1.0


def test_distance_rejects_length_mismatch():
    with pytest.raises(LengthMismatchError):
        z_normalized_distance((1.0, 2.0, 3.0), (1.0, 2.0))


def test_num_subsequences_rejects_too_short_window():
    series = TimeSeries(values=tuple(range(20)))
    with pytest.raises(InvalidSubsequenceLengthError):
        series.num_subsequences(2)


def test_num_subsequences_rejects_window_over_half_length():
    series = TimeSeries(values=tuple(range(20)))
    with pytest.raises(InvalidSubsequenceLengthError):
        series.num_subsequences(11)


def test_num_subsequences_correct_count():
    series = TimeSeries(values=tuple(range(20)))
    assert series.num_subsequences(5) == 16  # 20 - 5 + 1
