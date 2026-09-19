from hypothesis import assume, given, settings
from hypothesis import strategies as st

from echo.entities.matrix_profile import compute_matrix_profile
from echo.entities.time_series import TimeSeries, z_normalized_distance

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
window = st.lists(finite, min_size=4, max_size=12)


@given(window)
@settings(max_examples=40, deadline=None)
def test_distance_to_self_is_zero(values: list[float]) -> None:
    seq = tuple(values)
    assert z_normalized_distance(seq, seq) == 0.0


@given(window)
@settings(max_examples=40, deadline=None)
def test_distance_is_symmetric_for_equal_length(values: list[float]) -> None:
    a = tuple(values)
    b = tuple(reversed(values))
    ab = z_normalized_distance(a, b)
    ba = z_normalized_distance(b, a)
    assert ab >= 0.0
    assert abs(ab - ba) < 1e-9


@given(window, st.floats(0.1, 10.0), finite)
@settings(max_examples=40, deadline=None)
def test_distance_invariant_to_positive_scale_and_shift(
    values: list[float], scale: float, shift: float
) -> None:
    a = tuple(values)
    b = tuple(scale * v + shift for v in values)
    assert z_normalized_distance(a, b) < 1e-6


@given(st.lists(finite, min_size=16, max_size=36), st.integers(3, 8))
@settings(max_examples=25, deadline=None)
def test_exclusion_zone_always_respected(values: list[float], m: int) -> None:
    assume(m <= len(values) // 2)
    profile = compute_matrix_profile(TimeSeries(values=tuple(values)), m)
    radius = m // 2
    for entry in profile.entries:
        if entry.nearest_neighbor_index < 0:
            continue
        assert abs(entry.index - entry.nearest_neighbor_index) > radius
