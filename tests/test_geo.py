import pytest

from cycling_coach.geo import haversine_m


def test_one_degree_of_latitude_is_about_111_km():
    assert haversine_m(24.0, 120.0, 25.0, 120.0) == pytest.approx(111_195, rel=1e-3)


def test_same_point_is_zero():
    assert haversine_m(24.1, 120.7, 24.1, 120.7) == 0.0
