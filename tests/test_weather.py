import numpy as np
import pandas as pd
import pytest

from cycling_coach.weather import bearing_deg, headwind, linear_fit

M = 1 / 111_195


@pytest.mark.parametrize("dlat, dlon, expected", [(1, 0, 0), (0, 1, 90), (-1, 0, 180), (0, -1, 270)])
def test_bearing_cardinal_directions(dlat, dlon, expected):
    assert bearing_deg(24.0, 120.0, 24.0 + dlat * 0.01, 120.0 + dlon * 0.01) == pytest.approx(expected, abs=0.1)


def _northbound(n=60):
    return pd.DataFrame({"lat": 24.0 + np.arange(n) * 5 * M, "lon": np.full(n, 120.0)})


def test_wind_from_north_is_headwind_when_riding_north():
    hw = headwind(_northbound(), wind_speed=5.0, wind_from_deg=0.0, factor=1.0)
    assert hw.iloc[10:50].to_numpy() == pytest.approx(5.0)


def test_wind_from_south_is_tailwind_when_riding_north():
    hw = headwind(_northbound(), wind_speed=5.0, wind_from_deg=180.0, factor=1.0)
    assert hw.iloc[10:50].to_numpy() == pytest.approx(-5.0)


def test_crosswind_is_zero():
    hw = headwind(_northbound(), wind_speed=5.0, wind_from_deg=90.0, factor=1.0)
    assert hw.iloc[10:50].to_numpy() == pytest.approx(0.0, abs=1e-9)


def test_linear_fit_recovers_slope():
    x = np.arange(50.0)
    fit = linear_fit(x, 2.0 + 0.5 * x, n_boot=50)
    assert fit["slope"] == pytest.approx(0.5)
    assert fit["slope_lo"] <= 0.5 <= fit["slope_hi"]
