import numpy as np
import pandas as pd
import pytest

from cycling_coach.calibration import error_metrics, fit_cda_crr, predict_windows
from cycling_coach.physics import (
    ETA,
    Setup,
    aero_power,
    air_density,
    gravity_power,
    joules_to_kcal,
    leg_power,
    rolling_power,
    stop_energy_j,
)


def test_air_density_sea_level():
    assert air_density(0, 15) == pytest.approx(1.225, abs=0.002)


def test_air_density_drops_with_altitude():
    assert air_density(3000, 15) < 0.8 * air_density(0, 15)


def test_gravity_power_known_value():
    # 77 kg at 10 km/h up 8%: about 167 W
    assert gravity_power(77, 10 / 3.6, 0.08) == pytest.approx(167, abs=1)


def test_rolling_power():
    assert rolling_power(80, 0.005, 10.0, 0.0) == pytest.approx(0.005 * 80 * 9.80665 * 10)


def test_aero_power_cubic_and_headwind():
    assert aero_power(0.3, 1.2, 10.0) == pytest.approx(180.0)
    assert aero_power(0.3, 1.2, 10.0, headwind=2.0) == pytest.approx(0.5 * 1.2 * 0.3 * 144 * 10)


def test_leg_power_flat_steady_is_drag_plus_rolling_over_eta():
    s = Setup(80, 0.3, 0.005)
    expected = (aero_power(0.3, 1.2, 8.0) + rolling_power(80, 0.005, 8.0, 0.0)) / ETA
    assert leg_power(s, 8.0, 0.0, 0.0, 1.2) == pytest.approx(expected)


def test_descending_needs_negative_power():
    assert leg_power(Setup(80, 0.3, 0.005), 10.0, -0.08, 0.0, 1.2) < 0


def test_stop_energy_and_kcal():
    assert stop_energy_j(80, 7.0) == pytest.approx(0.5 * 80 * 49 / ETA)
    assert joules_to_kcal(4184 * 0.24) == pytest.approx(1.0)


def test_fit_recovers_known_cda_crr():
    rng = np.random.default_rng(0)
    n = 500
    x_aero = 0.5 * 1.2 * rng.uniform(4, 12, n) ** 3
    x_roll = 80 * 9.80665 * rng.uniform(4, 12, n)
    y = 0.32 * x_aero + 0.005 * x_roll + rng.normal(0, 5, n)
    cda, crr = fit_cda_crr(pd.DataFrame({"x_aero": x_aero, "x_roll": x_roll, "y": y}))
    assert cda == pytest.approx(0.32, abs=0.01)
    assert crr == pytest.approx(0.005, abs=0.0005)


def test_predict_windows_matches_measured_when_model_is_exact():
    setup = Setup(80, 0.3, 0.005)
    w = pd.DataFrame({"x_aero": [600.0], "x_roll": [4000.0], "power": [250.0]})
    w["y"] = setup.cda * w["x_aero"] + setup.crr * w["x_roll"]  # no gravity/accel part
    w["power"] = w["y"] / ETA
    assert predict_windows(w, setup).iloc[0] == pytest.approx(w["power"].iloc[0])


def test_error_metrics():
    e = error_metrics(pd.Series([110.0, 90.0]), pd.Series([100.0, 100.0]))
    assert e["mae_w"] == 10 and e["bias_w"] == 0 and e["mape"] == pytest.approx(0.1)
