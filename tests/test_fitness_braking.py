import numpy as np
import pandas as pd
import pytest

from cycling_coach.braking import hard_brakes
from cycling_coach.fitness import commute_efficiency


def _ride(speed, hr=150.0):
    n = len(speed)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="s", tz="UTC"),
        "lat": 24.0, "lon": 120.0, "speed_mps": speed, "heart_rate_bpm": hr,
        "is_moving": True, "is_stopped": False,
    })


def test_metres_per_beat():
    ride = _ride(np.full(600, 6.0), hr=120.0)  # 6 m/s at 2 beats/s -> 3 m per beat
    assert commute_efficiency(ride)["metres_per_beat"] == pytest.approx(3.0)


def test_watts_per_bpm():
    ride = _ride(np.full(600, 6.0), hr=120.0)
    out = commute_efficiency(ride, pd.Series(150.0, index=ride.index))
    assert out["watts_per_bpm"] == pytest.approx(1.25)


def test_efficiency_needs_heart_rate():
    ride = _ride(np.full(600, 6.0), hr=np.nan)
    assert np.isnan(commute_efficiency(ride)["metres_per_beat"])


def test_hard_brake_detected_once():
    speed = np.concatenate([np.full(20, 8.0), [6.0, 3.5, 2.0, 1.0], np.full(20, 0.5)])
    events = hard_brakes(_ride(speed))
    assert len(events) == 1
    assert events.loc[0, "drop_mps"] == pytest.approx(4.5)  # 8 -> 3.5 m/s within 2 s


def test_gentle_slowing_and_gps_glitch_ignored():
    gentle = np.linspace(8, 0, 40)               # 0.2 m/s^2
    glitch = np.concatenate([np.full(10, 8.0), [-6.0], np.full(10, 8.0)])  # 14 m/s drop
    assert hard_brakes(_ride(gentle)).empty
    assert hard_brakes(_ride(glitch)).empty
