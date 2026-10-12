import numpy as np
import pandas as pd
import pytest

from cycling_coach.cleaning import (
    detect_stops,
    distance_window_grade,
    elevation_gain_m,
    hold_position_while_stopped,
    remove_gps_spikes,
    smooth_altitude,
)

DEG_PER_M = 1 / 111_195  # degrees of latitude per metre


def _ride(n, speed=5.0, paused=(), filled=()):
    """Rider heading north at constant speed; seconds in `paused`/`filled` stand still."""
    moving = np.ones(n)
    for i in [*paused, *filled]:
        moving[i] = 0
    dist = np.concatenate([[0], np.cumsum(moving[1:] * speed)])
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01 00:00", periods=n, freq="s", tz="UTC"),
        "lat": 24.0 + dist * DEG_PER_M,
        "lon": 120.0,
        "distance_m": dist,
        "speed_mps": moving * speed,
        "altitude_m": 100.0,
        "is_paused": [i in paused for i in range(n)],
        "is_filled": [i in filled for i in range(n)],
    })
    df.loc[list(filled), ["lat", "lon", "speed_mps"]] = np.nan
    return df


def test_remove_gps_spikes_drops_and_refills_outlier():
    df = _ride(30)
    df.loc[15, "lat"] += 0.01  # ~1 km jump
    out = remove_gps_spikes(df)
    assert out.loc[15, "lat"] == pytest.approx(_ride(30).loc[15, "lat"], abs=1e-6)


def test_paused_run_is_one_stop_event():
    df = _ride(60, paused=range(20, 50))
    flags, stops = detect_stops(df)
    assert len(stops) == 1
    assert stops.loc[0, "kind"] == "pause"
    assert stops.loc[0, "duration_s"] >= 30
    assert flags.loc[25, "is_stopped"] and not flags.loc[25, "is_moving"]
    assert flags.loc[5, "is_moving"]


def test_gap_while_moving_is_not_a_stop():
    df = _ride(60)
    df.loc[20:39, ["lat", "lon", "speed_mps"]] = np.nan
    df.loc[20:39, "is_filled"] = True
    flags, stops = detect_stops(df)
    assert stops.empty
    assert flags.loc[30, "is_gap"]


def test_gap_without_movement_is_a_stop():
    df = _ride(60, filled=range(20, 40))
    flags, stops = detect_stops(df)
    assert len(stops) == 1
    assert stops.loc[0, "kind"] == "slow"


def test_short_stops_close_together_are_merged():
    df = _ride(60, paused=[*range(10, 20), *range(22, 30)])
    _, stops = detect_stops(df)
    assert len(stops) == 1


def test_hold_position_while_stopped_fills_location():
    df = _ride(60, filled=range(20, 40))
    flags, _ = detect_stops(df)
    out = hold_position_while_stopped(flags)
    assert out.loc[30, "lat"] == out.loc[19, "lat"]
    assert out.loc[30, "speed_mps"] == 0.0


def test_grade_of_constant_slope():
    dist = np.arange(0, 500, 5.0)
    df = pd.DataFrame({"distance_m": dist, "altitude_m": 100 + 0.06 * dist})
    out = distance_window_grade(smooth_altitude(df))
    assert out["grade"].iloc[20:80].to_numpy() == pytest.approx(0.06, abs=1e-6)


def test_grade_is_clipped():
    df = pd.DataFrame({"distance_m": np.arange(0, 200, 5.0), "altitude_m": np.arange(40) * 5.0})
    out = distance_window_grade(smooth_altitude(df))
    assert out["grade"].max() <= 0.25


def test_smooth_altitude_removes_spike():
    alt = pd.Series([100.0] * 20)
    alt[10] = 150.0
    out = smooth_altitude(pd.DataFrame({"altitude_m": alt}))
    assert out["altitude_smooth_m"].max() == pytest.approx(100.0)


def test_elevation_gain_ignores_small_wiggles():
    assert elevation_gain_m(pd.Series([100, 100.5, 100, 100.5, 110, 109, 120])) == pytest.approx(20)
