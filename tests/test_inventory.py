import numpy as np
import pandas as pd
import pytest

from cycling_coach.inventory import device_group, find_duplicates, is_in_taiwan, summarize_ride


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("bryton 1901", "Bryton Rider 420"),
        ("garmin fr255_music", "Garmin Forerunner 255"),
        ("unknown", "Xiaomi watch"),
        ("gpx", "GPX export"),
        ("strava 102", "Other"),
        (None, "Other"),
    ],
)
def test_device_group(raw, expected):
    assert device_group(raw) == expected


def test_summarize_ride_coverage_ignores_filled_rows():
    n = 10
    ride = pd.DataFrame({
        "lat": [24.1] * n, "lon": [120.7] * n, "altitude_m": [80.0] * n,
        "distance_m": np.arange(n) * 5.0, "power_w": [200.0] * 4 + [np.nan] * 6,
        "heart_rate_bpm": [np.nan] * n, "cadence_rpm": [90.0] * n,
        "temperature_c": [25.0] * n,
        "is_filled": [False] * 8 + [True] * 2,
        "is_paused": [False] * 6 + [True] * 4,
    })
    s = summarize_ride(ride)
    assert s["recorded_s"] == 8
    assert s["paused_s"] == 4
    assert s["ride_distance_m"] == 45.0
    assert s["cov_power_w"] == 0.5
    assert not s["has_power_w"]  # needs more than half
    assert s["has_cadence_rpm"]
    assert not s["has_heart_rate_bpm"]


def test_is_in_taiwan():
    lat = pd.Series([24.15, 5.37, np.nan])
    lon = pd.Series([120.68, 100.30, np.nan])
    assert is_in_taiwan(lat, lon).tolist() == [True, False, False]


def _rides(rows):
    df = pd.DataFrame(rows, columns=["activity_id", "start", "elapsed_s", "recorded_s"])
    df["start_time"] = pd.Timestamp("2025-10-09 22:00", tz="UTC") + pd.to_timedelta(df["start"], unit="s")
    return df


def test_find_duplicates_keeps_ride_with_more_data():
    rides = _rides([
        (1, 0, 3600, 3000),    # watch
        (2, 60, 3600, 3500),   # bike computer, same ride, more data -> keep
        (3, 7200, 1800, 1800), # separate ride later
    ])
    assert find_duplicates(rides).tolist() == [2, pd.NA, pd.NA]


def test_find_duplicates_ignores_back_to_back_rides():
    rides = _rides([(1, 0, 600, 600), (2, 600, 600, 600)])
    assert find_duplicates(rides).isna().all()
