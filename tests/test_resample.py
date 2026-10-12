from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from cycling_coach.io.common import SENSOR_COLUMNS
from cycling_coach.resample import (
    add_distance_if_missing,
    add_raw_grade,
    add_speed_if_missing,
    interpolate_short_gaps,
    mark_pauses,
    to_1hz,
)

T0 = pd.Timestamp("2025-01-01 08:00:00", tz="UTC")


def _samples(seconds, **columns) -> pd.DataFrame:
    df = pd.DataFrame({"timestamp": [T0 + timedelta(seconds=s) for s in seconds]})
    for column in SENSOR_COLUMNS:
        df[column] = columns.get(column, np.nan)
    return df


def test_interpolate_short_gaps_fills_only_short_runs():
    s = pd.Series([0.0, np.nan, 2.0, np.nan, np.nan, np.nan, 6.0])
    out = interpolate_short_gaps(s, max_gap=2)
    assert out[1] == 1.0
    assert out[3:6].isna().all()


def test_to_1hz_fills_short_gap_and_flags_it():
    df = _samples([0, 2], speed_mps=[4.0, 6.0])
    out = to_1hz(df)
    assert len(out) == 3
    assert out.loc[1, "speed_mps"] == 5.0
    assert out["is_filled"].tolist() == [False, True, False]


def test_to_1hz_leaves_long_gap_empty():
    df = _samples([0, 30], speed_mps=[4.0, 6.0])
    out = to_1hz(df, max_gap_s=5)
    assert len(out) == 31
    assert out.loc[1:29, "speed_mps"].isna().all()


def test_to_1hz_drops_duplicate_seconds():
    df = _samples([0, 0.4, 1], speed_mps=[1.0, 9.0, 2.0])
    out = to_1hz(df)
    assert out["speed_mps"].tolist() == [1.0, 2.0]


def test_to_1hz_rejects_implausible_duration():
    df = _samples([0, 3 * 24 * 3600])
    with pytest.raises(ValueError):
        to_1hz(df)


def test_mark_pauses():
    df = to_1hz(_samples(range(10)))
    out = mark_pauses(df, [(T0 + timedelta(seconds=3), T0 + timedelta(seconds=6))])
    assert out["is_paused"].tolist() == [False] * 3 + [True] * 3 + [False] * 4


def test_add_distance_if_missing_uses_gps():
    df = _samples([0, 1], lat=[24.0, 24.001], lon=[120.0, 120.0])
    out = add_distance_if_missing(df)
    assert out["distance_m"].tolist() == pytest.approx([0.0, 111.2], abs=0.1)


def test_add_distance_keeps_device_distance():
    df = _samples([0, 1], lat=[24.0, 24.001], lon=[120.0, 120.0], distance_m=[0.0, 100.0])
    assert add_distance_if_missing(df)["distance_m"].tolist() == [0.0, 100.0]


def test_add_speed_if_missing_from_distance():
    df = to_1hz(_samples([0, 1, 2], distance_m=[0.0, 5.0, 11.0]))
    assert add_speed_if_missing(df)["speed_mps"].tolist()[1:] == [5.0, 6.0]


def test_add_raw_grade():
    df = to_1hz(_samples([0, 1, 2], distance_m=[0.0, 10.0, 10.5], altitude_m=[100.0, 101.0, 101.0]))
    grade = add_raw_grade(df)["grade_raw"]
    assert grade[1] == pytest.approx(0.10)
    assert np.isnan(grade[2])  # moved less than 1 m: grade undefined
