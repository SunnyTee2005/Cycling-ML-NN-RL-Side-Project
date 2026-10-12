import numpy as np
import pandas as pd
import pytest

from cycling_coach.traffic import (
    find_intersections,
    forecast_wait,
    green_wave_pairs,
    intersection_stats,
    passes_for_ride,
    relevant_passes,
    shrunken_mean,
)

M = 1 / 111_195


def test_find_intersections_merges_queue_positions():
    rng = np.random.default_rng(1)
    lat = np.concatenate([24.0 + rng.normal(0, 5 * M, 10),            # stop line
                          24.0 + 40 * M + rng.normal(0, 5 * M, 10),   # queue 40 m back
                          24.0 + 500 * M + rng.normal(0, 5 * M, 10)]) # another light
    stops = pd.DataFrame({"lat": lat, "lon": 120.0})
    out = find_intersections(stops)
    assert len(out) == 2
    assert sorted(out["n_stops"]) == [10, 20]


def test_passes_for_ride_assigns_wait_to_nearest_light():
    track = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=101, freq="s", tz="UTC"),
        "lat": 24.0 + np.arange(101) * 10 * M, "lon": 120.0,
    })
    lights = pd.DataFrame({"intersection": [0, 1, 2],
                           "lat": [24.0 + 200 * M, 24.0 + 700 * M, 24.0], "lon": [120.0, 120.0, 120.1]})
    stops = pd.DataFrame({"lat": [24.0 + 205 * M], "lon": [120.0], "duration_s": [42]})
    p = passes_for_ride(track, stops, lights)
    assert p["intersection"].tolist() == [0, 1]       # light 2 is 10 km away
    assert p["wait_s"].tolist() == [42.0, 0.0]
    assert p["order"].tolist() == [20, 70]


def _passes(intersection, waits, route=("university", "outbound")):
    return pd.DataFrame({"intersection": intersection, "wait_s": waits,
                         "destination": route[0], "direction": route[1]})


def test_intersection_stats_smoothed_probability():
    p = _passes(0, [30.0] * 15 + [0.0] * 15)
    s = intersection_stats(p, min_passes=10).iloc[0]
    assert s["passes"] == 30 and s["stops"] == 15
    assert s["p_stop"] == pytest.approx(16 / 32)
    assert s["median_wait_s"] == 30.0


def test_relevant_passes_drops_lights_that_never_stop_route():
    p = pd.concat([_passes(0, [0.0] * 20), _passes(1, [10.0] * 5 + [0.0] * 15)])
    assert set(relevant_passes(p)["intersection"]) == {1}


def test_shrunken_mean_pulls_toward_prior():
    assert shrunken_mean(pd.Series([100.0]), prior_mean=0.0, k=1) == 50.0
    assert shrunken_mean(pd.Series([], dtype=float), prior_mean=7.0) == 7.0


def test_forecast_wait_uses_same_weekday_and_hour():
    history = pd.DataFrame({
        "weekday": [0] * 50 + [1] * 50,
        "depart_hour": 8.0,
        "light_wait_s": [300.0] * 50 + [100.0] * 50,
        "light_stops": 5.0,
    })
    monday = forecast_wait(history, weekday=0, hour=8.0)
    tuesday = forecast_wait(history, weekday=1, hour=8.0)
    assert monday["wait_s"] > 250 and tuesday["wait_s"] < 150
    assert monday["n"] == 50


def test_green_wave_pairs_detects_linked_lights():
    rows = []
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    for ride in range(40):
        red_a = ride % 2 == 0
        rows += [
            {"activity_id": ride, "intersection": 0, "order": 0, "time": t0, "wait_s": 30.0 if red_a else 0.0},
            {"activity_id": ride, "intersection": 1, "order": 60, "time": t0 + pd.Timedelta(seconds=90),
             "wait_s": 0.0 if red_a else 20.0},  # B is red exactly when A was green
        ]
    p = pd.DataFrame(rows).assign(destination="university", direction="outbound")
    pair = green_wave_pairs(p, min_both=10).iloc[0]
    assert pair["p_stop_after_green"] == 1.0
    assert pair["p_stop_after_red"] == 0.0
