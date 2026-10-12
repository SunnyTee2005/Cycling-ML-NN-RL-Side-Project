import numpy as np
import pandas as pd

from cycling_coach.routes import cluster_routes, dilate, leave_by, route_similarity, track_cells

M = 1 / 111_195


def _straight(lon_offset_m, n=100):
    lat = 24.0 + np.arange(n) * 20 * M
    lon = np.full(n, 120.0 + lon_offset_m * M)
    return track_cells(lat, lon)


def test_track_cells_ignores_nan():
    cells = track_cells(np.array([24.0, np.nan]), np.array([120.0, 120.0]))
    assert len(cells) == 1


def test_route_similarity_tolerates_cell_boundary():
    a, b = _straight(0), _straight(5)
    assert route_similarity(a, b, dilate(a), dilate(b)) == 1.0
    c = _straight(600)
    assert route_similarity(a, c, dilate(a), dilate(c)) == 0.0


def test_cluster_routes_separates_parallel_streets():
    same_street = [_straight(0), _straight(5), _straight(10)]
    other_street = [_straight(600), _straight(605)]
    labels = cluster_routes(same_street + other_street)
    assert labels.tolist() == [0, 0, 0, 1, 1]


def test_leave_by_uses_quantile():
    durations = pd.Series([1500.0] * 19 + [2100.0])  # 25 min usually, one 35-min day
    assert leave_by(durations, "09:00", 0.5) == "08:35"
    assert leave_by(durations, "09:00", 1.0) == "08:25"
