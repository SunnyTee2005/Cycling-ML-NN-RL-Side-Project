import numpy as np
import pandas as pd

from cycling_coach.commute import cluster_points, label_commutes, merge_close_places

M = 1 / 111_195  # degrees latitude per metre
HOME = (24.0, 120.0)
SCHOOL = (24.0 + 4000 * M, 120.0)      # 4 km north
UNI = (24.0 - 8000 * M, 120.0)         # 8 km south
rng = np.random.default_rng(0)


def _jitter(point, n):
    return point[0] + rng.normal(0, 30 * M, n), point[1] + rng.normal(0, 30 * M, n)


def test_cluster_points_finds_dense_places():
    lat_a, lon_a = _jitter(HOME, 30)
    lat_b, lon_b = _jitter(SCHOOL, 15)
    lat = np.concatenate([lat_a, lat_b, [25.0]])
    lon = np.concatenate([lon_a, lon_b, [121.0]])
    labels, centroids = cluster_points(lat, lon)
    assert (labels[:30] == 0).all()
    assert (labels[30:45] == 1).all()
    assert labels[-1] == -1
    assert len(centroids) == 2


def test_merge_close_places():
    centroids = np.array([HOME, (HOME[0] + 400 * M, HOME[1]), SCHOOL])
    assert merge_close_places(centroids).tolist() == [0, 0, 2]


def _trips(dest, n, start, outbound=True, km=None):
    o_lat, o_lon = _jitter(HOME, n)
    d_lat, d_lon = _jitter(dest, n)
    if not outbound:
        o_lat, o_lon, d_lat, d_lon = d_lat, d_lon, o_lat, o_lon
    straight = abs(dest[0] - HOME[0]) / M
    return pd.DataFrame({
        "start_time": pd.date_range(start, periods=n, freq="D", tz="UTC"),
        "start_lat": o_lat, "start_lon": o_lon, "end_lat": d_lat, "end_lon": d_lon,
        "distance_m": (km or straight * 1.2 / 1000) * 1000, "elapsed_s": 1200,
    })


def test_label_commutes_names_schools_by_era():
    rides = pd.concat([
        _trips(SCHOOL, 30, "2021-03-01"),
        _trips(SCHOOL, 25, "2021-03-01", outbound=False),
        _trips(UNI, 25, "2025-03-01"),
        _trips(SCHOOL, 25, "2022-03-01", km=60),  # long loop ending at school: not a commute
    ], ignore_index=True)
    rides["activity_id"] = range(len(rides))
    commutes, _ = label_commutes(rides)
    counts = commutes.groupby(["destination", "direction"]).size().to_dict()
    assert counts == {("high_school", "home"): 25, ("high_school", "outbound"): 30,
                      ("university", "outbound"): 25}
    assert str(commutes["depart_local"].dt.tz) == "Asia/Taipei"
