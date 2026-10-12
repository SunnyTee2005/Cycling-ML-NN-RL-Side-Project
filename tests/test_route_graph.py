import numpy as np
import pandas as pd
import pytest

from cycling_coach.route_graph import (
    Grid,
    build_graph,
    cell_sequence,
    dijkstra,
    nearest_node,
    path_time,
    transitions,
)

M = 1 / 111_195


def _ride(lats, lons):
    n = len(lats)
    return pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=n, freq="s", tz="UTC"),
                         "lat": lats, "lon": lons})


def test_cell_sequence_collapses_and_undoes_bounces():
    grid = Grid(24.0, cell_m=40)
    base = 24.0 + 20 * M  # middle of a cell row
    lats = [base, base, base + 40 * M, base, base + 40 * M, base + 80 * M]
    seq = cell_sequence(_ride(lats, [120.0] * 6), grid)
    rows = [c[0] for c, _ in seq]
    assert len(rows) == 3 and rows[1] - rows[0] == 1 and rows[2] - rows[1] == 1


def test_transitions_skip_gps_gaps():
    t = pd.Timestamp("2025-01-01", tz="UTC")
    seq = [((0, 0), t), ((0, 1), t + pd.Timedelta(seconds=5)), ((0, 9), t + pd.Timedelta(seconds=9))]
    assert transitions(seq) == [((0, 0), (0, 1), 5.0)]


def _graph():
    rows = [
        # usual route: straight but with a slow light at (0, 1)
        ((0, 0), (0, 1), 10), ((0, 1), (0, 2), 60), ((0, 2), (0, 3), 10),
        # detour: one extra cell, no light
        ((0, 0), (1, 1), 12), ((1, 1), (1, 2), 10), ((1, 2), (0, 3), 12),
    ]
    edges = pd.DataFrame([r for r in rows for _ in range(3)], columns=["src", "dst", "seconds"])
    return build_graph(edges, shrink_k=0)


def test_dijkstra_prefers_detour_around_slow_light():
    cost, path = dijkstra(_graph(), (0, 0), (0, 3))
    assert cost == 34
    assert path == [(0, 0), (1, 1), (1, 2), (0, 3)]


def test_dijkstra_unreachable():
    cost, path = dijkstra(_graph(), (0, 3), (0, 0))  # edges are one-way
    assert cost == np.inf and path == []


def test_build_graph_drops_rare_edges():
    edges = pd.DataFrame({"src": [(0, 0)], "dst": [(0, 1)], "seconds": [5.0]})
    assert build_graph(edges, min_traversals=2) == {}


def test_build_graph_uses_mean_so_occasional_red_lights_count():
    edges = pd.DataFrame({"src": [(0, 0)] * 5, "dst": [(0, 1)] * 5,
                          "seconds": [5.0, 5.0, 5.0, 65.0, 65.0]})  # red 2 times out of 5
    graph = build_graph(edges, min_traversals=1, shrink_k=0)
    assert graph[(0, 0)] == [((0, 1), 29.0)]  # median would say 5 s


def test_build_graph_shrinks_rarely_ridden_edges_toward_prior():
    common = [((0, 0), (0, 1), 10.0)] * 30
    lucky = [((5, 5), (5, 6), 1.0)] * 3
    edges = pd.DataFrame(common + lucky, columns=["src", "dst", "seconds"])
    graph = build_graph(edges, min_traversals=3, shrink_k=3)
    lucky_weight = graph[(5, 5)][0][1]
    assert 1.0 < lucky_weight < 10.0


def test_path_time_with_missing_edge():
    total, missing = path_time(_graph(), [(0, 0), (0, 1), (5, 5)], fallback_s_per_cell=6)
    assert missing == 1 and total == pytest.approx(10 + 6 * 5)


def test_nearest_node():
    assert nearest_node(_graph(), (1, 3)) in {(1, 2), (0, 2)}


def test_trim_path_to_common_endpoints():
    from cycling_coach.route_graph import trim_path

    cells = [(0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (1, 3)]
    assert trim_path(cells, (0, 0), (0, 2)) == [(0, 0), (0, 1), (0, 2)]
