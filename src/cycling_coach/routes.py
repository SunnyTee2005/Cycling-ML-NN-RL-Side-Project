"""Route variants of a commute and the 'leave-by' calculator."""

import numpy as np
import pandas as pd

CELL_M = 80.0
SAME_ROUTE_SHARE = 0.8


def track_cells(lat: np.ndarray, lon: np.ndarray, cell_m: float = CELL_M) -> frozenset:
    """The set of ~cell_m grid squares a track passes through."""
    ok = ~(np.isnan(lat) | np.isnan(lon))
    lat, lon = lat[ok], lon[ok]
    if len(lat) == 0:
        return frozenset()
    cell_deg_lat = cell_m / 111_195
    cell_deg_lon = cell_deg_lat / np.cos(np.radians(np.median(lat)))
    rows = np.floor(lat / cell_deg_lat).astype(int)
    cols = np.floor(lon / cell_deg_lon).astype(int)
    return frozenset(zip(rows.tolist(), cols.tolist()))


def dilate(cells: frozenset) -> frozenset:
    """Cells plus their 8 neighbours."""
    return frozenset((r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def route_similarity(a: frozenset, b: frozenset, a_wide: frozenset, b_wide: frozenset) -> float:
    """Share of each track lying within one cell of the other (the smaller share).

    Comparing against the neighbour-dilated set makes the measure robust to two
    tracks a few metres apart falling on opposite sides of a cell boundary.
    """
    if not a or not b:
        return 0.0
    return min(len(a & b_wide) / len(a), len(b & a_wide) / len(b))


def cluster_routes(cells: list[frozenset], threshold: float = SAME_ROUTE_SHARE) -> np.ndarray:
    """Group tracks that follow the same streets.

    Greedy: the track similar to the most others becomes a route's
    representative; every unassigned track similar enough joins it.
    Route 0 is the most ridden one.
    """
    n = len(cells)
    wide = [dilate(c) for c in cells]
    sim = np.array([[route_similarity(cells[i], cells[j], wide[i], wide[j]) for j in range(n)]
                    for i in range(n)])
    labels = np.full(n, -1)
    route = 0
    while (labels < 0).any():
        free = labels < 0
        support = ((sim >= threshold) & free[None, :]).sum(axis=1) * free
        rep = support.argmax()
        labels[(sim[rep] >= threshold) & free] = route
        route += 1
    return labels


def leave_by(durations_s: pd.Series, arrive: str, confidence: float = 0.95) -> str:
    """Latest departure time (HH:MM) to arrive by `arrive` with the given confidence.

    Uses the `confidence` quantile of past door-to-door durations, i.e. on that
    share of past days you would have made it.
    """
    need = float(np.quantile(durations_s, confidence))
    h, m = map(int, arrive.split(":"))
    depart = h * 3600 + m * 60 - need
    depart_min = int(np.floor(depart / 60))
    return f"{depart_min // 60:02d}:{depart_min % 60:02d}"
