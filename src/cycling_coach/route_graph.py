"""A road graph built only from my own GPS traces, and fastest-path search on it.

The map is cut into square cells (~40 m). Every time a ride moves from cell A
to a neighbouring cell B, that is one traversal of the directed edge A -> B,
and the time it took (including any wait at a light inside A) is recorded.
The edge weight is the expected (mean) time, shrunk toward a prior for edges
ridden only a few times. Because edges only exist where
I actually rode, every suggested route is one I know is rideable, and one-way
streets are respected automatically.
"""

import heapq
from collections import defaultdict

import numpy as np
import pandas as pd

CELL_M = 40.0
MAX_STEP_CELLS = 2        # larger jumps between consecutive cells are GPS gaps
MIN_TRAVERSALS = 3
CLIP_S = 180.0            # longer stays in a cell are breaks, not traffic
SHRINK_K = 3.0            # prior weight, in traversals


class Grid:
    """Maps lat/lon to integer (row, col) cells of about CELL_M metres."""

    def __init__(self, ref_lat: float, cell_m: float = CELL_M):
        self.cell_m = cell_m
        self.dlat = cell_m / 111_195
        self.dlon = self.dlat / np.cos(np.radians(ref_lat))

    def cells(self, lat, lon):
        return (np.floor(np.asarray(lat) / self.dlat).astype(int),
                np.floor(np.asarray(lon) / self.dlon).astype(int))

    def center(self, cell):
        return (cell[0] + 0.5) * self.dlat, (cell[1] + 0.5) * self.dlon


def cell_sequence(ride: pd.DataFrame, grid: Grid) -> list[tuple[tuple[int, int], pd.Timestamp]]:
    """Cells a ride passes through, with the time it entered each.

    GPS jitter at a standstill makes the position bounce between two cells
    (A, B, A, B, ...); a return to the cell before the last one is undone so the
    wait stays in one cell.
    """
    pts = ride.dropna(subset=["lat", "lon"])
    rows, cols = grid.cells(pts["lat"], pts["lon"])
    seq: list = []
    for r, c, t in zip(rows, cols, pts["timestamp"]):
        cell = (int(r), int(c))
        if seq and seq[-1][0] == cell:
            continue
        if len(seq) >= 2 and seq[-2][0] == cell:  # bounce back: drop the detour
            seq.pop()
            continue
        seq.append((cell, t))
    return seq


def transitions(seq) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    """(from_cell, to_cell, seconds) for consecutive cells of one ride."""
    out = []
    for (a, ta), (b, tb) in zip(seq, seq[1:]):
        step = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        if step <= MAX_STEP_CELLS:
            out.append((a, b, (tb - ta).total_seconds()))
    return out


def build_graph(edges: pd.DataFrame, min_traversals: int = MIN_TRAVERSALS,
                clip_s: float = CLIP_S, shrink_k: float = SHRINK_K) -> dict:
    """Adjacency dict {from: [(to, expected_seconds), ...]} from traversal records.

    `edges` has columns src, dst (cells) and seconds, one row per traversal.

    - The **mean**, not the median: a light that is red 40% of the time has a
      median wait of zero, but it still costs time on average.
    - **Shrinkage**: (sum + k * prior) / (n + k), with the prior being the average
      time per cell step over all edges. Without it the shortest-path search
      picks edges that were simply lucky the two or three times they were ridden.
    """
    e = edges.assign(seconds=edges["seconds"].clip(upper=clip_s))
    step = [max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(e["src"], e["dst"])]
    e = e.assign(step=step)
    prior_per_step = e["seconds"].sum() / e["step"].sum()
    g = e.groupby(["src", "dst"]).agg(total=("seconds", "sum"), n=("seconds", "size"),
                                      step=("step", "first")).reset_index()
    g = g[g["n"] >= min_traversals]
    g["weight"] = (g["total"] + shrink_k * prior_per_step * g["step"]) / (g["n"] + shrink_k)
    graph = defaultdict(list)
    for src, dst, w in zip(g["src"], g["dst"], g["weight"]):
        graph[src].append((dst, float(w)))
    return dict(graph)


def dijkstra(graph: dict, start, goal) -> tuple[float, list]:
    """Fastest path by summed edge time. Returns (seconds, [cells]); inf if unreachable."""
    best = {start: 0.0}
    previous = {}
    queue = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if node == goal:
            path = [node]
            while path[-1] in previous:
                path.append(previous[path[-1]])
            return cost, path[::-1]
        if cost > best.get(node, np.inf):
            continue  # stale queue entry
        for nxt, w in graph.get(node, ()):
            new = cost + w
            if new < best.get(nxt, np.inf):
                best[nxt] = new
                previous[nxt] = node
                heapq.heappush(queue, (new, nxt))
    return np.inf, []


def path_time(graph: dict, cells: list, fallback_s_per_cell: float) -> tuple[float, int]:
    """Time of a given cell path using graph edges; unknown edges use a fallback.

    Returns (seconds, number of edges that were not in the graph).
    """
    weights = {src: dict(nbrs) for src, nbrs in graph.items()}
    total, missing = 0.0, 0
    for a, b in zip(cells, cells[1:]):
        w = weights.get(a, {}).get(b)
        if w is None:
            missing += 1
            w = fallback_s_per_cell * max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        total += w
    return total, missing


def nearest_node(graph: dict, cell) -> tuple[int, int]:
    """Graph node closest to `cell` (in cell units)."""
    nodes = np.array(list(graph.keys()))
    d = np.abs(nodes - np.array(cell)).max(axis=1)
    return tuple(int(v) for v in nodes[d.argmin()])


def trim_path(cells: list, start, goal) -> list:
    """Cut a ridden path to the stretch between the cells nearest `start` and `goal`,
    so it is compared with a searched path over the same start and end points."""
    arr = np.array(cells)
    i = int(np.abs(arr - np.array(start)).max(axis=1).argmin())
    j = i + int(np.abs(arr[i:] - np.array(goal)).max(axis=1).argmin())
    return cells[i : j + 1]
