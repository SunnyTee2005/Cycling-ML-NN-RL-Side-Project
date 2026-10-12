"""Find faster commute routes on a graph built from my own rides.

Usage:
    python scripts/find_routes.py

Writes data/processed/route_edges.parquet, reports/route_finder.md and
route maps in reports/private/ (gitignored).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cycling_coach import config
from cycling_coach.plotting import GRID, SERIES, save, title
from cycling_coach.route_graph import (
    Grid,
    build_graph,
    cell_sequence,
    dijkstra,
    nearest_node,
    path_time,
    transitions,
    trim_path,
)
from cycling_coach.routes import dilate

PRIVATE = config.REPORTS_DIR / "private"
ROAD_BIKES = {"TCR BLACK", "Reacto 400"}
AREA_MARGIN_DEG = 0.03          # ~3 km around home and schools
FALLBACK_S_PER_CELL = 40 / 6.0  # 6 m/s where the graph has no edge
ROUTES = {
    ("university", "outbound"): "Home → university",
    ("university", "home"): "University → home",
    ("high_school", "outbound"): "Home → high school",
    ("high_school", "home"): "High school → home",
}


def fmt_min(seconds: float) -> str:
    seconds = int(round(seconds))
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    return f"{sign}{seconds // 60}:{seconds % 60:02d}"


def md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    return head + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))


def collect_edges(rides: pd.DataFrame, grid: Grid, box) -> tuple[pd.DataFrame, dict]:
    """Traversal records for every ride crossing the area, plus each ride's cell path."""
    lat0, lat1, lon0, lon1 = box
    frames, paths = [], {}
    for r in rides.itertuples():
        ride = pd.read_parquet(config.PROCESSED_DIR / "rides" / f"{r.activity_id}.parquet",
                               columns=["timestamp", "lat", "lon"])
        inside = ride["lat"].between(lat0, lat1) & ride["lon"].between(lon0, lon1)
        if inside.sum() < 60:
            continue
        seq = cell_sequence(ride[inside], grid)
        paths[r.activity_id] = [c for c, _ in seq]
        t = transitions(seq)
        if t:
            frames.append(pd.DataFrame(t, columns=["src", "dst", "seconds"]).assign(
                activity_id=r.activity_id, start_time=r.start_time))
    return pd.concat(frames, ignore_index=True), paths


def validate(route_commutes: pd.DataFrame, edges: pd.DataFrame, paths: dict) -> dict:
    """Predict the last 20% of commutes from a graph of strictly earlier rides."""
    rc = route_commutes[route_commutes["activity_id"].isin(paths)].sort_values("depart_local")
    cut = int(len(rc) * 0.8)
    train, test = rc.iloc[:cut], rc.iloc[cut:]
    graph = build_graph(edges[edges["start_time"] < test["start_time"].min()])
    pred = lambda aid: path_time(graph, paths[aid], FALLBACK_S_PER_CELL)[0]  # noqa: E731
    # Training rides include faster recreational rides; scale to commute pace.
    scale = np.median(train["elapsed_s"] / train["activity_id"].map(pred))
    predicted = test["activity_id"].map(pred) * scale
    actual = test["elapsed_s"]
    return {"n_test": len(test), "scale": scale,
            "mae": (predicted - actual).abs().mean(),
            "naive_mae": (train["elapsed_s"].median() - actual).abs().mean(),
            "median_actual": actual.median()}


def held_out_saving(edges: pd.DataFrame, start, goal, usual_paths: list, scale: float,
                    n_splits: int = 5, seed: int = 0) -> tuple[float, float]:
    """Honest saving: search on half of the rides, measure on the other half.

    Picking the minimum over many noisy paths is optimistic by construction. Splitting
    rides into a search half and an evaluation half removes that bias, like a test set
    for the optimiser. Returns (mean saving, share of splits with a positive saving).
    """
    rng = np.random.default_rng(seed)
    ids = edges["activity_id"].unique()
    savings = []
    for _ in range(n_splits):
        half = set(rng.choice(ids, size=len(ids) // 2, replace=False).tolist())
        # Half the data: allow edges seen twice (shrinkage still guards rare edges).
        search = build_graph(edges[edges["activity_id"].isin(half)], min_traversals=2)
        evaluate = build_graph(edges[~edges["activity_id"].isin(half)], min_traversals=2)
        try:
            s0 = nearest_node(search, start)
            g0 = nearest_node(search, goal)
        except ValueError:
            continue
        _, found = dijkstra(search, s0, g0)
        if not found:
            continue
        found_s = path_time(evaluate, found, FALLBACK_S_PER_CELL)[0]
        usual_s = np.median([path_time(evaluate, p, FALLBACK_S_PER_CELL)[0] for p in usual_paths])
        savings.append((usual_s - found_s) * scale)
    savings = np.array(savings)
    if len(savings) == 0:
        return np.nan, np.nan
    return float(savings.mean()), float((savings > 0).mean())


def path_length_km(cells: list, cell_m: float) -> float:
    steps = np.diff(np.array(cells), axis=0)
    return float(np.hypot(steps[:, 0], steps[:, 1]).sum() * cell_m / 1000)


def lights_on_path(cells: list, grid: Grid, lights: pd.DataFrame) -> int:
    lr, lc = grid.cells(lights["lat"], lights["lon"])
    wide = dilate(frozenset(cells))
    return int(sum((int(r), int(c)) in wide for r, c in zip(lr, lc)))


def weakest_link(graph_counts: dict, cells: list) -> int:
    return int(min(graph_counts.get((a, b), 0) for a, b in zip(cells, cells[1:])))


def plot_private(edges, paths_to_draw, grid, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    e = edges.drop_duplicates(["src", "dst"])
    for src, dst in zip(e["src"].head(60_000), e["dst"].head(60_000)):
        (a_lat, a_lon), (b_lat, b_lon) = grid.center(src), grid.center(dst)
        ax.plot([a_lon, b_lon], [a_lat, b_lat], color=GRID, linewidth=0.4)
    for (cells, color, name) in paths_to_draw:
        pts = np.array([grid.center(c) for c in cells])
        ax.plot(pts[:, 1], pts[:, 0], color=color, linewidth=2.5, label=name)
    ax.legend(frameon=False)
    ax.set_aspect(1 / np.cos(np.radians(np.mean([p[0] for p in [grid.center(paths_to_draw[0][0][0])]]))))
    title(ax, "Usual vs suggested route — PRIVATE")
    save(fig, path)


def main() -> None:
    places = pd.read_parquet(config.PROCESSED_DIR / "places.parquet").set_index("name")
    commutes = pd.read_parquet(config.PROCESSED_DIR / "commute_routes.parquet")
    commutes = commutes[commutes["bike"].isin(ROAD_BIKES)]
    rides = pd.read_parquet(config.PROCESSED_DIR / "rides.parquet")
    rides = rides[rides["bike"].isin(ROAD_BIKES) & rides["has_lat"] & rides["in_taiwan"]]
    lights = pd.read_parquet(config.PROCESSED_DIR / "intersections.parquet")

    key = places.loc[["home", "university", "high_school"]]
    box = (key["lat"].min() - AREA_MARGIN_DEG, key["lat"].max() + AREA_MARGIN_DEG,
           key["lon"].min() - AREA_MARGIN_DEG, key["lon"].max() + AREA_MARGIN_DEG)
    grid = Grid(key["lat"].mean())
    edges, paths = collect_edges(rides, grid, box)
    edges.assign(src=edges["src"].astype(str), dst=edges["dst"].astype(str)).to_parquet(
        config.PROCESSED_DIR / "route_edges.parquet", index=False)

    graph = build_graph(edges)
    counts = edges.groupby(["src", "dst"]).size().to_dict()
    n_edges = sum(len(v) for v in graph.values())

    val_rows, sugg_rows = [], []
    PRIVATE.mkdir(parents=True, exist_ok=True)
    for (dest, direction), name in ROUTES.items():
        rc = commutes[(commutes["destination"] == dest) & (commutes["direction"] == direction)]
        v = validate(rc, edges, paths)
        val_rows.append({"Route": name, "Test commutes": v["n_test"],
                         "Typical time": fmt_min(v["median_actual"]),
                         "Graph error (MAE)": fmt_min(v["mae"]),
                         "Naive error": fmt_min(v["naive_mae"])})

        origin, target = ("home", dest) if direction == "outbound" else (dest, "home")
        start = nearest_node(graph, grid.cells(places.loc[origin, "lat"], places.loc[origin, "lon"]))
        goal = nearest_node(graph, grid.cells(places.loc[target, "lat"], places.loc[target, "lon"]))
        best_s, best_path = dijkstra(graph, start, goal)

        # The usual route (variant A): model time = median over all its commutes, so
        # one noisy ride does not decide; also report what those rides actually took.
        usual = rc[(rc["variant"] == 0) & rc["activity_id"].isin(paths)]
        usual_paths = {aid: trim_path(paths[aid], start, goal) for aid in usual["activity_id"]}
        usual_times = usual["activity_id"].map(
            lambda aid: path_time(graph, usual_paths[aid], FALLBACK_S_PER_CELL)[0])
        usual_s = float(usual_times.median())
        rep = usual.iloc[(usual_times - usual_s).abs().argsort()[:1]]["activity_id"].iloc[0]
        usual_cells = usual_paths[rep]
        overlap = len(set(best_path) & dilate(frozenset(usual_cells))) / max(len(best_path), 1)

        scale = v["scale"]
        honest, share_pos = held_out_saving(edges, start, goal,
                                            list(usual_paths.values()), scale)
        sugg_rows.append({
            "Route": name,
            "Usual, actual median": fmt_min(usual["elapsed_s"].median()),
            "Usual (model)": fmt_min(usual_s * scale),
            "Fastest found (model)": fmt_min(best_s * scale),
            "Saving (search data)": fmt_min((usual_s - best_s) * scale),
            "Saving (held-out)": "no path in half graphs" if np.isnan(honest)
                                 else f"{fmt_min(honest)} ({share_pos:.0%} of splits > 0)",
            "Length usual / found": f"{path_length_km(usual_cells, grid.cell_m):.1f} / "
                                    f"{path_length_km(best_path, grid.cell_m):.1f} km",
            "Shared with usual": f"{overlap:.0%}",
            "Lights usual / found": f"{lights_on_path(usual_cells, grid, lights)} / "
                                    f"{lights_on_path(best_path, grid, lights)}",
            "Weakest edge (rides)": weakest_link(counts, best_path),
        })
        plot_private(edges, [(usual_cells, SERIES[0], "usual"), (best_path, SERIES[1], "suggested")],
                     grid, PRIVATE / f"route_{dest}_{direction}.png")

    report = f"""# Faster-route finder

Generated by `scripts/find_routes.py`.

## Method

1. Cut the area around home and both schools into **{grid.cell_m:.0f} m cells**.
2. Replay every road-bike ride that crosses the area ({edges['activity_id'].nunique()} rides, not only
   commutes): each move from one cell to a neighbouring cell is a traversal of a **directed edge**,
   timed from entering the first cell to entering the next — so a red-light wait inside a cell
   is part of that edge's time. GPS jitter at standstill (A→B→A) is undone.
3. Edge weight = **expected (mean) traversal time**, not the median: a light that is red 40% of
   the time has a median wait of zero but still costs time on average. Stays over 3 min are
   clipped (breaks), edges ridden fewer than 3 times are dropped, and the rest are **shrunk
   toward the average time per cell** — otherwise the search picks edges that were just lucky
   the few times they were ridden (the "winner's curse").
   The graph has **{len(graph):,} nodes and {n_edges:,} edges**.
4. **Dijkstra** finds the path with the smallest total time between home and school.

Because every edge is a piece of road I actually rode in that direction, suggestions are
always rideable and respect one-way streets. No external map is needed.

## Can the graph predict commute times?

Chronological test: build the graph only from rides *before* the last 20% of commutes, then
predict each test commute by summing the edge times along the path it actually took.
Recreational rides are faster than commutes, so predictions are scaled by the median
actual/predicted ratio of the earlier commutes. *Naive*: always predict the median earlier commute.

{md(pd.DataFrame(val_rows))}

## Suggestions

*Usual (model)* is the median model time over all variant-A commutes, trimmed to the same start
and end cells as the search (so door and campus manoeuvring is not counted); it should be a
little below the actual door-to-door median. *Saving (search data)* compares model with model on the
graph the search used — optimistic, because the minimum of many noisy paths is biased low.
**Saving (held-out)** is the honest number: rides are split in half five times; the route is
searched on one half and timed on the other. *Shared with usual*: share of the suggested path within ~40 m of the usual one.
*Weakest edge*: the least-ridden piece of the suggestion (low = less certain). Lengths are
measured along grid cells, so both read longer than the true road distance.
Route maps (usual vs suggested) are in the private folder, because they show where I live.

{md(pd.DataFrame(sugg_rows))}

## Caveats and next step

- Edge times mix all times of day and all ride types; rush-hour traffic is averaged in.
- Edges ridden only a few times are uncertain even after shrinkage; check the weakest edge.
- The graph only knows roads I have ridden. Roads never ridden cannot be suggested — predicting
  them is the machine-learning step (phase 2).
- **Next:** ride the suggested route for two weeks and compare with the prediction — the
  only real test of a route recommender.
"""
    (config.REPORTS_DIR / "route_finder.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
