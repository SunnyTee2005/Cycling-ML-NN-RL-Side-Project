"""Traffic-light analytics from commute stops.

Idea: a traffic light is a place where commutes stop again and again. We
cluster stop locations into intersections, then for every commute check
which intersections it passed and whether it stopped there. That gives a
per-intersection stop probability and waiting-time distribution.
"""

import numpy as np
import pandas as pd

from cycling_coach.commute import cluster_points, merge_close_places
from cycling_coach.geo import haversine_m

ENDPOINT_RADIUS_M = 200.0   # ignore stops near home/school (and keep them private)
MAX_LIGHT_WAIT_S = 180      # longer stops are shops, chats, ... not lights
INTERSECTION_RADIUS_M = 30.0
MIN_INTERSECTION_STOPS = 8
MERGE_INTERSECTIONS_M = 60.0  # queue positions / both sides of one junction
PASS_RADIUS_M = 35.0


def light_stops(stops: pd.DataFrame, commutes: pd.DataFrame, places: pd.DataFrame) -> pd.DataFrame:
    """Commute stops that can be traffic lights: not near an endpoint, not too long."""
    s = stops.merge(commutes[["activity_id", "destination", "direction"]], on="activity_id")
    near_endpoint = np.zeros(len(s), dtype=bool)
    for p in places.itertuples():
        near_endpoint |= haversine_m(s["lat"], s["lon"], p.lat, p.lon) < ENDPOINT_RADIUS_M
    keep = ~near_endpoint & (s["duration_s"] <= MAX_LIGHT_WAIT_S) & s["lat"].notna()
    return s[keep].reset_index(drop=True)


def find_intersections(stops: pd.DataFrame) -> pd.DataFrame:
    """Cluster stop locations; each dense cluster is an intersection.

    Clusters closer than MERGE_INTERSECTIONS_M (queue positions, both sides of
    one junction) are merged into one intersection.
    """
    lat, lon = stops["lat"].to_numpy(), stops["lon"].to_numpy()
    labels, centroids = cluster_points(
        lat, lon, radius_m=INTERSECTION_RADIUS_M, min_size=MIN_INTERSECTION_STOPS,
    )
    mapping = merge_close_places(centroids, radius_m=MERGE_INTERSECTIONS_M)
    labels = np.where(labels >= 0, mapping[np.maximum(labels, 0)], -1)
    rows = []
    for new_id, old_id in enumerate(np.unique(labels[labels >= 0])):
        members = labels == old_id
        rows.append({"intersection": new_id, "lat": lat[members].mean(),
                     "lon": lon[members].mean(), "n_stops": int(members.sum())})
    return pd.DataFrame(rows, columns=["intersection", "lat", "lon", "n_stops"])


def passes_for_ride(track: pd.DataFrame, ride_stops: pd.DataFrame,
                    intersections: pd.DataFrame) -> pd.DataFrame:
    """Which intersections one ride passed, when, and how long it waited there.

    `track` has timestamp/lat/lon at 1 Hz, `ride_stops` the ride's stop events.
    """
    pts = track.dropna(subset=["lat", "lon"])
    if pts.empty or intersections.empty:
        return pd.DataFrame(columns=["intersection", "time", "order", "wait_s"])
    d = haversine_m(intersections["lat"].to_numpy()[:, None], intersections["lon"].to_numpy()[:, None],
                    pts["lat"].to_numpy()[None, :], pts["lon"].to_numpy()[None, :])
    closest = d.argmin(axis=1)
    passed = d.min(axis=1) < PASS_RADIUS_M

    # Each stop belongs to its nearest intersection (if close enough).
    wait = np.zeros(len(intersections))
    if not ride_stops.empty:
        ds = haversine_m(intersections["lat"].to_numpy()[:, None],
                         intersections["lon"].to_numpy()[:, None],
                         ride_stops["lat"].to_numpy()[None, :], ride_stops["lon"].to_numpy()[None, :])
        nearest = ds.argmin(axis=0)
        close = ds.min(axis=0) < PASS_RADIUS_M
        np.add.at(wait, nearest[close], ride_stops["duration_s"].to_numpy()[close])

    rows = [{"intersection": int(intersections["intersection"].iloc[k]),
             "time": pts["timestamp"].iloc[closest[k]], "order": int(closest[k]),
             "wait_s": float(wait[k])}
            for k in np.flatnonzero(passed)]
    return pd.DataFrame(rows, columns=["intersection", "time", "order", "wait_s"])


def relevant_passes(passes: pd.DataFrame, min_stops: int = 3) -> pd.DataFrame:
    """Keep passes of lights that actually stop this route direction.

    A ride also "passes" stop clusters of the opposite carriageway; those never
    stop it and would inflate the number of lights on the route.
    """
    keys = ["intersection", "destination", "direction"]
    stops = passes.assign(s=passes["wait_s"] > 0).groupby(keys)["s"].transform("sum")
    return passes[stops >= min_stops]


def intersection_stats(passes: pd.DataFrame, min_passes: int = 20) -> pd.DataFrame:
    """Per intersection and route direction: stop probability and waiting time.

    The stop probability uses add-one smoothing ((stops+1)/(passes+2)) so an
    intersection seen only a few times does not get 0% or 100%.
    """
    p = passes.assign(stopped=passes["wait_s"] > 0)
    g = p.groupby(["intersection", "destination", "direction"])
    stats = g.agg(passes=("stopped", "size"), stops=("stopped", "sum"))
    waits = p[p["stopped"]].groupby(["intersection", "destination", "direction"])["wait_s"]
    stats["p_stop"] = (stats["stops"] + 1) / (stats["passes"] + 2)
    stats["median_wait_s"] = waits.median()
    stats["mean_wait_s"] = waits.mean()
    stats["p90_wait_s"] = waits.quantile(0.9)
    # The longest typical wait approximates the red phase (arriving just as it turns red).
    stats["red_phase_estimate_s"] = waits.quantile(0.95)
    stats["expected_wait_s"] = stats["p_stop"] * stats["mean_wait_s"].fillna(0)
    return stats[stats["passes"] >= min_passes].reset_index()


def shrunken_mean(values: pd.Series, prior_mean: float, k: float = 5.0) -> float:
    """Average pulled toward `prior_mean` when there are few observations."""
    return (values.sum() + k * prior_mean) / (len(values) + k)


def forecast_wait(history: pd.DataFrame, weekday: int, hour: float,
                  window_h: float = 1.0, k: float = 5.0) -> dict:
    """Expected waiting time and number of stops for a departure.

    `history` holds one row per past commute on the same route with columns
    weekday, depart_hour, light_wait_s, light_stops. We average commutes on
    the same weekday within +-window_h hours, shrunk toward all commutes in
    that time window (any weekday) so sparse slots stay sensible.
    """
    in_window = (history["depart_hour"] - hour).abs() <= window_h
    slot = history[in_window & (history["weekday"] == weekday)]
    base = history[in_window] if in_window.any() else history
    return {
        "wait_s": shrunken_mean(slot["light_wait_s"], base["light_wait_s"].mean(), k),
        "stops": shrunken_mean(slot["light_stops"], base["light_stops"].mean(), k),
        "p10_s": base["light_wait_s"].quantile(0.1),
        "p90_s": base["light_wait_s"].quantile(0.9),
        "n": len(slot),
    }


def green_wave_pairs(passes: pd.DataFrame, min_both: int = 30) -> pd.DataFrame:
    """For consecutive intersections A -> B on the same route, compare
    P(stop at B | stopped at A) with P(stop at B | not stopped at A).

    A big difference means the two lights are coordinated (a "green wave").
    """
    p = passes.assign(stopped=passes["wait_s"] > 0).sort_values(["activity_id", "order"])
    p["next_intersection"] = p.groupby("activity_id")["intersection"].shift(-1)
    p["next_stopped"] = p.groupby("activity_id")["stopped"].shift(-1)
    p["next_time"] = p.groupby("activity_id")["time"].shift(-1)
    p = p.dropna(subset=["next_intersection"])
    p["travel_s"] = (p["next_time"] - p["time"]).dt.total_seconds() - p["wait_s"]
    rows = []
    for (a, b, dest, direction), g in p.groupby(
            ["intersection", "next_intersection", "destination", "direction"]):
        if len(g) < min_both:
            continue
        after_green = g[~g["stopped"]]
        after_red = g[g["stopped"]]
        rows.append({
            "from": a, "to": int(b), "destination": dest, "direction": direction, "n": len(g),
            "p_stop_after_green": after_green["next_stopped"].astype(float).mean(),
            "p_stop_after_red": after_red["next_stopped"].astype(float).mean(),
            "travel_s_when_next_green": after_green.loc[~after_green["next_stopped"].astype(bool), "travel_s"].median(),
            "travel_s_when_next_red": after_green.loc[after_green["next_stopped"].astype(bool), "travel_s"].median(),
        })
    return pd.DataFrame(rows)
