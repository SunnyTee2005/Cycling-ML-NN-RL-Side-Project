"""Find frequent places (home, schools) from ride endpoints and label commutes.

No coordinates are configured anywhere: home is simply the place where the
most rides start or end, and schools are the places most often ridden to
from home.
"""

import numpy as np
import pandas as pd

from cycling_coach.geo import haversine_m

PLACE_RADIUS_M = 250.0      # endpoints within this radius belong to one place
MIN_PLACE_VISITS = 10
MERGE_PLACES_M = 600.0      # e.g. two gates of the same campus
MIN_COMMUTE_TRIPS = 20
MAX_COMMUTE_KM = 15.0
MAX_DETOUR = 2.5            # ridden distance / straight-line distance
MAX_COMMUTE_S = 2 * 3600
LOCAL_TZ = "Asia/Taipei"
SCHOOL_SWITCH = pd.Timestamp("2023-08-01", tz=LOCAL_TZ)  # high school -> university


def cluster_points(lat: np.ndarray, lon: np.ndarray, radius_m: float = PLACE_RADIUS_M,
                   min_size: int = MIN_PLACE_VISITS) -> tuple[np.ndarray, np.ndarray]:
    """Greedy density clustering: repeatedly take the point with the most
    unassigned neighbours within `radius_m` and make its neighbourhood a cluster.

    Returns (labels, centroids); label -1 means "not in any cluster".
    """
    dist = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    near = dist < radius_m
    labels = np.full(len(lat), -1)
    centroids = []
    while True:
        free = labels < 0
        counts = (near & free[None, :]).sum(axis=1) * free
        best = counts.argmax()
        if counts[best] < min_size:
            break
        members = near[best] & free
        labels[members] = len(centroids)
        centroids.append((lat[members].mean(), lon[members].mean()))
    return labels, np.array(centroids).reshape(-1, 2)


def merge_close_places(centroids: np.ndarray, radius_m: float = MERGE_PLACES_M) -> np.ndarray:
    """Map each place id to the id of the first (largest) place within `radius_m`."""
    mapping = np.arange(len(centroids))
    for i in range(len(centroids)):
        for j in range(i):
            if mapping[j] == j and haversine_m(*centroids[i], *centroids[j]) < radius_m:
                mapping[i] = j
                break
    return mapping


def find_places(rides: pd.DataFrame) -> tuple[pd.Series, pd.Series, np.ndarray]:
    """Assign each ride's start and end to a place id (-1 = elsewhere).

    Place 0 is home (most visits). Returns (origin, destination, centroids).
    """
    n = len(rides)
    lat = np.concatenate([rides["start_lat"], rides["end_lat"]])
    lon = np.concatenate([rides["start_lon"], rides["end_lon"]])
    labels, centroids = cluster_points(lat, lon)
    mapping = merge_close_places(centroids)
    merged = np.where(labels >= 0, mapping[np.maximum(labels, 0)], -1)
    origin = pd.Series(merged[:n], index=rides.index)
    destination = pd.Series(merged[n:], index=rides.index)
    return origin, destination, centroids


def name_destinations(trips: pd.DataFrame) -> dict[int, str]:
    """Name the two main destinations by era: before the 2023 school switch it is
    the high school, after it the university. Others become 'other_<id>'."""
    names = {}
    local = trips["start_time"].dt.tz_convert(LOCAL_TZ)
    before = trips[local < SCHOOL_SWITCH]["place"].value_counts()
    after = trips[local >= SCHOOL_SWITCH]["place"].value_counts()
    if len(before):
        names[before.index[0]] = "high_school"
    if len(after) and after.index[0] not in names:
        names[after.index[0]] = "university"
    for place in trips["place"].unique():
        names.setdefault(place, f"other_{place}")
    return names


def label_commutes(rides: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (one row per commute ride with destination and direction, place centroids).

    Expects columns: activity_id, start_time, start/end lat/lon, distance_m, elapsed_s.
    """
    rides = rides[rides["start_lat"].notna() & rides["end_lat"].notna()]
    origin, destination, centroids = find_places(rides)
    home = 0

    straight = haversine_m(rides["start_lat"], rides["start_lon"], rides["end_lat"], rides["end_lon"])
    direct = (rides["distance_m"] <= MAX_DETOUR * np.maximum(straight, 1.0)) & (
        rides["distance_m"] <= MAX_COMMUTE_KM * 1000) & (rides["elapsed_s"] <= MAX_COMMUTE_S)
    to_place = (origin == home) & (destination > 0)
    from_place = (destination == home) & (origin > 0)
    trips = rides[direct & (to_place | from_place)].copy()
    trips["place"] = np.where(to_place[trips.index], destination[trips.index], origin[trips.index])
    trips["direction"] = np.where(to_place[trips.index], "outbound", "home")

    frequent = trips["place"].value_counts()
    trips = trips[trips["place"].isin(frequent[frequent >= MIN_COMMUTE_TRIPS].index)]
    trips["destination"] = trips["place"].map(name_destinations(trips))

    local = trips["start_time"].dt.tz_convert(LOCAL_TZ)
    trips["depart_local"] = local
    trips["depart_hour"] = local.dt.hour + local.dt.minute / 60
    trips["weekday"] = local.dt.dayofweek  # Monday = 0
    return trips.reset_index(drop=True), centroids
