"""Clean a parsed 1 Hz ride: GPS spikes, stops, smoothed altitude and grade.

Every second is kept so the time axis stays regular; cleaning adds flags:
- is_stopped: standing still (auto-pause, crawling below SLOW_MPS, or a
  recording gap where the position did not change)
- is_gap: recording gap where the rider DID move (GPS/recording loss)
- is_moving: neither of the above
"""

import numpy as np
import pandas as pd

from cycling_coach.geo import haversine_m
from cycling_coach.resample import interpolate_short_gaps

MAX_SPEED_MPS = 25.0        # 90 km/h; faster than any realistic ride segment
SPIKE_DEVIATION_M = 50.0    # distance from the local median track
SLOW_MPS = 1.0              # below this the rider is effectively stopped
MAX_STOP_DRIFT_M = 30.0     # position change allowed across a stop
MERGE_STOPS_WITHIN_S = 3    # stops separated by a short roll are one stop
MIN_STOP_S = 3


# --- GPS -----------------------------------------------------------------------

def remove_gps_spikes(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Drop positions far from the rolling-median track or implying impossible speed."""
    df = df.copy()
    valid = df[["lat", "lon"]].dropna()
    if len(valid) < window:
        return df
    med = valid.rolling(window, center=True, min_periods=3).median()
    deviation = haversine_m(valid["lat"], valid["lon"], med["lat"], med["lon"])
    seconds = valid.index.to_series().diff()  # rows are 1 s apart
    step = haversine_m(valid["lat"].shift(), valid["lon"].shift(), valid["lat"], valid["lon"])
    too_fast = (step / seconds) > MAX_SPEED_MPS
    bad = valid.index[(deviation > SPIKE_DEVIATION_M) | too_fast]
    df.loc[bad, ["lat", "lon"]] = np.nan
    for column in ("lat", "lon"):
        df[column] = interpolate_short_gaps(df[column], max_gap=10)
    return df


def recompute_distance_from_gps(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild distance and speed from cleaned positions (for GPX-derived rides)."""
    step = haversine_m(df["lat"].shift(), df["lon"].shift(), df["lat"], df["lon"])
    step = pd.Series(step, index=df.index)
    distance = step.fillna(0).cumsum().where(df["lat"].notna())
    return df.assign(distance_m=distance, speed_mps=step)


def clip_speed(df: pd.DataFrame) -> pd.DataFrame:
    speed = df["speed_mps"].where(df["speed_mps"] <= MAX_SPEED_MPS)
    return df.assign(speed_mps=speed)


# --- Stops ---------------------------------------------------------------------

def _runs(mask: pd.Series) -> list[tuple[int, int]]:
    """(first, last) positional indices of consecutive True runs."""
    values = mask.to_numpy()
    edges = np.diff(np.concatenate([[0], values.astype(int), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts, ends))


def _merge_close_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    merged = []
    for start, end in runs:
        if merged and start - merged[-1][1] - 1 <= max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def detect_stops(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag stopped / gap seconds and return a table of stop events.

    A candidate second is paused, slower than SLOW_MPS, or missing (filled).
    A candidate run counts as a stop when the position before and after it is
    within MAX_STOP_DRIFT_M; otherwise it is a recording gap while moving.
    """
    slow = df["speed_mps"] < SLOW_MPS
    candidate = df["is_paused"] | df["is_filled"] | slow
    runs = _merge_close_runs(_runs(candidate), MERGE_STOPS_WITHIN_S)

    # Last known position at/before each row, first known position at/after it.
    before = df[["lat", "lon"]].ffill().to_numpy()
    after = df[["lat", "lon"]].bfill().to_numpy()
    last = len(df) - 1

    is_stopped = np.zeros(len(df), dtype=bool)
    is_gap = np.zeros(len(df), dtype=bool)
    events = []
    for start, end in runs:
        lat0, lon0 = before[max(start - 1, 0)]
        lat1, lon1 = after[min(end + 1, last)]
        drift = haversine_m(lat0, lon0, lat1, lon1)
        part = df.iloc[start : end + 1]
        if np.isfinite(drift) and drift > MAX_STOP_DRIFT_M and not part["is_paused"].any() \
                and not slow.iloc[start : end + 1].all():
            is_gap[start : end + 1] = True
            continue
        is_stopped[start : end + 1] = True
        duration = end - start + 1
        if duration >= MIN_STOP_S:
            events.append({
                "start": df["timestamp"].iloc[start],
                "duration_s": duration,
                "lat": lat0,
                "lon": lon0,
                "kind": "pause" if part["is_paused"].any() else "slow",
                "start_index": start,
            })
    flags = df.assign(is_stopped=is_stopped, is_gap=is_gap, is_moving=~(is_stopped | is_gap))
    return flags, pd.DataFrame(events, columns=["start", "duration_s", "lat", "lon", "kind", "start_index"])


def hold_position_while_stopped(df: pd.DataFrame) -> pd.DataFrame:
    """During stops the rider is where they stopped: forward-fill position fields."""
    df = df.copy()
    columns = ["lat", "lon", "distance_m", "altitude_m"]
    filled = df[columns].ffill()
    df.loc[df["is_stopped"], columns] = filled.loc[df["is_stopped"], columns]
    df.loc[df["is_stopped"], "speed_mps"] = 0.0
    return df


# --- Altitude and grade --------------------------------------------------------

def smooth_altitude(df: pd.DataFrame, median_s: int = 5, mean_s: int = 15) -> pd.DataFrame:
    """Median filter (removes spikes) followed by a moving average (removes jitter)."""
    alt = df["altitude_m"]
    smooth = (
        alt.rolling(median_s, center=True, min_periods=1).median()
        .rolling(mean_s, center=True, min_periods=1).mean()
        .where(alt.notna())
    )
    return df.assign(altitude_smooth_m=smooth)


def distance_window_grade(
    df: pd.DataFrame, half_window_m: float = 25.0, max_grade: float = 0.25
) -> pd.DataFrame:
    """Grade over +-half_window_m of road instead of per second.

    Per-second distance steps are tiny, so per-second grade explodes; looking
    up the smoothed altitude 25 m behind and ahead gives a stable estimate.
    """
    ok = df["distance_m"].notna() & df["altitude_smooth_m"].notna()
    grade = pd.Series(np.nan, index=df.index)
    if ok.sum() >= 2:
        dist = df.loc[ok, "distance_m"].cummax().to_numpy()
        alt = df.loc[ok, "altitude_smooth_m"].to_numpy()
        ahead = np.interp(dist + half_window_m, dist, alt)
        behind = np.interp(dist - half_window_m, dist, alt)
        run = np.minimum(dist + half_window_m, dist[-1]) - np.maximum(dist - half_window_m, dist[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(run > 5.0, (ahead - behind) / run, np.nan)
        grade[ok] = np.clip(g, -max_grade, max_grade)
    return df.assign(grade=grade)


def elevation_gain_m(altitude: pd.Series, threshold_m: float = 1.0) -> float:
    """Total climbing, ignoring wiggles smaller than `threshold_m` (hysteresis)."""
    values = altitude.dropna().to_numpy()
    if len(values) == 0:
        return 0.0
    gain, ref = 0.0, values[0]
    for v in values[1:]:
        if v > ref + threshold_m:
            gain += v - ref
            ref = v
        elif v < ref - threshold_m:
            ref = v
    return gain


# --- Pipeline ------------------------------------------------------------------

def clean_ride(df: pd.DataFrame, gps_distance: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full cleaning pipeline. Set gps_distance for rides whose distance came from GPS."""
    df = remove_gps_spikes(df)
    if gps_distance:
        df = recompute_distance_from_gps(df)
    df = clip_speed(df)
    df, stops = detect_stops(df)
    df = hold_position_while_stopped(df)
    df = smooth_altitude(df)
    df = distance_window_grade(df)
    return df, stops
