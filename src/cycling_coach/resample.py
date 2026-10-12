"""Turn irregular samples into a regular one-row-per-second table."""

from datetime import datetime

import numpy as np
import pandas as pd

from cycling_coach.geo import haversine_m

MAX_RIDE_DURATION = pd.Timedelta(hours=48)


def add_distance_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill `distance_m` with cumulative GPS distance when the device did not record it."""
    if df["distance_m"].notna().any():
        return df
    gps = df[["lat", "lon"]].dropna()
    step = haversine_m(gps["lat"].shift(), gps["lon"].shift(), gps["lat"], gps["lon"])
    df = df.copy()
    df.loc[gps.index, "distance_m"] = np.nan_to_num(step).cumsum()
    return df


def interpolate_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """Linearly fill runs of at most `max_gap` consecutive NaNs; leave longer runs NaN."""
    is_nan = series.isna()
    run_id = (is_nan != is_nan.shift()).cumsum()
    run_length = is_nan.groupby(run_id).transform("size")
    fillable = is_nan & (run_length <= max_gap)
    filled = series.interpolate(limit_area="inside")
    return series.where(~fillable, filled)


def to_1hz(df: pd.DataFrame, max_gap_s: int = 5) -> pd.DataFrame:
    """Resample to exactly one row per second.

    Rows that did not exist in the source get `is_filled=True`. Gaps of up to
    `max_gap_s` seconds are interpolated; longer gaps (pauses, signal loss)
    stay NaN so later steps can see them.
    """
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.assign(timestamp=df["timestamp"].dt.floor("s"))
    df = df.drop_duplicates("timestamp", keep="first").set_index("timestamp")
    if df.empty:
        raise ValueError("no timestamped samples")
    if df.index[-1] - df.index[0] > MAX_RIDE_DURATION:
        raise ValueError("implausible ride duration (> 48 h)")

    full_index = pd.date_range(df.index[0], df.index[-1], freq="s", name="timestamp")
    out = df.reindex(full_index)
    for column in out.columns:
        out[column] = interpolate_short_gaps(out[column].astype(float), max_gap_s)
    out["is_filled"] = ~out.index.isin(df.index)
    return out.reset_index()


def mark_pauses(df: pd.DataFrame, pauses: list[tuple[datetime, datetime]]) -> pd.DataFrame:
    """Add `is_paused`: True for seconds inside a device timer pause [start, end)."""
    is_paused = np.zeros(len(df), dtype=bool)
    for start, end in pauses:
        is_paused |= ((df["timestamp"] >= start) & (df["timestamp"] < end)).to_numpy()
    return df.assign(is_paused=is_paused)


def add_speed_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Derive speed (m/s) from 1 Hz distance when the device did not record it."""
    if df["speed_mps"].notna().any():
        return df
    return df.assign(speed_mps=df["distance_m"].diff())


def add_raw_grade(df: pd.DataFrame, min_step_m: float = 1.0) -> pd.DataFrame:
    """Unsmoothed grade (rise / run) per second. Very noisy; smoothed in the cleaning step."""
    d_dist = df["distance_m"].diff()
    d_alt = df["altitude_m"].diff()
    grade = (d_alt / d_dist).where(d_dist >= min_step_m)
    return df.assign(grade_raw=grade)
