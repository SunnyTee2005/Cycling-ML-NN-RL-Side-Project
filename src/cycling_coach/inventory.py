"""Per-ride summaries and duplicate detection for the data inventory report."""

import pandas as pd

# Sensor fields reported in the inventory, with display names.
FIELDS = {
    "lat": "GPS",
    "altitude_m": "Altitude",
    "heart_rate_bpm": "Heart rate",
    "cadence_rpm": "Cadence",
    "power_w": "Power",
    "temperature_c": "Temperature",
}
HAS_FIELD_THRESHOLD = 0.5  # a ride "has" a field if >50% of recorded seconds have it

# Rough bounding box of Taiwan's main island.
TAIWAN_LAT = (21.8, 25.4)
TAIWAN_LON = (119.9, 122.1)


def device_group(device: str | None) -> str:
    """Map raw FIT manufacturer/product strings to a readable device group."""
    if device is None:
        return "Other"
    if device.startswith("bryton"):
        return "Bryton Rider 420"
    if device.startswith("garmin"):
        return "Garmin Forerunner 255"
    if device == "unknown":
        return "Xiaomi watch"
    if device == "gpx":
        return "GPX export"
    return "Other"


def summarize_ride(ride: pd.DataFrame) -> dict:
    """Duration, distance, pause time and sensor coverage for one 1 Hz ride."""
    recorded = ride[~ride["is_filled"]]
    summary = {
        "elapsed_s": len(ride),
        "recorded_s": len(recorded),
        "paused_s": int(ride["is_paused"].sum()),
        "ride_distance_m": ride["distance_m"].max() - ride["distance_m"].min(),
        "median_lat": ride["lat"].median(),
        "median_lon": ride["lon"].median(),
    }
    for column in FIELDS:
        coverage = recorded[column].notna().mean() if len(recorded) else 0.0
        summary[f"cov_{column}"] = coverage
        summary[f"has_{column}"] = coverage > HAS_FIELD_THRESHOLD
    return summary


def is_in_taiwan(lat: pd.Series, lon: pd.Series) -> pd.Series:
    return lat.between(*TAIWAN_LAT) & lon.between(*TAIWAN_LON)


def find_duplicates(rides: pd.DataFrame, min_overlap: float = 0.5) -> pd.Series:
    """Return, for each ride, the activity_id it duplicates (or NA).

    Two rides are duplicates when their time windows overlap by more than
    `min_overlap` of the shorter one (e.g. the same ride recorded on both the
    bike computer and the watch).
    The ride with more recorded seconds is kept as the original.
    Expects columns: activity_id, start_time, elapsed_s, recorded_s.
    """
    df = rides.sort_values("start_time")
    end = df["start_time"] + pd.to_timedelta(df["elapsed_s"], unit="s")
    duplicate_of = pd.Series(pd.NA, index=df.index, dtype="Int64")

    keeper = None  # index of the current "original" ride
    for i in df.index:
        if keeper is not None:
            overlap = (min(end[i], end[keeper]) - df.at[i, "start_time"]).total_seconds()
            shorter = min(df.at[i, "elapsed_s"], df.at[keeper, "elapsed_s"])
            if shorter > 0 and overlap / shorter > min_overlap:
                if df.at[i, "recorded_s"] > df.at[keeper, "recorded_s"]:
                    duplicate_of[keeper] = df.at[i, "activity_id"]
                    keeper = i
                else:
                    duplicate_of[i] = df.at[keeper, "activity_id"]
                continue
        keeper = i
    return duplicate_of.reindex(rides.index)
