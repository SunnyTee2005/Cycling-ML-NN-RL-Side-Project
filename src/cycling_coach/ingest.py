"""Load the Strava activity list and parse individual ride files into 1 Hz tables."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cycling_coach.io.common import SENSOR_COLUMNS, blank_placeholder_values, file_format
from cycling_coach.io.fit_reader import read_fit
from cycling_coach.io.gpx_reader import read_gpx
from cycling_coach.resample import (
    add_distance_if_missing,
    add_raw_grade,
    add_speed_if_missing,
    mark_pauses,
    to_1hz,
)

RIDE_TYPES = {"Ride", "E-Bike Ride"}
RIDE_COLUMNS = [
    "timestamp",
    *SENSOR_COLUMNS,
    "grade_raw",
    "is_filled",
    "is_paused",
]

# Strava CSV column -> our name. Note: Strava's CSV has two "Distance" columns;
# pandas renames the second one (metres, more precise) to "Distance.1".
ACTIVITY_COLUMNS = {
    "Activity ID": "activity_id",
    "Activity Date": "start_time",
    "Activity Name": "name",
    "Activity Type": "activity_type",
    "Activity Gear": "gear",
    "Commute": "commute_tag",
    "Filename": "filename",
    "Distance.1": "distance_m",
    "Moving Time": "moving_time_s",
    "Elapsed Time": "elapsed_time_s",
    "Weighted Average Power": "weighted_power_w",
}


@dataclass
class ParsedRide:
    data: pd.DataFrame
    device: str
    warning: str | None


def load_activity_index(csv_path: Path) -> pd.DataFrame:
    """Read Strava's activities.csv and keep only rides that have a data file."""
    df = pd.read_csv(csv_path)[list(ACTIVITY_COLUMNS)].rename(columns=ACTIVITY_COLUMNS)
    df["start_time"] = pd.to_datetime(
        df["start_time"], format="%b %d, %Y, %I:%M:%S %p", utc=True
    )
    # Only a real power meter produces Strava's weighted average power;
    # Strava's estimated power for rides without one leaves it empty.
    df["has_power_meter"] = df["weighted_power_w"].notna()
    df = df.drop(columns="weighted_power_w")
    rides = df[df["activity_type"].isin(RIDE_TYPES) & df["filename"].notna()]
    return rides.sort_values("start_time").reset_index(drop=True)


def parse_activity_file(path: Path) -> ParsedRide:
    """Parse one FIT/GPX file into the standard 1 Hz ride table."""
    if file_format(path) == "fit":
        fit = read_fit(path)
        raw, pauses, device, warning = fit.records, fit.pauses, fit.device, fit.warning
    else:
        raw, pauses, device, warning = read_gpx(path), [], "gpx", None

    df = blank_placeholder_values(raw)
    df = add_distance_if_missing(df)
    df = to_1hz(df)
    df = mark_pauses(df, pauses)
    df = add_speed_if_missing(df)
    df = add_raw_grade(df)
    return ParsedRide(df[RIDE_COLUMNS], device, warning)


def crop_to_activity(df: pd.DataFrame, start: pd.Timestamp, elapsed_s: float) -> pd.DataFrame:
    """Keep only the rows inside Strava's activity window.

    One recording can be split into several Strava activities that all point
    to the same file; cropping makes each activity contain only its own part.
    """
    end = start + pd.Timedelta(seconds=elapsed_s)
    inside = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    return df[inside].reset_index(drop=True)
