"""Shared helpers for activity readers."""

import gzip
from pathlib import Path

import pandas as pd

# Sensor columns every reader produces (missing sensors are filled with NaN).
SENSOR_COLUMNS = [
    "lat",
    "lon",
    "altitude_m",
    "distance_m",
    "speed_mps",
    "power_w",
    "heart_rate_bpm",
    "cadence_rpm",
    "temperature_c",
]


def read_bytes(path: Path) -> bytes:
    """Return the file content, transparently decompressing `.gz` files."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return f.read()
    return path.read_bytes()


def file_format(path: Path) -> str:
    """Return 'fit' or 'gpx' based on the file name (ignoring a trailing .gz)."""
    name = Path(path).name.lower().removesuffix(".gz")
    if name.endswith(".fit"):
        return "fit"
    if name.endswith(".gpx"):
        return "gpx"
    raise ValueError(f"Unsupported activity file: {path}")


# Columns where a device writes a constant dummy value (e.g. distance always 0,
# altitude always -500 m) when the sensor is absent.
_PLACEHOLDER_CHECK_COLUMNS = ["altitude_m", "distance_m", "speed_mps", "power_w", "cadence_rpm"]


def blank_placeholder_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace dummy sensor values with NaN so they read as 'not recorded'.

    - heart rate 0 means no reading;
    - a column holding one single value for the whole ride is a placeholder.
    """
    df = df.copy()
    df.loc[df["heart_rate_bpm"] == 0, "heart_rate_bpm"] = float("nan")
    for column in _PLACEHOLDER_CHECK_COLUMNS:
        if len(df) > 60 and df[column].nunique() == 1:
            df[column] = float("nan")
    return df
