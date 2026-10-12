"""Read GPX tracks, including Garmin TrackPointExtension sensor data (hr, cad, temp)."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from cycling_coach.io.common import SENSOR_COLUMNS, read_bytes

# Extension tag (namespace stripped) -> our column name.
EXTENSION_MAP = {
    "hr": "heart_rate_bpm",
    "cad": "cadence_rpm",
    "atemp": "temperature_c",
    "power": "power_w",
}


def _local_name(tag: str) -> str:
    """'{http://...}trkpt' -> 'trkpt'."""
    return tag.rsplit("}", 1)[-1]


def _parse_trackpoint(trkpt: ET.Element) -> dict:
    row = {"lat": float(trkpt.get("lat")), "lon": float(trkpt.get("lon"))}
    for child in trkpt.iter():
        name = _local_name(child.tag)
        if child.text is None:
            continue
        if name == "time":
            row["timestamp"] = child.text
        elif name == "ele":
            row["altitude_m"] = float(child.text)
        elif name in EXTENSION_MAP:
            row[EXTENSION_MAP[name]] = float(child.text)
    return row


def read_gpx(path: Path) -> pd.DataFrame:
    root = ET.fromstring(read_bytes(path))
    rows = [_parse_trackpoint(el) for el in root.iter() if _local_name(el.tag) == "trkpt"]
    df = pd.DataFrame(rows, columns=["timestamp", *SENSOR_COLUMNS])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    return df
