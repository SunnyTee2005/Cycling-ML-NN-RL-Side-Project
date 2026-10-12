"""Read Garmin FIT files (Bryton and Garmin devices both use this format).

Some files in the dataset (written by a third-party app in 2022-2023) declare
multi-byte fields as raw byte arrays in big-endian order. `fitdecode` returns
those as tuples of bytes, so we decode them ourselves.
"""

import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fitdecode
import pandas as pd

from cycling_coach.io.common import SENSOR_COLUMNS, read_bytes

FIT_EPOCH = datetime(1989, 12, 31, tzinfo=timezone.utc)
SEMICIRCLE_TO_DEG = 180.0 / 2**31

# FIT record field -> our column name. Earlier entries win when both exist.
FIELD_MAP = {
    "position_lat": "lat",
    "position_long": "lon",
    "enhanced_altitude": "altitude_m",
    "altitude": "altitude_m",
    "distance": "distance_m",
    "enhanced_speed": "speed_mps",
    "speed": "speed_mps",
    "power": "power_w",
    "heart_rate": "heart_rate_bpm",
    "cadence": "cadence_rpm",
    "temperature": "temperature_c",
}

# (scale, offset) from the FIT profile, needed only when we decode bytes ourselves.
RAW_SCALING = {
    "enhanced_altitude": (5.0, 500.0),
    "altitude": (5.0, 500.0),
    "distance": (100.0, 0.0),
    "enhanced_speed": (1000.0, 0.0),
    "speed": (1000.0, 0.0),
}
SIGNED_FIELDS = {"position_lat", "position_long", "temperature"}
PAUSE_EVENT_TYPES = {"stop", "stop_all", "stop_disable", "stop_disable_all"}


@dataclass
class FitActivity:
    records: pd.DataFrame              # one row per FIT record message
    pauses: list[tuple[datetime, datetime]] = field(default_factory=list)
    device: str = "unknown"
    warning: str | None = None         # set when the file was truncated/corrupt


class _TolerantProcessor(fitdecode.DefaultDataProcessor):
    """Skip timestamp conversion for byte-array fields instead of crashing."""

    def process_type_date_time(self, reader, field_data):
        if not isinstance(field_data.value, tuple):
            super().process_type_date_time(reader, field_data)

    def process_type_local_date_time(self, reader, field_data):
        if not isinstance(field_data.value, tuple):
            super().process_type_local_date_time(reader, field_data)


def decode_byte_array(raw: tuple[int, ...], endian: str, signed: bool) -> int | None:
    """Join a tuple of bytes into one integer; None if it is the 'invalid' marker."""
    data = bytes(raw)
    if all(b == 0xFF for b in data):
        return None
    byteorder = "big" if endian == ">" else "little"
    value = int.from_bytes(data, byteorder, signed=signed)
    if signed and value == 2 ** (8 * len(data) - 1) - 1:  # e.g. 0x7FFFFFFF
        return None
    return value


def _decode_malformed_field(name: str, raw: tuple[int, ...], endian: str):
    """Decode a byte-array field and apply the FIT profile scaling."""
    value = decode_byte_array(raw, endian, signed=name in SIGNED_FIELDS)
    if value is None:
        return None
    if name == "timestamp":
        return FIT_EPOCH + timedelta(seconds=value)
    scale, offset = RAW_SCALING.get(name, (1.0, 0.0))
    return value / scale - offset


def _field_value(fd, endian: str):
    if isinstance(fd.value, tuple) and isinstance(fd.raw_value, tuple):
        return _decode_malformed_field(fd.name, fd.raw_value, endian)
    return fd.value


def _record_to_row(frame: fitdecode.FitDataMessage) -> dict:
    endian = frame.def_mesg.endian
    values = {fd.name: _field_value(fd, endian) for fd in frame.fields}
    row = {"timestamp": values.get("timestamp")}
    for fit_name, column in FIELD_MAP.items():
        if row.get(column) is None and values.get(fit_name) is not None:
            row[column] = values[fit_name]
    for column in ("lat", "lon"):
        if row.get(column) is not None:
            row[column] = row[column] * SEMICIRCLE_TO_DEG
    return row


def pause_intervals(events: list[tuple[datetime, str]]) -> list[tuple[datetime, datetime]]:
    """Pair timer 'stop' events with the next 'start' event.

    `events` is a time-ordered list of (timestamp, event_type) for timer events.
    A trailing stop without a following start (end of ride) is ignored.
    """
    intervals = []
    paused_at = None
    for ts, event_type in events:
        if event_type in PAUSE_EVENT_TYPES and paused_at is None:
            paused_at = ts
        elif event_type == "start" and paused_at is not None:
            intervals.append((paused_at, ts))
            paused_at = None
    return intervals


def _device_name(frame: fitdecode.FitDataMessage) -> str:
    manufacturer = frame.get_value("manufacturer", fallback=None)
    product = frame.get_value("garmin_product", fallback=None) or frame.get_value(
        "product", fallback=None
    )
    if isinstance(manufacturer, tuple):  # malformed third-party file
        return "unknown"
    return f"{manufacturer} {product}".strip()


def read_fit(path: Path) -> FitActivity:
    rows, timer_events = [], []
    device, warning = "unknown", None
    reader = fitdecode.FitReader(
        io.BytesIO(read_bytes(path)),
        processor=_TolerantProcessor(),
        check_crc=fitdecode.CrcCheck.DISABLED,
    )
    try:
        for frame in reader:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name == "record":
                rows.append(_record_to_row(frame))
            elif (
                frame.name == "event"
                and frame.has_field("timestamp")
                and frame.get_value("event", fallback=None) == "timer"
            ):
                ts = _field_value(frame.get_field("timestamp"), frame.def_mesg.endian)
                timer_events.append((ts, frame.get_value("event_type", fallback=None)))
            elif frame.name == "file_id":
                device = _device_name(frame)
    except fitdecode.FitError as exc:
        # Devices sometimes leave a truncated file after a crash; keep what we read.
        if not rows:
            raise
        warning = f"truncated: {exc}"
    finally:
        reader.close()

    records = pd.DataFrame(rows, columns=["timestamp", *SENSOR_COLUMNS])
    records["timestamp"] = pd.to_datetime(records["timestamp"], utc=True)
    return FitActivity(records, pause_intervals(timer_events), device, warning)
