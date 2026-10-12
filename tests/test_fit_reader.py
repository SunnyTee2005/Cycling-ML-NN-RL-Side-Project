from datetime import datetime, timezone

import pytest

from cycling_coach.io.fit_reader import (
    _decode_malformed_field,
    decode_byte_array,
    pause_intervals,
)


def test_decode_byte_array_big_endian():
    assert decode_byte_array((0x00, 0x00, 0x01, 0x3F), ">", signed=False) == 319


def test_decode_byte_array_little_endian():
    assert decode_byte_array((0x3F, 0x01, 0x00, 0x00), "<", signed=False) == 319


def test_decode_byte_array_negative_signed():
    assert decode_byte_array((0xFF, 0xFF, 0xFF, 0xFE), ">", signed=True) == -2


@pytest.mark.parametrize(
    "raw, signed",
    [((0xFF, 0xFF, 0xFF, 0xFF), False), ((0x7F, 0xFF, 0xFF, 0xFF), True)],
)
def test_decode_byte_array_invalid_marker_is_none(raw, signed):
    assert decode_byte_array(raw, ">", signed=signed) is None


def test_malformed_timestamp_uses_fit_epoch():
    # Bytes taken from a real malformed file in the dataset.
    ts = _decode_malformed_field("timestamp", (62, 105, 167, 190), ">")
    assert ts == datetime(2023, 3, 7, 8, 20, 14, tzinfo=timezone.utc)


def test_malformed_distance_is_scaled_to_metres():
    assert _decode_malformed_field("distance", (0, 0, 1, 63), ">") == pytest.approx(3.19)


def _t(second: int) -> datetime:
    return datetime(2025, 1, 1, 8, 0, second, tzinfo=timezone.utc)


def test_pause_intervals_pairs_stop_with_next_start():
    events = [(_t(0), "start"), (_t(10), "stop_all"), (_t(40), "start"), (_t(50), "stop_all")]
    assert pause_intervals(events) == [(_t(10), _t(40))]


def test_pause_intervals_ignores_repeated_stops():
    events = [(_t(10), "stop_all"), (_t(12), "stop"), (_t(20), "start")]
    assert pause_intervals(events) == [(_t(10), _t(20))]
