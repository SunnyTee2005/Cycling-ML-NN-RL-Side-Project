import gzip

import pytest

from cycling_coach.io.gpx_reader import read_gpx

GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1"
     xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">
 <trk><trkseg>
  <trkpt lat="24.1" lon="120.7">
   <ele>84.2</ele><time>2020-11-06T23:30:24+00:00</time>
   <extensions><gpxtpx:TrackPointExtension>
     <gpxtpx:hr>120</gpxtpx:hr><gpxtpx:cad>85</gpxtpx:cad>
   </gpxtpx:TrackPointExtension></extensions>
  </trkpt>
  <trkpt lat="24.2" lon="120.8">
   <ele>85.0</ele><time>2020-11-06T23:30:26Z</time>
  </trkpt>
 </trkseg></trk>
</gpx>"""


@pytest.fixture(params=["plain", "gz"])
def gpx_file(tmp_path, request):
    if request.param == "gz":
        path = tmp_path / "ride.gpx.gz"
        path.write_bytes(gzip.compress(GPX.encode()))
    else:
        path = tmp_path / "ride.gpx"
        path.write_text(GPX)
    return path


def test_read_gpx_parses_points_and_extensions(gpx_file):
    df = read_gpx(gpx_file)
    assert len(df) == 2
    assert df.loc[0, "lat"] == 24.1
    assert df.loc[0, "altitude_m"] == 84.2
    assert df.loc[0, "heart_rate_bpm"] == 120
    assert df.loc[0, "cadence_rpm"] == 85
    assert (df["timestamp"].diff().dt.total_seconds().iloc[1]) == 2


def test_read_gpx_missing_sensors_are_nan(gpx_file):
    df = read_gpx(gpx_file)
    assert df["power_w"].isna().all()
    assert df["heart_rate_bpm"].isna().iloc[1]
