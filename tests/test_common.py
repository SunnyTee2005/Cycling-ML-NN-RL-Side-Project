import numpy as np
import pandas as pd
import pytest

from cycling_coach.io.common import SENSOR_COLUMNS, blank_placeholder_values, file_format


@pytest.mark.parametrize(
    "name, expected",
    [("1.fit", "fit"), ("1.fit.gz", "fit"), ("1.GPX", "gpx"), ("1.gpx.gz", "gpx")],
)
def test_file_format(name, expected):
    assert file_format(name) == expected


def test_file_format_rejects_tcx():
    with pytest.raises(ValueError):
        file_format("1.tcx.gz")


def _ride(n=120) -> pd.DataFrame:
    df = pd.DataFrame({c: np.linspace(1, 2, n) for c in SENSOR_COLUMNS})
    return df


def test_constant_column_is_treated_as_missing():
    df = _ride().assign(distance_m=0.0, altitude_m=-500.0)
    out = blank_placeholder_values(df)
    assert out["distance_m"].isna().all()
    assert out["altitude_m"].isna().all()
    assert out["speed_mps"].notna().all()


def test_zero_heart_rate_is_missing():
    df = _ride().assign(heart_rate_bpm=[0.0] + [120.0] * 119)
    out = blank_placeholder_values(df)
    assert np.isnan(out.loc[0, "heart_rate_bpm"])
    assert out.loc[1, "heart_rate_bpm"] == 120.0
