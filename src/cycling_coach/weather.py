"""Wind and weather helpers."""

import numpy as np
import pandas as pd

# Strava's wind is measured at 10 m; near the ground (rider height, buildings) it is
# weaker. 0.6 is a common rule of thumb; calibrated against power data in step 10.
DEFAULT_WIND_FACTOR = 0.6


def bearing_deg(lat1, lon1, lat2, lon2):
    """Compass direction of travel from point 1 to point 2 (0 = north, 90 = east)."""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def headwind(ride: pd.DataFrame, wind_speed: float, wind_from_deg: float,
             factor: float = DEFAULT_WIND_FACTOR, lag_s: int = 5) -> pd.Series:
    """Headwind component (m/s, negative = tailwind) for each second of a ride.

    Heading comes from positions `lag_s` seconds apart (single-second GPS steps are
    too noisy). Meteorological wind direction is where the wind comes FROM, so riding
    toward that direction means a headwind.
    """
    heading = bearing_deg(ride["lat"].shift(lag_s), ride["lon"].shift(lag_s),
                          ride["lat"].shift(-lag_s), ride["lon"].shift(-lag_s))
    angle = np.radians(wind_from_deg - heading)
    hw = factor * wind_speed * np.cos(angle)
    return pd.Series(hw, index=ride.index).fillna(0.0)


def linear_fit(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Least-squares line y = a + b x with a bootstrap 90% interval for the slope."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    r2 = 1 - resid.var() / y.var()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    slopes = [np.polyfit(x[i], y[i], 1)[0] for i in idx]
    lo, hi = np.percentile(slopes, [5, 95])
    return {"slope": b, "intercept": a, "r2": r2, "slope_lo": lo, "slope_hi": hi, "n": len(x)}
