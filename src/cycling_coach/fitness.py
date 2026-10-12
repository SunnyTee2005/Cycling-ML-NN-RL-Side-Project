"""Fitness trend from everyday commutes.

Riding the same route again and again is a free, repeated fitness test:
if you cover more road per heartbeat, or produce more power per heartbeat,
your aerobic fitness has improved.
"""

import pandas as pd

MIN_HR_COVERAGE = 0.8


def commute_efficiency(ride: pd.DataFrame, est_power: pd.Series | None = None) -> dict:
    """Metres per heartbeat and (estimated) watts per bpm over the moving seconds."""
    moving = ride[ride["is_moving"]]
    hr = moving["heart_rate_bpm"]
    if len(moving) < 120 or hr.notna().mean() < MIN_HR_COVERAGE:
        return {"metres_per_beat": float("nan"), "watts_per_bpm": float("nan"), "avg_hr": float("nan")}
    beats = (hr.dropna() / 60.0).sum()           # one row = one second
    metres = moving["speed_mps"].fillna(0).sum()
    out = {"metres_per_beat": metres / beats, "avg_hr": hr.mean(), "watts_per_bpm": float("nan")}
    if est_power is not None:
        out["watts_per_bpm"] = est_power[moving.index].mean() / hr.mean()
    return out
