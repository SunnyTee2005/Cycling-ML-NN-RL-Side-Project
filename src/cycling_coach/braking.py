"""Hard-braking events: sudden speed drops while riding."""

import numpy as np
import pandas as pd

MIN_SPEED = 4.0        # m/s before braking (~15 km/h)
MIN_DROP = 4.0         # m/s lost within WINDOW_S seconds (~0.2 g)
MAX_DROP = 10.0        # larger drops in 2 s are GPS artefacts
WINDOW_S = 2
DEDUPE_S = 10          # one event per braking manoeuvre


def hard_brakes(ride: pd.DataFrame) -> pd.DataFrame:
    """Return one row per hard-braking event with time, location and speed drop."""
    v = ride["speed_mps"].where(ride["is_moving"] | ride["is_stopped"])
    before = v.shift(WINDOW_S)
    drop = before - v
    candidate = (before > MIN_SPEED) & drop.between(MIN_DROP, MAX_DROP)
    idx = np.flatnonzero(candidate.to_numpy())
    events, last = [], -DEDUPE_S - 1
    for i in idx:
        if i - last <= DEDUPE_S:
            if drop.iloc[i] > events[-1]["drop_mps"]:  # keep the strongest second
                events[-1].update(_event(ride, i, drop.iloc[i], before.iloc[i]))
            last = i
            continue
        events.append(_event(ride, i, drop.iloc[i], before.iloc[i]))
        last = i
    return pd.DataFrame(events, columns=["time", "lat", "lon", "drop_mps", "speed_before_mps"])


def _event(ride: pd.DataFrame, i: int, drop: float, before: float) -> dict:
    j = max(i - WINDOW_S, 0)  # where braking started
    return {"time": ride["timestamp"].iloc[i], "lat": ride["lat"].iloc[j], "lon": ride["lon"].iloc[j],
            "drop_mps": float(drop), "speed_before_mps": float(before)}
