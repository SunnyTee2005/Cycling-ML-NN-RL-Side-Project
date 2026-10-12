"""Calibrate CdA and Crr from rides with a power meter.

Rearranging the power model so the unknowns are on one side:

    eta * P_measured - P_gravity - P_accel  =  CdA * (0.5 rho v^3)  +  Crr * (m g v cos theta)
    \\__________________ y _______________/          \\___ x1 ___/            \\_____ x2 _____/

This is a linear regression with two coefficients and no intercept. We fit it
on 30-second windows of steady riding (averaging cancels sensor noise and
small timing offsets between power and speed).
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from cycling_coach import config
from cycling_coach.physics import ETA, G, Setup, accel_power, air_density, gravity_power, leg_power

WINDOW_S = 30
MIN_SPEED = 3.0          # m/s
MIN_POWER = 50.0         # W, skip coasting windows
MAX_ABS_ACCEL = 0.3      # m/s^2 averaged over the window: steady riding only
BOUNDS = ([0.15, 0.002], [0.60, 0.015])  # plausible CdA, Crr ranges

# Bikes without a power meter use literature values (assumed); the others reuse
# the TCR fit (the Reacto is ridden in a similar position).
ASSUMED = {
    "MATTS 6.40": {"cda": 0.45, "crr": 0.010},   # upright MTB, knobby tyres
    "Ubike": {"cda": 0.55, "crr": 0.010},        # upright city bike
}


def setup_for_bike(bike: str, cda: float, crr: float) -> Setup:
    """Mass from config; CdA/Crr from the fit unless the bike has assumed values."""
    mass = config.RIDER_MASS_KG + config.KIT_MASS_KG + config.BIKE_MASS_KG.get(bike, 8.0)
    if bike in ASSUMED:
        return Setup(mass, ASSUMED[bike]["cda"], ASSUMED[bike]["crr"])
    return Setup(mass, cda, crr)


def add_dynamics(ride: pd.DataFrame) -> pd.DataFrame:
    """Add acceleration (m/s^2) and air density to a cleaned 1 Hz ride."""
    speed = ride["speed_mps"].rolling(3, center=True, min_periods=1).mean()
    accel = speed.diff().rolling(3, center=True, min_periods=1).mean()
    temp = ride["temperature_c"].fillna(25.0) if "temperature_c" in ride else 25.0
    rho = air_density(ride["altitude_smooth_m"].fillna(100.0), temp)
    return ride.assign(speed_s=speed, accel=accel, rho=rho)


def regression_windows(ride: pd.DataFrame, mass_kg: float) -> pd.DataFrame:
    """Per-second regression terms averaged over non-overlapping windows."""
    r = add_dynamics(ride)
    v, grade = r["speed_s"], r["grade"]
    terms = pd.DataFrame({
        "y": ETA * r["power_w"] - gravity_power(mass_kg, v, grade) - accel_power(mass_kg, v, r["accel"]),
        "x_aero": 0.5 * r["rho"] * v**3,
        "x_roll": mass_kg * G * v * np.cos(np.arctan(grade)),
        "power": r["power_w"], "speed": v, "grade": grade, "accel": r["accel"],
    })
    valid = r["is_moving"] & r["power_w"].notna() & grade.notna() & (v > MIN_SPEED)
    window = np.arange(len(r)) // WINDOW_S
    g = terms[valid].groupby(window[valid])
    out = g.mean()
    out["n"] = g.size()
    keep = (out["n"] == WINDOW_S) & (out["power"] > MIN_POWER) & (out["accel"].abs() < MAX_ABS_ACCEL)
    return out[keep].drop(columns="n")


def fit_cda_crr(windows: pd.DataFrame) -> tuple[float, float]:
    """Robust least squares (soft-L1 loss down-weights windows hit by gusts)."""
    X = windows[["x_aero", "x_roll"]].to_numpy()
    y = windows["y"].to_numpy()

    def residuals(params):
        return X @ params - y

    result = least_squares(residuals, x0=[0.32, 0.005], bounds=BOUNDS, loss="soft_l1", f_scale=30.0)
    return float(result.x[0]), float(result.x[1])


def predict_windows(windows: pd.DataFrame, setup: Setup) -> pd.Series:
    """Predicted pedal power for regression windows."""
    wheel = setup.cda * windows["x_aero"] + setup.crr * windows["x_roll"]
    measured_wheel = ETA * windows["power"]
    # y = eta*P - grav - accel  =>  eta*P_pred = wheel + (eta*P - y)
    return (wheel + measured_wheel - windows["y"]) / ETA


def error_metrics(predicted: pd.Series, measured: pd.Series) -> dict:
    err = predicted - measured
    return {
        "n": len(err),
        "mae_w": float(err.abs().mean()),
        "rmse_w": float(np.sqrt((err**2).mean())),
        "bias_w": float(err.mean()),
        "mape": float((err.abs() / measured).mean()),
    }


def estimate_ride_power(ride: pd.DataFrame, setup: Setup, headwind=0.0) -> pd.Series:
    """Per-second estimated pedal power for a cleaned ride (0 when coasting/stopped)."""
    r = add_dynamics(ride)
    p = leg_power(setup, r["speed_s"], r["grade"].fillna(0.0), r["accel"].fillna(0.0), r["rho"], headwind)
    return pd.Series(np.where(r["is_moving"], np.clip(p, 0, None), 0.0), index=ride.index)
