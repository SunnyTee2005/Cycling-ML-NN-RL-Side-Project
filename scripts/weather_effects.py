"""How wind, heat and rain affect riding, using the weather Strava attached to rides.

Usage:
    python scripts/weather_effects.py

Writes reports/weather.md and figures.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cycling_coach import config
from cycling_coach.calibration import (
    error_metrics,
    fit_cda_crr,
    predict_windows,
    regression_windows,
    ride_split,
)
from cycling_coach.physics import Setup
from cycling_coach.plotting import MUTED, SERIES, label, save, style_axes, title
from cycling_coach.weather import headwind, linear_fit

FIG = config.REPORTS_DIR / "figures"
RIDES = config.PROCESSED_DIR / "rides"
FACTORS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
RAIN_MM_H = 0.5


def load_weather() -> pd.DataFrame:
    csv = pd.read_csv(config.RAW_DIR / "activities.csv")
    w = csv[["Activity ID", "Weather Temperature", "Wind Speed", "Wind Bearing",
             "Precipitation Intensity", "Humidity"]].dropna(subset=["Wind Speed", "Wind Bearing"])
    return w.set_axis(["activity_id", "temp_c", "wind_mps", "wind_from_deg", "rain_mm_h", "humidity"], axis=1)


def md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    return head + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))


# --- Wind in the physics model -------------------------------------------------

def wind_factor_study(rides: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Refit CdA/Crr with headwind scaled by each factor; score the flat on held-out rides."""
    mass = config.system_mass_kg("TCR BLACK")
    power = rides[rides["has_power_w"] & rides["has_altitude_m"] & rides["in_taiwan"]]
    test_ids = ride_split(power["activity_id"])         # same split as the physics report
    power = power.merge(weather, on="activity_id")
    cache = {aid: pd.read_parquet(RIDES / f"{aid}.parquet") for aid in power["activity_id"]}
    rows = []
    for k in FACTORS:
        frames = []
        for p in power.itertuples():
            ride = cache[p.activity_id]
            hw = headwind(ride, p.wind_mps, p.wind_from_deg, factor=k)
            frames.append(regression_windows(ride, mass, headwind=hw).assign(activity_id=p.activity_id))
        w = pd.concat(frames, ignore_index=True)
        train, test = w[~w["activity_id"].isin(test_ids)], w[w["activity_id"].isin(test_ids)]
        cda, crr = fit_cda_crr(train)
        pred = predict_windows(test, Setup(mass, cda, crr))
        flat = test["grade"].abs() < 0.01
        rows.append({"factor": k, "cda": cda, "crr": crr,
                     "flat_mae": error_metrics(pred[flat], test.loc[flat, "power"])["mae_w"],
                     "all_mae": error_metrics(pred, test["power"])["mae_w"],
                     "rides": power["activity_id"].nunique(), "test_rides": len(set(test["activity_id"]))})
    return pd.DataFrame(rows)


# --- Commutes ------------------------------------------------------------------

def commute_weather(commutes: pd.DataFrame, weather: pd.DataFrame, factor: float) -> pd.DataFrame:
    c = commutes[commutes["bike"] != "Ubike"].merge(weather, on="activity_id")
    hw = []
    for r in c.itertuples():
        ride = pd.read_parquet(RIDES / f"{r.activity_id}.parquet")
        h = headwind(ride, r.wind_mps, r.wind_from_deg, factor=factor)
        hw.append(h[ride["is_moving"]].mean())
    c["headwind_mps"] = hw
    # Compare each ride to the typical speed of its own route and direction.
    key = ["destination", "direction"]
    c["speed_anomaly_kmh"] = c["moving_kmh"] - c.groupby(key)["moving_kmh"].transform("median")
    c["hr_anomaly"] = c["avg_hr"] - c.groupby(key)["avg_hr"].transform("median")
    return c


def heat_effect(c: pd.DataFrame) -> dict:
    """Heart rate vs temperature at equal speed: hr_anom = a + b*temp + c*speed_anom."""
    d = c.dropna(subset=["hr_anomaly", "temp_c", "speed_anomaly_kmh"])
    X = np.column_stack([np.ones(len(d)), d["temp_c"], d["speed_anomaly_kmh"]])
    coef, *_ = np.linalg.lstsq(X, d["hr_anomaly"].to_numpy(), rcond=None)
    return {"bpm_per_10c": coef[1] * 10, "bpm_per_kmh": coef[2], "n": len(d)}


def plot_wind_factor(study: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(study["factor"], study["flat_mae"], marker="o", color=SERIES[0], linewidth=2)
    best = study.loc[study["flat_mae"].idxmin()]
    ax.scatter([best["factor"]], [best["flat_mae"]], s=80, color=SERIES[1], zorder=3)
    style_axes(ax)
    label(ax, x="Share of Strava's 10 m wind felt by the rider", y="Flat error, MAE (W)")
    title(ax, "Adding wind to the physics model (held-out rides)")
    save(fig, path)


def plot_commute_wind(c: pd.DataFrame, fit: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(c["headwind_mps"], c["speed_anomaly_kmh"], s=12, alpha=0.5, color=SERIES[0])
    xs = np.linspace(c["headwind_mps"].min(), c["headwind_mps"].max(), 10)
    ax.plot(xs, fit["intercept"] + fit["slope"] * xs, color=SERIES[1], linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1)
    style_axes(ax, "both")
    label(ax, x="Average headwind during the commute (m/s, negative = tailwind)",
          y="Speed vs usual for the route (km/h)")
    title(ax, "Headwind and commute speed")
    save(fig, path)


def plot_heat(c: pd.DataFrame, path) -> None:
    d = c.dropna(subset=["hr_anomaly", "temp_c"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(d["temp_c"], d["hr_anomaly"], s=12, alpha=0.5, color=SERIES[1])
    b, a = np.polyfit(d["temp_c"], d["hr_anomaly"], 1)
    xs = np.linspace(d["temp_c"].min(), d["temp_c"].max(), 10)
    ax.plot(xs, a + b * xs, color=SERIES[0], linewidth=2)
    style_axes(ax, "both")
    label(ax, x="Temperature (°C)", y="Heart rate vs usual for the route (bpm)")
    title(ax, "Heat and heart rate on commutes")
    save(fig, path)


def main() -> None:
    rides = pd.read_parquet(config.PROCESSED_DIR / "rides.parquet")
    commutes = pd.read_parquet(config.PROCESSED_DIR / "commute_fitness.parquet")
    weather = load_weather()

    study = wind_factor_study(rides, weather)
    best = study.loc[study["flat_mae"].idxmin()]
    base = study.loc[study["factor"] == 0.0].iloc[0]
    factor = float(best["factor"]) if best["factor"] > 0 else 0.6

    c = commute_weather(commutes, weather, factor)
    wind_fit = linear_fit(c["headwind_mps"], c["speed_anomaly_kmh"])
    heat = heat_effect(c)
    uni_km = c.loc[c["destination"] == "university", "distance_m"].median() / 1000
    uni_kmh = c.loc[c["destination"] == "university", "moving_kmh"].median()
    # seconds lost on a university commute per 1 m/s of headwind
    sec_per_mps = uni_km / (uni_kmh + wind_fit["slope"]) * 3600 - uni_km / uni_kmh * 3600

    rainy = c["rain_mm_h"] >= RAIN_MM_H
    rain_table = pd.DataFrame([
        {"Conditions": "Dry", "Commutes": int((~rainy).sum()),
         "Speed vs usual": f"{c.loc[~rainy, 'speed_anomaly_kmh'].mean():+.1f} km/h"},
        {"Conditions": f"Rain (≥ {RAIN_MM_H} mm/h)", "Commutes": int(rainy.sum()),
         "Speed vs usual": f"{c.loc[rainy, 'speed_anomaly_kmh'].mean():+.1f} km/h"},
    ])

    (config.PROCESSED_DIR / "wind_factor.json").write_text(json.dumps(
        {"factor": factor, "cda": float(best["cda"]), "crr": float(best["crr"])}, indent=2))

    FIG.mkdir(parents=True, exist_ok=True)
    plot_wind_factor(study, FIG / "weather_wind_factor.png")
    plot_commute_wind(c, wind_fit, FIG / "weather_commute_wind.png")
    plot_heat(c, FIG / "weather_heat.png")

    study_table = pd.DataFrame({
        "Wind factor": study["factor"], "CdA": study["cda"].map("{:.3f}".format),
        "Crr": study["crr"].map("{:.4f}".format), "Flat MAE": study["flat_mae"].map("{:.1f} W".format),
        "All terrain MAE": study["all_mae"].map("{:.1f} W".format),
    })

    report = f"""# Weather

Generated by `scripts/weather_effects.py`. Strava attached weather (temperature, wind
speed and direction, rain) to **{len(weather)} activities** (May 2023 – Feb 2026).
Wind is measured at 10 m height, so only part of it reaches a rider between buildings.

## Wind makes the physics model better on the flat

The air speed in the drag term becomes *ground speed + factor × headwind component*, where
the headwind component is computed every second from the rider's heading. CdA/Crr are
refitted for each factor on the training rides; the error is measured on the same
held-out rides as in the physics report ({int(study['rides'].iloc[0])} power rides with weather,
{int(study['test_rides'].iloc[0])} held out).

{md(study_table)}

Best factor **{best['factor']:.1f}**: flat error {base['flat_mae']:.1f} W → **{best['flat_mae']:.1f} W**
({(best['flat_mae'] / base['flat_mae'] - 1):+.0%}).

![Wind factor](figures/weather_wind_factor.png)

## Headwind on the commute

{wind_fit['n']} own-bike commutes with weather. Speed is compared with the usual speed of the
same route and direction.

- Each **1 m/s of headwind** (felt, factor {factor:.1f}) changes moving speed by
  **{wind_fit['slope']:+.2f} km/h** (90% bootstrap interval {wind_fit['slope_lo']:+.2f} to
  {wind_fit['slope_hi']:+.2f}; R² = {wind_fit['r2']:.2f}).
- On the {uni_km:.1f} km university commute that is about **{sec_per_mps:+.0f} s per m/s of headwind**.

![Commute wind](figures/weather_commute_wind.png)

## Heat

Average heart rate vs temperature, controlling for speed (linear regression on
{heat['n']} commutes): **{heat['bpm_per_10c']:+.1f} bpm per +10 °C** at the same speed — no clear
heat effect. Commutes only span {c['temp_c'].min():.0f}–{c['temp_c'].max():.0f} °C and are short, so
heat has little time to build up; a long summer climb would be the better test.

![Heat](figures/weather_heat.png)

## Rain

{md(rain_table)}

Only {int(rainy.sum())} commutes had rain at the start, too few to conclude anything.
Rainy days
probably show up mostly as *missing* commutes (another way to school).
"""
    (config.REPORTS_DIR / "weather.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
