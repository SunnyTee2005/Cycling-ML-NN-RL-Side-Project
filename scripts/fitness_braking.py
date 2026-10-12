"""Fitness trend from commutes and hard-braking hotspots.

Usage:
    python scripts/fitness_braking.py

Writes data/processed/commute_fitness.parquet, data/processed/brakes.parquet,
reports/fitness_safety.md and figures (hotspot map in reports/private/).
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cycling_coach import config
from cycling_coach.braking import hard_brakes
from cycling_coach.calibration import estimate_ride_power, setup_for_bike
from cycling_coach.commute import cluster_points
from cycling_coach.fitness import commute_efficiency
from cycling_coach.geo import haversine_m
from cycling_coach.plotting import SERIES, TEXT_PRIMARY, label, save, style_axes, title
from cycling_coach.traffic import passes_for_ride

FIG = config.REPORTS_DIR / "figures"
PRIVATE = config.REPORTS_DIR / "private"
HOTSPOT_RADIUS_M = 30.0
MIN_HOTSPOT_EVENTS = 4
AT_LIGHT_M = 40.0


def md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    return head + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))


def analyse_commutes(commutes: pd.DataFrame, cda: float, crr: float):
    fitness, brakes = [], []
    for c in commutes.itertuples():
        ride = pd.read_parquet(config.PROCESSED_DIR / "rides" / f"{c.activity_id}.parquet")
        est = None
        if ride["altitude_smooth_m"].notna().any():
            est = estimate_ride_power(ride, setup_for_bike(c.bike, cda, crr))
        fitness.append({"activity_id": c.activity_id, **commute_efficiency(ride, est)})
        b = hard_brakes(ride)
        if not b.empty:
            brakes.append(b.assign(activity_id=c.activity_id))
    return pd.DataFrame(fitness), pd.concat(brakes, ignore_index=True)


def find_hotspots(brakes: pd.DataFrame, commutes: pd.DataFrame, lights: pd.DataFrame) -> pd.DataFrame:
    labels, centroids = cluster_points(brakes["lat"].to_numpy(), brakes["lon"].to_numpy(),
                                       radius_m=HOTSPOT_RADIUS_M, min_size=MIN_HOTSPOT_EVENTS)
    spots = pd.DataFrame(centroids, columns=["lat", "lon"])
    spots["intersection"] = np.arange(len(spots))   # column name expected by passes_for_ride
    spots["events"] = np.bincount(labels[labels >= 0], minlength=len(spots))
    # How often each hotspot was ridden through, to turn counts into a rate.
    passes = np.zeros(len(spots))
    empty = pd.DataFrame(columns=["lat", "lon", "duration_s"])
    for aid in commutes["activity_id"]:
        track = pd.read_parquet(config.PROCESSED_DIR / "rides" / f"{aid}.parquet",
                                columns=["timestamp", "lat", "lon"])
        p = passes_for_ride(track, empty, spots)
        passes[p["intersection"].astype(int).to_numpy()] += 1
    spots["passes"] = passes
    spots["rate"] = spots["events"] / spots["passes"].clip(lower=1)
    d = haversine_m(spots["lat"].to_numpy()[:, None], spots["lon"].to_numpy()[:, None],
                    lights["lat"].to_numpy()[None, :], lights["lon"].to_numpy()[None, :])
    spots["at_light"] = d.min(axis=1) < AT_LIGHT_M
    return spots


def fitness_finding(yearly: pd.DataFrame, min_rides: int = 15) -> str:
    """Change in watts per bpm between the first and last well-covered year of a route."""
    hs = yearly[(yearly["destination"] == "high_school") & (yearly["rides"] >= min_rides)]
    hs = hs.dropna(subset=["wpb"])
    if len(hs) < 2:
        return ""
    first, last = hs.iloc[0], hs.iloc[-1]
    change = last["wpb"] / first["wpb"] - 1
    return (f"On the same high-school route, estimated watts per bpm went from "
            f"**{first['wpb']:.2f} ({first['year']}) to {last['wpb']:.2f} ({last['year']})**, "
            f"{change:+.0%}: the same effort at a lower heart rate. "
            f"(2021 has only {int(first['rides'])} rides with heart rate, so treat it as indicative.)")


def plot_fitness(c: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, column, ylabel in zip(axes, ["metres_per_beat", "watts_per_bpm"],
                                  ["Metres per heartbeat", "Estimated watts per bpm"]):
        for i, dest in enumerate(["high_school", "university"]):
            g = c[(c["destination"] == dest) & c[column].notna()]
            if g.empty:
                continue
            month = g["depart_local"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
            m = g.groupby(month)[column].agg(["median", "size"])
            m = m[m["size"] >= 3]
            gap = m.index.to_series().diff() > pd.Timedelta(days=62)
            m.loc[gap, "median"] = np.nan  # don't draw across holidays
            ax.scatter(g["depart_local"].dt.tz_localize(None), g[column], s=6, alpha=0.2, color=SERIES[i])
            ax.plot(m.index, m["median"], marker="o", markersize=4, linewidth=2, color=SERIES[i],
                    label=f"{dest.replace('_', ' ')} commute (monthly median)")
        style_axes(ax)
        label(ax, y=ylabel)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    title(axes[0], "Fitness from commutes: more road (or power) per heartbeat = fitter")
    save(fig, path)


def plot_brake_rate(c: pd.DataFrame, brakes: pd.DataFrame, path) -> None:
    per = brakes.groupby("activity_id").size().rename("brakes")
    c = c.merge(per, on="activity_id", how="left").fillna({"brakes": 0})
    g = c.groupby(["year", "destination"]).agg(brakes=("brakes", "sum"), km=("distance_m", "sum"))
    g["rate"] = g["brakes"] / (g["km"] / 100_000)
    g = g.reset_index()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, dest in enumerate(["high_school", "university"]):
        d = g[g["destination"] == dest]
        ax.bar(d["year"] + (i - 0.5) * 0.4, d["rate"], width=0.4, color=SERIES[i],
               label=dest.replace("_", " "))
    style_axes(ax)
    label(ax, y="Hard brakes per 100 km")
    title(ax, "Hard braking on commutes")
    ax.legend(frameon=False, fontsize=9)
    save(fig, path)


def main() -> None:
    commutes = pd.read_parquet(config.PROCESSED_DIR / "commute_routes.parquet")
    setups = json.loads((config.PROCESSED_DIR / "setups.json").read_text())
    fitness, brakes = analyse_commutes(commutes, setups["cda"], setups["crr"])
    c = commutes.drop(columns=["avg_hr"]).merge(fitness, on="activity_id")
    c.to_parquet(config.PROCESSED_DIR / "commute_fitness.parquet", index=False)
    own = c[c["bike"] != "Ubike"]  # a heavy upright YouBike is not comparable
    brakes.to_parquet(config.PROCESSED_DIR / "brakes.parquet", index=False)

    lights = pd.read_parquet(config.PROCESSED_DIR / "intersections.parquet")
    spots = find_hotspots(brakes, commutes, lights)

    FIG.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    plot_fitness(own, FIG / "fitness_commutes.png")
    plot_brake_rate(c, brakes, FIG / "braking_rate.png")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(spots["lon"], spots["lat"], s=spots["events"] * 15,
               c=np.where(spots["at_light"], SERIES[0], SERIES[7]))
    for i, r in spots.iterrows():
        ax.annotate(f"H{i}", (r["lon"], r["lat"]), fontsize=7, color=TEXT_PRIMARY)
    title(ax, "Hard-braking hotspots (red = not at a traffic light) — PRIVATE")
    save(fig, PRIVATE / "braking_hotspots.png")

    # Fitness summary per year (same route, so comparable within a route)
    f = own.dropna(subset=["metres_per_beat"])
    yearly = f.groupby(["destination", "year"]).agg(
        rides=("activity_id", "size"),
        mpb=("metres_per_beat", "median"),
        wpb=("watts_per_bpm", "median"),
        hr=("avg_hr", "median"),
        kmh=("moving_kmh", "median"),
    ).reset_index()
    fit_table = pd.DataFrame({
        "Route": yearly["destination"].str.replace("_", " "),
        "Year": yearly["year"], "Rides with HR": yearly["rides"],
        "Moving speed": yearly["kmh"].map("{:.1f} km/h".format),
        "Avg HR": yearly["hr"].map("{:.0f}".format),
        "Metres / beat": yearly["mpb"].map("{:.2f}".format),
        "Watts / bpm (est.)": yearly["wpb"].map(lambda v: "–" if pd.isna(v) else f"{v:.2f}"),
    })

    top = spots.sort_values("events", ascending=False).head(10)
    spot_table = pd.DataFrame({
        "Hotspot": "H" + top.index.astype(str),
        "Hard brakes": top["events"],
        "Rides through": top["passes"].astype(int),
        "Per 100 rides": (top["rate"] * 100).map("{:.0f}".format),
        "At a traffic light": np.where(top["at_light"], "yes", "**no**"),
    })
    mid_block = spots[~spots["at_light"]]
    total_km = commutes["distance_m"].sum() / 1000

    report = f"""# Fitness and safety from commutes

Generated by `scripts/fitness_braking.py`.

## Fitness: the commute as a daily test

Riding the same route hundreds of times is a repeated, free fitness test. Two measures
over the moving seconds of each commute with heart rate:

- **Metres per heartbeat**: road covered per beat. Simple and works without altitude.
- **Watts per bpm** (efficiency factor): estimated power from the physics model divided
  by heart rate. Corrects for wind-free terrain differences, needs altitude.

Only own-bike commutes are used (a heavy upright YouBike is not comparable), and
values should be compared within one route (the routes differ in hills and traffic).

{md(fit_table)}

{fitness_finding(yearly)}

![Fitness](figures/fitness_commutes.png)

## Hard braking

A hard brake is losing ≥ 4 m/s (14 km/h) within 2 s from above 15 km/h, about 0.2 g;
larger drops are GPS artefacts. Events within 10 s count once.

- **{len(brakes)} hard brakes** on {len(commutes)} commutes ({total_km:,.0f} km):
  **{len(brakes) / total_km * 100:.1f} per 100 km**.
- **{len(spots)} hotspots** (≥ {MIN_HOTSPOT_EVENTS} hard brakes within 30 m);
  **{len(mid_block)} of them are not at a traffic light** — places where something
  unexpected (cars turning or pulling out, pedestrians, scooters) makes you brake.

{md(spot_table)}

![Braking rate](figures/braking_rate.png)
"""
    (config.REPORTS_DIR / "fitness_safety.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
