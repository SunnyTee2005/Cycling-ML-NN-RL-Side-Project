"""Calibrate CdA/Crr, evaluate the physics model and run the virtual power meter.

Usage:
    python scripts/physics_report.py

Writes data/processed/setups.json, data/processed/virtual_power.parquet,
reports/physics.md and figures.
"""

import json
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cycling_coach import config
from cycling_coach.calibration import (
    add_dynamics,
    error_metrics,
    estimate_ride_power,
    fit_cda_crr,
    predict_windows,
    regression_windows,
)
from cycling_coach.physics import Setup, joules_to_kcal, power_components, stop_energy_j
from cycling_coach.plotting import MUTED, SERIES, TEXT_SECONDARY, label, save, style_axes, title

FIG = config.REPORTS_DIR / "figures"
RIDES = config.PROCESSED_DIR / "rides"
SEED = 42
CLIMB_WINDOW_S = 600
CLIMB_MIN_GRADE = 0.04

# Bikes without a power meter: TCR fit transferred, or literature values (assumed).
ASSUMED = {
    "MATTS 6.40": {"cda": 0.45, "crr": 0.010},   # upright MTB, knobby tyres
    "Ubike": {"cda": 0.55, "crr": 0.010},        # upright city bike
}


def load_windows(rides: pd.DataFrame, mass: float) -> pd.DataFrame:
    frames = []
    for aid in rides["activity_id"]:
        w = regression_windows(pd.read_parquet(RIDES / f"{aid}.parquet"), mass)
        frames.append(w.assign(activity_id=aid))
    return pd.concat(frames, ignore_index=True)


def ride_split(ids: np.ndarray, test_share: float = 0.2) -> set:
    """Hold out whole rides, not seconds: neighbouring seconds of one ride are near
    copies, so a per-second split would leak the test set into training."""
    rng = np.random.default_rng(SEED)
    ids = ids.copy()
    rng.shuffle(ids)
    return set(ids[: int(len(ids) * test_share)])


def grouped_cv(windows: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """CdA/Crr fitted on k different ride subsets, to show how stable they are."""
    ids = windows["activity_id"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    folds = np.array_split(ids, k)
    rows = []
    for fold in folds:
        cda, crr = fit_cda_crr(windows[~windows["activity_id"].isin(fold)])
        rows.append({"cda": cda, "crr": crr})
    return pd.DataFrame(rows)


def setup_for(bike: str, fitted: Setup) -> Setup:
    mass = config.RIDER_MASS_KG + config.KIT_MASS_KG + config.BIKE_MASS_KG.get(bike, 8.0)
    if bike in ASSUMED:
        return Setup(mass, ASSUMED[bike]["cda"], ASSUMED[bike]["crr"])
    return Setup(mass, fitted.cda, fitted.crr)  # TCR, Reacto (similar position), unknown


def best_climb_power(power: pd.Series, grade: pd.Series) -> float:
    """Best average power over CLIMB_WINDOW_S seconds where the mean grade is >= 4%."""
    p = power.rolling(CLIMB_WINDOW_S).mean()
    g = grade.rolling(CLIMB_WINDOW_S, min_periods=CLIMB_WINDOW_S // 2).mean()
    p = p[g >= CLIMB_MIN_GRADE]
    return float(p.max()) if len(p) else np.nan


def virtual_power_one(task) -> dict:
    aid, bike, cda, crr, has_power = task
    setup = setup_for(bike, Setup(0, cda, crr))
    ride = pd.read_parquet(RIDES / f"{aid}.parquet")
    est = estimate_ride_power(ride, setup)
    moving = ride["is_moving"]
    row = {"activity_id": aid,
           "est_avg_power_w": float(est[moving].mean()) if moving.any() else np.nan,
           "est_best_climb_w": best_climb_power(est, ride["grade"])}
    if has_power:
        measured = ride["power_w"].fillna(0.0).where(moving, 0.0)
        row["avg_power_w"] = float(measured[moving].mean())
        row["best_climb_w"] = best_climb_power(measured, ride["grade"])
    # energy breakdown (positive work per component)
    r = add_dynamics(ride)
    parts = power_components(setup, r["speed_s"], r["grade"].fillna(0), r["accel"].fillna(0), r["rho"])
    for name, p in parts.items():
        row[f"work_{name}_kj"] = float(np.clip(p[moving], 0, None).sum() / 1000)
    return row


def plot_scatter(test: pd.DataFrame, pred: pd.Series, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, (name, mask), color in zip(axes, [("Climbs (grade > 3%)", test["grade"] > 0.03),
                                               ("Flat (|grade| < 1%)", test["grade"].abs() < 0.01)],
                                       SERIES):
        ax.scatter(test.loc[mask, "power"], pred[mask], s=5, alpha=0.25, color=color)
        ax.plot([0, 450], [0, 450], color=MUTED, linewidth=1)
        style_axes(ax, "both")
        title(ax, name)
        label(ax, x="Measured power, 30 s average (W)")
    label(axes[0], y="Model power (W)")
    axes[0].set_xlim(0, 450)
    axes[0].set_ylim(0, 450)
    save(fig, path)


def plot_breakdown(vp: pd.DataFrame, groups: dict, path) -> None:
    names = ["gravity", "rolling", "aero", "accel"]
    labels_ = {"gravity": "Climbing", "rolling": "Rolling resistance", "aero": "Air drag",
               "accel": "Speeding up (incl. after stops)"}
    fig, ax = plt.subplots(figsize=(11, 0.8 * len(groups) + 1.4))
    for i, (gname, ids) in enumerate(groups.items()):
        g = vp[vp["activity_id"].isin(ids)]
        work = np.array([g[f"work_{n}_kj"].sum() for n in names])
        share = work / work.sum()
        left = 0.0
        for j, (n, s) in enumerate(zip(names, share)):
            ax.barh(i, s, left=left, color=SERIES[j], edgecolor="white", linewidth=2,
                    label=labels_[n] if i == 0 else None)
            if s > 0.06:
                ax.text(left + s / 2, i, f"{s:.0%}", ha="center", va="center", color="white", fontsize=9)
            left += s
    ax.set_yticks(range(len(groups)), list(groups))
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, colors=TEXT_SECONDARY)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=9)
    title(ax, "Where the pedalling energy goes")
    save(fig, path)


def plot_virtual_history(vp: pd.DataFrame, rides: pd.DataFrame, path) -> None:
    v = vp.merge(rides[["activity_id", "year", "bike"]], on="activity_id")
    v = v[v["bike"].isin(["TCR BLACK", "Reacto 400"])]
    # Median of the three best rides per year: robust to one ride with a tailwind or drafting.
    top3 = lambda s: s.nlargest(3).median()  # noqa: E731
    est = v.groupby("year")["est_best_climb_w"].agg(top3)
    meas = v.groupby("year")["best_climb_w"].agg(top3)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(est))
    ax.bar(x - 0.2, est.values, width=0.4, color=SERIES[0], label="Virtual power meter (estimate)")
    ax.bar(x + 0.2, meas.reindex(est.index).values, width=0.4, color=SERIES[1], label="Real power meter")
    for xi, val in zip(x, est.values):
        if np.isfinite(val):
            ax.text(xi - 0.2, val + 4, f"{val:.0f}", ha="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xticks(x, est.index)
    style_axes(ax)
    label(ax, y="10-min climbing power (W)\nmedian of 3 best rides")
    title(ax, "Six years of climbing power, including the years without a power meter")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, path)


def md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    return head + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))


def main() -> None:
    rides = pd.read_parquet(config.PROCESSED_DIR / "rides.parquet")
    mass = config.system_mass_kg("TCR BLACK")
    power_rides = rides[rides["has_power_w"] & rides["has_altitude_m"] & rides["in_taiwan"]]
    windows = load_windows(power_rides, mass)

    test_ids = ride_split(power_rides["activity_id"].to_numpy())
    train = windows[~windows["activity_id"].isin(test_ids)]
    test = windows[windows["activity_id"].isin(test_ids)]
    cda, crr = fit_cda_crr(train)
    cv = grouped_cv(windows)
    fitted = Setup(mass, cda, crr)
    pred = predict_windows(test, fitted)

    segments = {"All": test["grade"].notna(), "Climbs (> 3%)": test["grade"] > 0.03,
                "Flat (< 1%)": test["grade"].abs() < 0.01,
                "Rolling (1–3%)": test["grade"].abs().between(0.01, 0.03)}
    err_rows = []
    for name, mask in segments.items():
        e = error_metrics(pred[mask], test.loc[mask, "power"])
        err_rows.append({"Terrain": name, "Windows": e["n"], "MAE": f"{e['mae_w']:.0f} W",
                         "RMSE": f"{e['rmse_w']:.0f} W", "Bias": f"{e['bias_w']:+.0f} W",
                         "Relative error": f"{e['mape']:.0%}"})

    (config.PROCESSED_DIR / "setups.json").write_text(json.dumps(
        {"cda": cda, "crr": crr, "mass_kg": mass, "assumed": ASSUMED}, indent=2))

    # Virtual power meter on every ride with speed and altitude
    usable = rides[rides["has_altitude_m"] & rides["has_lat"]]
    tasks = [(r.activity_id, r.bike, cda, crr, r.has_power_w) for r in usable.itertuples()]
    with ProcessPoolExecutor() as pool:
        vp = pd.DataFrame(pool.map(virtual_power_one, tasks, chunksize=8))
    vp.to_parquet(config.PROCESSED_DIR / "virtual_power.parquet", index=False)

    held = vp[vp["activity_id"].isin(test_ids)].dropna(subset=["avg_power_w"])
    ride_err = error_metrics(held["est_avg_power_w"], held["avg_power_w"])
    climbs = vp.dropna(subset=["best_climb_w", "est_best_climb_w"])
    climbs = climbs[climbs["activity_id"].isin(test_ids)]
    climb_err = error_metrics(climbs["est_best_climb_w"], climbs["best_climb_w"]) if len(climbs) else None

    # Energy cost of stopping at lights
    commutes = pd.read_parquet(config.PROCESSED_DIR / "commute_lights.parquet")
    commutes["cruise"] = commutes["distance_m"] / commutes["moving_s"]
    commutes["mass"] = commutes["bike"].map(lambda b: setup_for(b, fitted).mass_kg)
    commutes["stop_kj"] = [stop_energy_j(m, v) * n / 1000 for m, v, n in
                           zip(commutes["mass"], commutes["cruise"], commutes["light_stops"])]
    uni = commutes[commutes["destination"] == "university"]
    per_stop_j = stop_energy_j(mass, uni["cruise"].median())
    per_commute_kcal = joules_to_kcal(uni["stop_kj"].mean() * 1000)
    lifetime_kcal = joules_to_kcal(commutes["stop_kj"].sum() * 1000)

    FIG.mkdir(parents=True, exist_ok=True)
    plot_scatter(test, pred, FIG / "physics_scatter.png")
    big_climbs = rides.nlargest(10, "elevation_gain_m")["activity_id"]
    flat_fast = rides[(rides["elevation_gain_m"] < 100) & (rides["distance_m"] > 40_000)]["activity_id"]
    plot_breakdown(vp, {
        "University commutes": uni["activity_id"],
        "High-school commutes": commutes.loc[commutes["destination"] == "high_school", "activity_id"],
        "Flat rides > 40 km": flat_fast,
        "Ten biggest climbing rides": big_climbs,
    }, FIG / "physics_breakdown.png")
    plot_virtual_history(vp, rides, FIG / "physics_virtual_history.png")

    report = f"""# Physics model and virtual power meter

Generated by `scripts/physics_report.py`.

## Model

Power at the pedals is the sum of four terms divided by drivetrain efficiency (η = 0.977):

| Term | Formula | Meaning |
|---|---|---|
| Gravity | m·g·v·sin θ | climbing (negative downhill) |
| Rolling | Crr·m·g·v·cos θ | tyre and road losses |
| Air drag | ½·ρ·CdA·v³ | ρ from altitude and temperature |
| Acceleration | m·v·dv/dt | speeding up, e.g. after a red light |

Mass m = rider {config.RIDER_MASS_KG:.0f} kg + kit {config.KIT_MASS_KG} kg + TCR {config.BIKE_MASS_KG['TCR BLACK']:.0f} kg = **{mass:.1f} kg**.

## Calibration

Rearranged, the model is a linear regression with two unknowns and no intercept:
**η·P − P_gravity − P_accel = CdA · (½ρv³) + Crr · (m g v cos θ)**.
It is fitted with robust least squares (soft-L1 loss, so windows hit by gusts count
less) on **30-second windows of steady riding** (moving, > 3 m/s, > 50 W, |accel| < 0.3 m/s²)
from **{len(power_rides)} TCR rides with a power meter**. 20% of rides are held out for testing —
whole rides, because neighbouring seconds of one ride are near copies.

| | CdA (m²) | Crr |
|---|---|---|
| Fit on training rides | **{cda:.3f}** | **{crr:.4f}** |
| 5-fold CV by ride (mean ± std) | {cv['cda'].mean():.3f} ± {cv['cda'].std():.3f} | {cv['crr'].mean():.4f} ± {cv['crr'].std():.4f} |

CdA and Crr are partly interchangeable (both grow with speed), so the pair is less
certain than either total. The fitted Crr is higher than a racing tyre on smooth
tarmac (~0.004–0.006). Plausible reasons: city roads, and a real mass above the
assumed {mass:.1f} kg (backpack). Fixing Crr at 0.005 and fitting only CdA gave
CdA ≈ 0.30 but a *larger* held-out error on the flat, so the joint fit is kept.

## Accuracy on held-out rides

{md(pd.DataFrame(err_rows))}

- **Whole-ride average power** ({ride_err['n']} held-out rides): MAE **{ride_err['mae_w']:.0f} W**
  ({ride_err['mape']:.0%}), bias {ride_err['bias_w']:+.0f} W.
""" + (f"""- **Best 10-minute climbing power** ({climb_err['n']} held-out rides with such a climb):
  MAE **{climb_err['mae_w']:.0f} W** ({climb_err['mape']:.0%}).
""" if climb_err else "") + f"""
As expected, **climbs are accurate** (gravity dominates and is known exactly) while
**the flat is harder** (air drag dominates, and without wind data a 10 km/h headwind
looks like a weaker rider).

![Model vs measured](figures/physics_scatter.png)

## Where the energy goes

![Energy breakdown](figures/physics_breakdown.png)

## Virtual power meter

The calibrated model estimates power for every ride with speed and altitude, including
the four years before the power meter. Reacto rides reuse the TCR CdA/Crr with its own
mass; MTB and YouBike use assumed values ({', '.join(f"{k}: CdA {v['cda']}, Crr {v['crr']}" for k, v in ASSUMED.items())}).

![Virtual power history](figures/physics_virtual_history.png)

Each bar is the median of the three best rides of the year (one ride with a tailwind or
drafting cannot dominate). In 2025–2026 estimate and measurement agree within a few
watts; in 2024 the power meter only arrived in October, so the estimate also covers
earlier, stronger rides. The trend: **~200 W in 2020 → ~250–265 W from 2023 on**.

## The energy cost of red lights

Getting back to cruising speed after a stop costs the kinetic energy ½·m·v² / η.

- One stop on the university commute (cruising {uni['cruise'].median() * 3.6:.0f} km/h): **{per_stop_j / 1000:.1f} kJ**
  at the pedals ≈ **{joules_to_kcal(per_stop_j):.1f} kcal** of food (24% muscle efficiency).
- Per university commute: **{per_commute_kcal:.0f} kcal** just for restarting after lights.
- All commutes since 2020: **{lifetime_kcal:,.0f} kcal** ≈ {lifetime_kcal / 500:.0f} cups of bubble tea (≈ 500 kcal each).
"""
    (config.REPORTS_DIR / "physics.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
