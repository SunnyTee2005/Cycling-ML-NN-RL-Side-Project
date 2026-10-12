"""Commute times, route variants, out vs back, YouBike vs own bike.

Usage:
    python scripts/commute_routes.py

Writes data/processed/commute_routes.parquet, reports/commutes.md and figures.
Route maps go to reports/private/ (gitignored).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cycling_coach import config
from cycling_coach.plotting import MUTED, SERIES, TEXT_SECONDARY, label, save, style_axes, title
from cycling_coach.routes import cluster_routes, leave_by, track_cells

FIG = config.REPORTS_DIR / "figures"
PRIVATE = config.REPORTS_DIR / "private"
ROUTE_NAMES = {
    ("university", "outbound"): "Home → university",
    ("university", "home"): "University → home",
    ("high_school", "outbound"): "Home → high school",
    ("high_school", "home"): "High school → home",
}
ROAD_BIKES = {"TCR BLACK", "Reacto 400", "MATTS 6.40"}


def fmt_min(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    return head + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))


def load_tracks(commutes: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    cells, extra = [], []
    for c in commutes.itertuples():
        ride = pd.read_parquet(config.PROCESSED_DIR / "rides" / f"{c.activity_id}.parquet",
                               columns=["lat", "lon", "heart_rate_bpm", "is_moving"])
        cells.append(track_cells(ride["lat"].to_numpy(), ride["lon"].to_numpy()))
        moving = ride[ride["is_moving"]]
        extra.append({"activity_id": c.activity_id, "avg_hr": moving["heart_rate_bpm"].mean()})
    return cells, pd.DataFrame(extra)


def assign_variants(commutes: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for route in ROUTE_NAMES:
        sub = commutes[(commutes["destination"] == route[0]) & (commutes["direction"] == route[1])]
        cells, extra = load_tracks(sub)
        labels = cluster_routes(cells)
        frames.append(sub.assign(variant=labels).merge(extra, on="activity_id"))
    out = pd.concat(frames, ignore_index=True)
    out["moving_kmh"] = out["distance_m"] / out["moving_s"] * 3.6
    out["variant_name"] = out["variant"].map(lambda v: chr(ord("A") + v) if v < 26 else str(v))
    return out


def variant_table(c: pd.DataFrame, min_rides: int = 5) -> pd.DataFrame:
    rows = []
    for route, name in ROUTE_NAMES.items():
        sub = c[(c["destination"] == route[0]) & (c["direction"] == route[1])]
        for v, g in sub.groupby("variant_name"):
            if len(g) < min_rides:
                continue
            rows.append({
                "Route": name, "Variant": v, "Rides": len(g),
                "Distance (km)": f"{g['distance_m'].median() / 1000:.1f}",
                "Median time": fmt_min(g["elapsed_s"].median()),
                "Slowest 10%": fmt_min(g["elapsed_s"].quantile(0.9)),
                "Spread (std)": fmt_min(g["elapsed_s"].std()),
                "Lights (mean wait)": fmt_min(g["light_wait_s"].mean()),
                "Moving speed": f"{g['moving_kmh'].median():.1f} km/h",
            })
    return pd.DataFrame(rows)


def variant_findings(c: pd.DataFrame, min_rides: int = 10) -> str:
    """One line per route: is a less-ridden variant faster than the usual one?"""
    lines = []
    for route, name in ROUTE_NAMES.items():
        sub = c[(c["destination"] == route[0]) & (c["direction"] == route[1])]
        med = sub.groupby("variant_name")["elapsed_s"].agg(["median", "size"])
        med = med[med["size"] >= min_rides]
        if "A" not in med.index or len(med) < 2:
            continue
        best = med["median"].idxmin()
        if best == "A":
            lines.append(f"- **{name}:** the usual variant A is already the fastest.")
        else:
            gain = med.loc["A", "median"] - med.loc[best, "median"]
            lines.append(f"- **{name}:** variant {best} was {fmt_min(gain)} faster than the usual "
                         f"variant A (median), but ridden only {int(med.loc[best, 'size'])} times.")
    return "\n".join(lines)


def leave_by_table(c: pd.DataFrame, route, arrive: str) -> pd.DataFrame:
    sub = c[(c["destination"] == route[0]) & (c["direction"] == route[1])]
    rows = []
    for bikes, name in [(ROAD_BIKES, "Own bike"), ({"Ubike"}, "YouBike")]:
        g = sub[sub["bike"].isin(bikes)]
        if len(g) < 10:
            continue
        rows.append({
            "Bike": name, "Rides": len(g),
            "50% sure": leave_by(g["elapsed_s"], arrive, 0.5),
            "80% sure": leave_by(g["elapsed_s"], arrive, 0.8),
            "95% sure": leave_by(g["elapsed_s"], arrive, 0.95),
        })
    return pd.DataFrame(rows)


def out_vs_back(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    road = c[c["bike"].isin(ROAD_BIKES)]
    for dest in ["high_school", "university"]:
        for direction in ["outbound", "home"]:
            g = road[(road["destination"] == dest) & (road["direction"] == direction)]
            rows.append({
                "Route": ROUTE_NAMES[(dest, direction)], "Rides": len(g),
                "Median time": fmt_min(g["elapsed_s"].median()),
                "Distance (km)": f"{g['distance_m'].median() / 1000:.1f}",
                "Moving speed": f"{g['moving_kmh'].median():.1f} km/h",
                "Climbing (m)": f"{g['elevation_gain_m'].median():.0f}",
                "Avg heart rate": "–" if g["avg_hr"].isna().all() else f"{g['avg_hr'].median():.0f} bpm",
                "Waiting at lights": fmt_min(g["light_wait_s"].mean()),
            })
    return pd.DataFrame(rows)


def youbike_table(c: pd.DataFrame) -> pd.DataFrame:
    u = c[c["destination"] == "university"]
    rows = []
    for bikes, name in [(ROAD_BIKES, "Own road bike"), ({"Ubike"}, "YouBike")]:
        g = u[u["bike"].isin(bikes)]
        rows.append({
            "Bike": name, "Rides": len(g),
            "Median time": fmt_min(g["elapsed_s"].median()),
            "Moving speed": f"{g['moving_kmh'].median():.1f} km/h",
            "Waiting at lights": fmt_min(g["light_wait_s"].mean()),
        })
    return pd.DataFrame(rows)


def plot_trend(c: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, dest, name in zip(axes, ["high_school", "university"], ["High school", "University"]):
        for i, direction in enumerate(["outbound", "home"]):
            g = c[(c["destination"] == dest) & (c["direction"] == direction)].sort_values("depart_local")
            x = g["depart_local"].dt.tz_localize(None)
            ax.scatter(x, g["elapsed_s"] / 60, s=8, color=SERIES[i], alpha=0.35)
            trend = (g["elapsed_s"] / 60).rolling(15, center=True, min_periods=5).median()
            trend[x.diff() > pd.Timedelta(days=30)] = np.nan  # don't bridge holidays
            ax.plot(x, trend, color=SERIES[i], linewidth=2,
                    label="to school" if direction == "outbound" else "to home")
        top = c.loc[c["destination"] == dest, "elapsed_s"].quantile(0.98) / 60
        ax.set_ylim(0, top * 1.1)
        style_axes(ax)
        title(ax, f"{name} commute time")
        label(ax, y="Door-to-door (min)")
        ax.legend(frameon=False, fontsize=8)
        ax.tick_params(axis="x", labelrotation=30)
    save(fig, path)


def plot_private_variants(c: pd.DataFrame, path) -> None:
    sub = c[(c["destination"] == "university") & (c["direction"] == "outbound")]
    fig, ax = plt.subplots(figsize=(8, 8))
    for v, g in sub.groupby("variant"):
        if len(g) < 3:
            continue
        for aid in g["activity_id"].head(5):
            ride = pd.read_parquet(config.PROCESSED_DIR / "rides" / f"{aid}.parquet", columns=["lat", "lon"])
            ax.plot(ride["lon"], ride["lat"], color=SERIES[v % len(SERIES)], alpha=0.5, linewidth=1)
    title(ax, "University route variants — PRIVATE")
    save(fig, path)


def main() -> None:
    commutes = pd.read_parquet(config.PROCESSED_DIR / "commute_lights.parquet")
    c = assign_variants(commutes)
    c.to_parquet(config.PROCESSED_DIR / "commute_routes.parquet", index=False)

    FIG.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    plot_trend(c, FIG / "commute_trend.png")
    plot_private_variants(c, PRIVATE / "university_variants.png")

    recent = c[c["depart_local"] >= c["depart_local"].max() - pd.Timedelta(days=365)]
    report = f"""# Commutes

Generated by `scripts/commute_routes.py`. Times are door to door (from pressing start
to pressing stop), including traffic lights.

- **{len(c)} commutes**: {(c['destination'] == 'high_school').sum()} to/from high school
  (2020–2023, 4 km) and {(c['destination'] == 'university').sum()} to/from university
  (2025–2026, 8–9 km).

![Commute time trend](figures/commute_trend.png)

## Route variants

Rides are grouped into variants when at least 80% of each ride lies within ~80 m
(one grid cell) of the other. Variant A is the most ridden one; variants with fewer than 5 rides are hidden.

{md(variant_table(c))}

{variant_findings(c)}

## Leave-by calculator

When to leave home to reach campus by **09:00**, from the last 12 months of rides.
"95% sure" means that on 95% of past days you would have arrived in time.
Run `python scripts/leave_by.py --arrive 08:10` for any other time.

{md(leave_by_table(recent, ("university", "outbound"), "09:00"))}

## Out vs back (own bikes)

{md(out_vs_back(c))}

## YouBike vs own bike (university commute)

{md(youbike_table(c))}
"""
    (config.REPORTS_DIR / "commutes.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
