"""Traffic-light analytics for the commutes.

Usage:
    python scripts/traffic_lights.py

Writes data/processed/intersections.parquet, data/processed/light_passes.parquet,
data/processed/commute_lights.parquet, reports/traffic_lights.md and figures.
A map with intersection locations goes to reports/private/ (gitignored).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse

from cycling_coach import config
from cycling_coach.plotting import (
    GRID, MUTED, SERIES, TEXT_PRIMARY, TEXT_SECONDARY, label, save, style_axes, title,
)
from cycling_coach.traffic import (
    find_intersections,
    forecast_wait,
    green_wave_pairs,
    intersection_stats,
    light_stops,
    passes_for_ride,
    relevant_passes,
)

FIG = config.REPORTS_DIR / "figures"
PRIVATE = config.REPORTS_DIR / "private"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
ROUTES = {
    ("university", "outbound"): "Home → university",
    ("university", "home"): "University → home",
    ("high_school", "outbound"): "Home → high school",
    ("high_school", "home"): "High school → home",
}


def fmt_s(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


# --- Data ----------------------------------------------------------------------

def compute_passes(commutes, stops, intersections) -> pd.DataFrame:
    frames = []
    stops_by_ride = dict(tuple(stops.groupby("activity_id")))
    for c in commutes.itertuples():
        track = pd.read_parquet(config.PROCESSED_DIR / "rides" / f"{c.activity_id}.parquet",
                                columns=["timestamp", "lat", "lon"])
        p = passes_for_ride(track, stops_by_ride.get(c.activity_id, pd.DataFrame()), intersections)
        if p.empty:
            continue
        p["position"] = p["order"] / len(track)  # 0 = start of ride, 1 = end
        frames.append(p.assign(activity_id=c.activity_id, destination=c.destination,
                               direction=c.direction))
    return pd.concat(frames, ignore_index=True)


def per_commute_totals(commutes, passes) -> pd.DataFrame:
    totals = passes.groupby("activity_id").agg(
        light_wait_s=("wait_s", "sum"),
        light_stops=("wait_s", lambda w: int((w > 0).sum())),
        lights_passed=("wait_s", "size"),
    )
    out = commutes.merge(totals, on="activity_id", how="left")
    return out.fillna({"light_wait_s": 0, "light_stops": 0, "lights_passed": 0})


def evaluate_forecast(history: pd.DataFrame) -> dict:
    """Chronological split: forecast the last 20% of commutes from the earlier ones."""
    history = history.sort_values("depart_local")
    cut = int(len(history) * 0.8)
    train, test = history.iloc[:cut], history.iloc[cut:]
    pred = [forecast_wait(train, r.weekday, r.depart_hour)["wait_s"] for r in test.itertuples()]
    err = np.abs(np.array(pred) - test["light_wait_s"].to_numpy())
    naive = np.abs(train["light_wait_s"].mean() - test["light_wait_s"].to_numpy())
    return {"n_test": len(test), "mae_s": err.mean(), "naive_mae_s": naive.mean()}


# --- Figures -------------------------------------------------------------------

def typical_hour(history: pd.DataFrame) -> float:
    """Most common departure time, rounded to the half hour."""
    return (np.round(history["depart_hour"] * 2) / 2).mode().iloc[0]


def _hhmm(hour: float, _pos=None) -> str:
    return f"{int(hour):02d}:{int(round(hour % 1 * 60)):02d}"


def _icon(ax, x, y, level):
    """Weather-style icon: 0 sun (few lights), 1 sun+cloud, 2 rain cloud (many)."""
    if level == 0:
        ax.add_patch(Circle((x, y), 0.16, color="#eda100"))
    elif level == 1:
        ax.add_patch(Circle((x + 0.08, y + 0.05), 0.12, color="#eda100"))
        ax.add_patch(Ellipse((x - 0.03, y - 0.04), 0.36, 0.18, color="#b5b3ad"))
    else:
        ax.add_patch(Ellipse((x, y + 0.04), 0.38, 0.2, color="#7d7b76"))
        for dx in (-0.1, 0.0, 0.1):
            ax.plot([x + dx, x + dx - 0.04], [y - 0.1, y - 0.2], color=SERIES[0], linewidth=2)


def plot_forecast_card(history_by_route, path) -> None:
    """A weekly 'red-light weather forecast' for the current commute."""
    slots = []
    for route, name in [(("university", "outbound"), "To campus"), (("university", "home"), "Back home")]:
        hour = typical_hour(history_by_route[route])
        slots.append((route, hour, f"{name}, leaving at {int(hour):02d}:{int(hour % 1 * 60):02d}"))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 2.25)
    ax.axis("off")
    all_waits = [forecast_wait(history_by_route[r], d, h)["wait_s"]
                 for r, h, _ in slots for d in range(5)]
    lo, hi = np.percentile(all_waits, [33, 67])
    for row, (route, hour, name) in enumerate(slots):
        y = 1.55 - row * 1.05
        ax.text(0.02, y + 0.5, name, color=TEXT_SECONDARY, fontsize=10)
        for d in range(5):
            f = forecast_wait(history_by_route[route], d, hour)
            level = 0 if f["wait_s"] <= lo else (1 if f["wait_s"] <= hi else 2)
            x = d + 0.5
            ax.add_patch(plt.Rectangle((d + 0.06, y - 0.38), 0.88, 0.84, fill=False,
                                       edgecolor=GRID, linewidth=1))
            ax.text(x, y + 0.32, WEEKDAYS[d], ha="center", color=TEXT_PRIMARY, fontsize=11)
            _icon(ax, x - 0.22, y + 0.02, level)
            ax.text(x + 0.12, y + 0.06, fmt_s(f["wait_s"]), ha="left", color=TEXT_PRIMARY,
                    fontsize=14, fontweight="bold")
            ax.text(x + 0.12, y - 0.12, f"{f['stops']:.1f} stops", ha="left",
                    color=TEXT_SECONDARY, fontsize=9)
            ax.text(x, y - 0.32, f"usually {fmt_s(f['p10_s'])}–{fmt_s(f['p90_s'])}",
                    ha="center", color=MUTED, fontsize=8)
    title(ax, "Red-light forecast: expected waiting time at traffic lights "
              "(icons compare the days of this week)")
    save(fig, path)


def plot_intersections(stats, route, path) -> None:
    s = stats[(stats["destination"] == route[0]) & (stats["direction"] == route[1])]
    s = s.sort_values("position")
    labels = [f"#{i}" for i in s["intersection"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.32 * len(s) + 1.6), sharey=True)
    y = np.arange(len(s))
    ax1.barh(y, s["p_stop"] * 100, color=SERIES[0], height=0.6)
    for yi, v in zip(y, s["p_stop"]):
        ax1.text(v * 100 + 1, yi, f"{v:.0%}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax1.set_yticks(y, labels)
    ax1.invert_yaxis()
    style_axes(ax1, "x")
    label(ax1, x="Chance of stopping (%)")
    title(ax1, f"{ROUTES[route]}: lights in riding order")
    ax2.barh(y, s["median_wait_s"], color=SERIES[1], height=0.6, label="Median wait")
    ax2.scatter(s["red_phase_estimate_s"], y, color=TEXT_PRIMARY, s=14, zorder=3,
                label="Longest typical wait (≈ red phase)")
    style_axes(ax2, "x")
    label(ax2, x="Seconds when stopped")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, path)


def plot_departure_time(history_by_route, path) -> None:
    routes = [("university", "outbound"), ("high_school", "outbound")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, route, color in zip(axes, routes, SERIES):
        h = history_by_route[route]
        h = h[(h["depart_hour"] >= 5) & (h["depart_hour"] < 22)]
        bins = np.floor(h["depart_hour"] * 2) / 2
        g = h.groupby(bins)["light_wait_s"].agg(["mean", "size"])
        g = g[g["size"] >= 5]
        ax.bar(g.index, g["mean"] / 60, width=0.42, color=color, align="edge")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_hhmm))
        style_axes(ax)
        label(ax, x="Departure time (local, 30-min bins with ≥ 5 rides)")
        title(ax, ROUTES[route])
    label(axes[0], y="Avg. waiting at lights (min)")
    save(fig, path)


def plot_private_map(intersections, stats, path) -> None:
    exp = stats.groupby("intersection")["expected_wait_s"].max()
    ix = intersections.set_index("intersection").join(exp).dropna()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ix["lon"], ix["lat"], s=ix["expected_wait_s"] * 4, color=SERIES[1], alpha=0.7)
    for i, r in ix.iterrows():
        ax.annotate(f"#{i}", (r["lon"], r["lat"]), fontsize=7, color=TEXT_PRIMARY)
    ax.set_aspect(1 / np.cos(np.radians(ix["lat"].mean())))
    title(ax, "Intersections (size = expected wait) — PRIVATE")
    save(fig, path)


# --- Report --------------------------------------------------------------------

def main() -> None:
    commutes = pd.read_parquet(config.PROCESSED_DIR / "commutes.parquet")
    commutes = commutes[commutes["destination"].isin(["high_school", "university"])]
    places = pd.read_parquet(config.PROCESSED_DIR / "places.parquet")
    stops = light_stops(pd.read_parquet(config.PROCESSED_DIR / "stops.parquet"), commutes, places)

    intersections = find_intersections(stops)
    passes = relevant_passes(compute_passes(commutes, stops, intersections))
    totals = per_commute_totals(commutes, passes)
    stats = intersection_stats(passes)
    position = passes.groupby(["intersection", "destination", "direction"])["position"].median()
    stats = stats.merge(position.reset_index(), on=["intersection", "destination", "direction"])
    waves = green_wave_pairs(passes)

    intersections.to_parquet(config.PROCESSED_DIR / "intersections.parquet", index=False)
    passes.to_parquet(config.PROCESSED_DIR / "light_passes.parquet", index=False)
    totals.to_parquet(config.PROCESSED_DIR / "commute_lights.parquet", index=False)
    stats.to_parquet(config.PROCESSED_DIR / "intersection_stats.parquet", index=False)

    history = {r: totals[(totals["destination"] == r[0]) & (totals["direction"] == r[1])]
               for r in ROUTES}
    FIG.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    plot_forecast_card(history, FIG / "lights_forecast.png")
    plot_intersections(stats, ("university", "outbound"), FIG / "lights_university_outbound.png")
    plot_intersections(stats, ("high_school", "outbound"), FIG / "lights_high_school_outbound.png")
    plot_departure_time(history, FIG / "lights_departure_time.png")
    plot_private_map(intersections, stats, PRIVATE / "intersections_map.png")

    # Route summary table
    rows = []
    for route, name in ROUTES.items():
        h = history[route]
        ev = evaluate_forecast(h)
        rows.append({
            "Route": name, "Commutes": len(h),
            "Lights passed (median)": int(h["lights_passed"].median()),
            "Stops (mean)": f"{h['light_stops'].mean():.1f}",
            "Waiting (mean)": fmt_s(h["light_wait_s"].mean()),
            "Share of ride time": f"{(h['light_wait_s'] / h['elapsed_s']).mean():.0%}",
            "Forecast error (MAE)": fmt_s(ev["mae_s"]),
            "Naive error": fmt_s(ev["naive_mae_s"]),
        })
    route_table = pd.DataFrame(rows)

    worst = stats.sort_values("expected_wait_s", ascending=False).head(8)
    worst_table = pd.DataFrame({
        "Light": "#" + worst["intersection"].astype(str),
        "Route": [ROUTES[(d, r)] for d, r in zip(worst["destination"], worst["direction"])],
        "Passes": worst["passes"],
        "Chance of red": worst["p_stop"].map("{:.0%}".format),
        "Median wait": worst["median_wait_s"].map(fmt_s),
        "≈ Red phase": worst["red_phase_estimate_s"].map(fmt_s),
        "Expected cost per pass": worst["expected_wait_s"].map(fmt_s),
    })

    linked = waves.assign(gap=waves["p_stop_after_red"] - waves["p_stop_after_green"])
    linked = linked.sort_values("gap", ascending=False).head(6)
    wave_table = pd.DataFrame({
        "Pair": "#" + linked["from"].astype(str) + " → #" + linked["to"].astype(str),
        "Route": [ROUTES[(d, r)] for d, r in zip(linked["destination"], linked["direction"])],
        "Rides": linked["n"],
        "Red at next if green here": linked["p_stop_after_green"].map("{:.0%}".format),
        "Red at next if red here": linked["p_stop_after_red"].map("{:.0%}".format),
        "Travel time if next is green": linked["travel_s_when_next_green"].map(
            lambda v: "–" if pd.isna(v) else f"{v:.0f} s"),
        "… if next is red": linked["travel_s_when_next_red"].map(
            lambda v: "–" if pd.isna(v) else f"{v:.0f} s"),
    })

    all_stops = pd.read_parquet(config.PROCESSED_DIR / "stops.parquet")
    lifetime_light_s = totals["light_wait_s"].sum()
    all_short_s = all_stops.loc[all_stops["duration_s"] <= MAX_SHORT_STOP_S, "duration_s"].sum()

    def md(df):
        head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
        return head + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))

    report = f"""# Traffic lights

Generated by `scripts/traffic_lights.py`.

**How it works.** Stops on commutes (excluding 200 m around home/school and stops
over 3 minutes) are clustered within 30 m; every cluster with ≥ 8 stops is treated as a
traffic light. For each commute we then check which lights it passed (track within
35 m) and whether it stopped there. Stop chance uses add-one smoothing, and the
95th-percentile wait estimates the red phase (arriving just as it turns red).
Lights are shown by ID only; their locations stay private.

- **{len(intersections)} traffic lights** found from **{len(stops):,} stops** on **{len(commutes)} commutes**.
- Lifetime ledger: **{lifetime_light_s / 3600:.1f} h** waiting at commute lights, and
  **{all_short_s / 3600:.0f} h** in all short stops (≤ 3 min) across every ride since 2020.

## Per route

{md(route_table)}

*Forecast error*: forecast the last 20% of commutes (chronologically) from the earlier
ones, using same-weekday rides within ±1 h of the departure time, shrunk toward all
rides in that time window. *Naive*: always predict the overall average.

## Red-light forecast

![Forecast](figures/lights_forecast.png)

**Honest finding:** the forecast is barely better (sometimes worse) than always
guessing the average. Which lights turn red for you is close to random; weekday and
departure time explain little of the day-to-day spread. What *is* predictable is the
typical cost of a route and its range ("usually 2:41–6:04"), which is what the card
shows. Where departure time does matter, see *Best time to leave* below.

## The worst lights

{md(worst_table)}

![University lights](figures/lights_university_outbound.png)

![High-school lights](figures/lights_high_school_outbound.png)

## Best time to leave

![Departure time](figures/lights_departure_time.png)

## Green waves: are neighbouring lights linked?

If light B is coordinated with light A, catching green at A should make green at B
more likely. The table lists the pairs where the outcome at A changes the chance of
red at B the most; the travel-time columns show whether riding faster between them
helps.

{md(wave_table)}

Pairs only a few seconds apart with "red here → red next" close to 100% are most
likely a **two-stage left turn** (wait for one light, roll to the waiting box, wait
for the crossing light, whose phase is opposite). Pairs where the travel time when
the next light is green is clearly shorter than when it is red suggest that
**riding a little faster between them catches the green**.
"""
    (config.REPORTS_DIR / "traffic_lights.md").write_text(report, encoding="utf-8")
    print(report)


MAX_SHORT_STOP_S = 180

if __name__ == "__main__":
    main()
