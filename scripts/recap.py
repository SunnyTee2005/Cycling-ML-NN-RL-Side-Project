"""Yearly recap ("Wrapped") and calendar heatmap.

Usage:
    python scripts/recap.py

Writes reports/recap.md and figures.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from cycling_coach import config
from cycling_coach.plotting import GRID, TEXT_PRIMARY, TEXT_SECONDARY, save, title

FIG = config.REPORTS_DIR / "figures"
TZ = "Asia/Taipei"
AROUND_TAIWAN_KM = 1000          # a classic round-the-island ride is roughly this long
SCOOTER_L_PER_100KM = 2.5        # typical 125 cc scooter
FUEL_NTD_PER_L = 30.0
CO2_KG_PER_L = 2.3
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#6da7ec", "#2a78d6", "#104281"])


def fmt_min(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    return head + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))


def yearly_recap(rides: pd.DataFrame, commutes: pd.DataFrame, vp: pd.DataFrame) -> pd.DataFrame:
    rides = rides.merge(vp[["activity_id", "est_best_climb_w"]], on="activity_id", how="left")
    commutes = commutes.assign(local_date=commutes["depart_local"].dt.date)
    rows = []
    for year, r in rides.groupby("year"):
        c = commutes[commutes["year"] == year]
        km = r["distance_m"].sum() / 1000
        commute_km = c["distance_m"].sum() / 1000
        fuel_l = commute_km * SCOOTER_L_PER_100KM / 100
        rows.append({
            "Year": year,
            "Rides": len(r),
            "Distance": f"{km:,.0f} km",
            "≈ Round-Taiwan trips": f"{km / AROUND_TAIWAN_KM:.1f}",
            "Climbing": f"{r['elevation_gain_m'].sum():,.0f} m",
            "Riding days": r["start_time"].dt.tz_convert(TZ).dt.date.nunique(),
            "Commute days": c["local_date"].nunique(),
            "Waiting at lights": f"{c['light_wait_s'].sum() / 3600:.1f} h",
            "Earliest departure": "–" if c.empty else c["depart_local"].dt.strftime("%H:%M").min(),
            "Fuel saved vs scooter": f"NT${fuel_l * FUEL_NTD_PER_L:,.0f} / {fuel_l * CO2_KG_PER_L:.0f} kg CO₂",
        })
    return pd.DataFrame(rows)


def plot_calendar(rides: pd.DataFrame, years: list[int], path) -> None:
    """GitHub-style grid: one row per year, one cell per day, colour = km ridden."""
    local = rides.assign(day=rides["start_time"].dt.tz_convert(TZ).dt.normalize().dt.tz_localize(None))
    daily = local.groupby("day")["distance_m"].sum() / 1000
    fig, axes = plt.subplots(len(years), 1, figsize=(12, 1.55 * len(years)))
    for ax, year in zip(np.atleast_1d(axes), years):
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        km = daily.reindex(days, fill_value=0.0)
        week = ((days.dayofyear - 1 + days[0].dayofweek) // 7).to_numpy()
        grid = np.full((7, week.max() + 1), np.nan)
        grid[days.dayofweek, week] = km.to_numpy()
        ax.imshow(np.where(grid > 0, 1, 0), cmap=LinearSegmentedColormap.from_list("bg", ["#efeeea", "#efeeea"]),
                  aspect="equal")
        ax.imshow(np.where(grid > 0, np.log1p(np.minimum(grid, 100)), np.nan), cmap=BLUES,
                  vmin=np.log1p(1), vmax=np.log1p(100), aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([0, 3, 6], ["Mon", "Thu", "Sun"], fontsize=7, color=TEXT_SECONDARY)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(str(year), rotation=0, labelpad=22, va="center", color=TEXT_PRIMARY)
        active = int((km > 0).sum())
        ax.text(grid.shape[1] + 0.8, 3, f"{active} days\n{km.sum():,.0f} km", va="center",
                fontsize=8, color=TEXT_SECONDARY)
    title(np.atleast_1d(axes)[0], "Every day on the bike (darker = longer)")
    save(fig, path)


def plot_wrapped(year: int, rides: pd.DataFrame, commutes: pd.DataFrame, path) -> None:
    """A 'Wrapped'-style card of headline numbers for one year."""
    r, c = rides[rides["year"] == year], commutes[commutes["year"] == year]
    km = r["distance_m"].sum() / 1000
    tiles = [
        (f"{km:,.0f} km", f"≈ {km / AROUND_TAIWAN_KM:.1f}× around Taiwan"),
        (f"{len(r)}", "rides"),
        (f"{r['elevation_gain_m'].sum() / 1000:,.1f} km", "climbed"),
        (f"{c['depart_local'].dt.date.nunique()}", "commute days"),
        (f"{c['light_wait_s'].sum() / 3600:.1f} h", "waiting at red lights"),
        (f"{int(c['light_stops'].sum()):,}", "red-light stops on commutes"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, (big, small) in enumerate(tiles):
        x, y = i % 3, 1 - i // 3
        ax.add_patch(plt.Rectangle((x + 0.05, y + 0.08), 0.9, 0.84, fill=False, edgecolor=GRID))
        ax.text(x + 0.5, y + 0.55, big, ha="center", va="center", fontsize=24,
                fontweight="bold", color="#2a78d6")
        ax.text(x + 0.5, y + 0.25, small, ha="center", va="center", fontsize=11, color=TEXT_SECONDARY)
    title(ax, f"My {year} on the bike")
    save(fig, path)


def main() -> None:
    rides = pd.read_parquet(config.PROCESSED_DIR / "rides.parquet")
    commutes = pd.read_parquet(config.PROCESSED_DIR / "commute_lights.parquet")
    vp = pd.read_parquet(config.PROCESSED_DIR / "virtual_power.parquet")

    recap = yearly_recap(rides, commutes, vp)
    full_years = sorted(y for y in rides["year"].unique() if 2021 <= y <= 2025)
    FIG.mkdir(parents=True, exist_ok=True)
    plot_calendar(rides, full_years, FIG / "recap_calendar.png")
    plot_wrapped(2025, rides, commutes, FIG / "recap_2025.png")

    total_km = rides["distance_m"].sum() / 1000
    commute_km = commutes["distance_m"].sum() / 1000
    report = f"""# Recap

Generated by `scripts/recap.py`.

**{len(rides):,} rides and {total_km:,.0f} km since May 2020** — about
{total_km / AROUND_TAIWAN_KM:.0f} times around Taiwan. Of that, {commute_km:,.0f} km were
commutes to high school and university.

![2025](figures/recap_2025.png)

![Calendar](figures/recap_calendar.png)

## Year by year

{md(recap)}

Fuel and CO₂ compare commute kilometres with a typical 125 cc scooter
({SCOOTER_L_PER_100KM} L/100 km, NT${FUEL_NTD_PER_L:.0f}/L, {CO2_KG_PER_L} kg CO₂/L). 2020 and 2026 are partial years.
"""
    (config.REPORTS_DIR / "recap.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
