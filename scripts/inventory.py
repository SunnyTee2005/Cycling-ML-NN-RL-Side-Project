"""Build the data inventory: per-ride summaries, duplicate detection and a report.

Usage:
    python scripts/inventory.py

Reads  data/interim/rides_index.parquet and data/interim/rides/*.parquet
Writes data/interim/inventory.parquet (one row per ride, gitignored)
       reports/inventory.md and reports/figures/*.png (aggregates only, no GPS)
"""

from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")  # render to files, no window
import matplotlib.pyplot as plt
import pandas as pd

from cycling_coach import config
from cycling_coach.inventory import (
    FIELDS,
    device_group,
    find_duplicates,
    is_in_taiwan,
    summarize_ride,
)

RIDES_DIR = config.INTERIM_DIR / "rides"
FIGURES_DIR = config.REPORTS_DIR / "figures"

# Validated categorical palette (fixed order) and text colours.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DEVICE_ORDER = ["Bryton Rider 420", "Garmin Forerunner 255", "Xiaomi watch", "GPX export", "Other"]


def summarize_file(activity_id: int) -> dict:
    ride = pd.read_parquet(RIDES_DIR / f"{activity_id}.parquet")
    return {"activity_id": activity_id, **summarize_ride(ride)}


def build_inventory() -> pd.DataFrame:
    index = pd.read_parquet(config.INTERIM_DIR / "rides_index.parquet")
    index = index[index["status"].str.startswith("ok")]
    with ProcessPoolExecutor() as pool:
        summaries = pd.DataFrame(pool.map(summarize_file, index["activity_id"], chunksize=16))

    inv = index.merge(summaries, on="activity_id")
    inv["device_group"] = inv["device"].map(device_group)
    inv["in_taiwan"] = is_in_taiwan(inv["median_lat"], inv["median_lon"])
    inv["duplicate_of"] = find_duplicates(inv)
    inv["year"] = inv["start_time"].dt.year
    return inv.drop(columns=["median_lat", "median_lon"])


# --- Markdown helpers ----------------------------------------------------------

def md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def pct(x: float) -> str:
    return f"{x:.0%}"


def yearly_table(rides: pd.DataFrame) -> pd.DataFrame:
    g = rides.groupby("year")
    return pd.DataFrame({
        "Year": g.size().index,
        "Rides": g.size().values,
        "Distance (km)": (g["distance_m"].sum() / 1000).round().astype(int).values,
        "Moving time (h)": (g["moving_time_s"].sum() / 3600).round().astype(int).values,
        "With heart rate": g["has_heart_rate_bpm"].sum().values,
        "With cadence": g["has_cadence_rpm"].sum().values,
        "With power meter": g["has_power_w"].sum().values,
    })


def device_table(rides: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for device in DEVICE_ORDER:
        d = rides[rides["device_group"] == device]
        if d.empty:
            continue
        row = {
            "Device": device,
            "Rides": len(d),
            "Period": f"{d['start_time'].min():%Y-%m} – {d['start_time'].max():%Y-%m}",
        }
        row.update({name: pct(d[f"has_{col}"].mean()) for col, name in FIELDS.items()})
        row["Auto-pause events"] = pct((d["paused_s"] > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def bike_table(rides: pd.DataFrame) -> pd.DataFrame:
    g = rides.fillna({"gear": "(none)"}).groupby("gear")
    out = pd.DataFrame({
        "Rides": g.size(),
        "Distance (km)": (g["distance_m"].sum() / 1000).round().astype(int),
        "With power meter": g["has_power_w"].sum(),
    }).sort_values("Rides", ascending=False)
    return out.rename_axis("Bike (Strava gear)").reset_index()


# --- Figures -----------------------------------------------------------------

def _style_axes(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_monthly_distance(rides: pd.DataFrame, path) -> None:
    month = rides["start_time"].dt.tz_localize(None).dt.to_period("M")
    km = (rides.assign(month=month).pivot_table(
        index="month", columns="device_group", values="distance_m", aggfunc="sum",
        fill_value=0) / 1000)
    full = pd.period_range(km.index.min(), km.index.max(), freq="M")
    km = km.reindex(full, fill_value=0)
    devices = [d for d in DEVICE_ORDER if d in km.columns]

    fig, ax = plt.subplots(figsize=(11, 4), dpi=150)
    bottom = pd.Series(0.0, index=km.index)
    x = km.index.to_timestamp()
    for device in devices:
        color = SERIES_COLORS[DEVICE_ORDER.index(device)]
        ax.bar(x, km[device], bottom=bottom, width=25, color=color, label=device,
               edgecolor="white", linewidth=1)
        bottom += km[device]
    _style_axes(ax)
    ax.set_ylabel("Distance per month (km)", color=TEXT_SECONDARY)
    ax.set_title("Monthly riding distance by recording device", loc="left",
                 color=TEXT_PRIMARY, fontsize=12)
    ax.legend(frameon=False, ncol=len(devices), loc="upper left", labelcolor=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_field_coverage(rides: pd.DataFrame, path) -> None:
    devices = [d for d in DEVICE_ORDER if d in set(rides["device_group"])]
    share = pd.DataFrame(
        {name: [rides.loc[rides["device_group"] == d, f"has_{col}"].mean() for d in devices]
         for col, name in FIELDS.items()},
        index=devices,
    )
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(devices) + 1.2), dpi=150)
    ax.imshow(share.values, cmap="Blues", vmin=0, vmax=1.15, aspect="auto")
    for (i, j), value in pd.DataFrame(share.values).stack().items():
        ax.text(j, i, pct(value), ha="center", va="center", fontsize=9,
                color="white" if value > 0.6 else TEXT_PRIMARY)
    ax.set_xticks(range(share.shape[1]), share.columns, color=TEXT_SECONDARY)
    ax.set_yticks(range(share.shape[0]), share.index, color=TEXT_SECONDARY)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Share of rides that recorded each field", loc="left",
                 color=TEXT_PRIMARY, fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --- Report ------------------------------------------------------------------

def render_report(inv: pd.DataFrame) -> str:
    duplicates = inv["duplicate_of"].notna()
    rides = inv[~duplicates]
    outdoor = rides[rides["has_lat"]]
    power = rides[rides["has_power_w"]]
    strava_flag_agrees = (rides["has_power_meter"] == rides["has_power_w"]).mean()

    overview = pd.DataFrame([
        ("Rides (after removing duplicates)", f"{len(rides):,}"),
        ("Date range", f"{rides['start_time'].min():%Y-%m-%d} – {rides['start_time'].max():%Y-%m-%d}"),
        ("Total distance", f"{rides['distance_m'].sum() / 1000:,.0f} km"),
        ("Total moving time", f"{rides['moving_time_s'].sum() / 3600:,.0f} h"),
        ("Rides with GPS", f"{len(outdoor):,}"),
        ("Rides without GPS (indoor / trainer)", f"{(~rides['has_lat']).sum():,}"),
        ("Rides outside Taiwan", f"{(outdoor['in_taiwan'] == False).sum():,}"),  # noqa: E712
        ("Same ride recorded on two devices (one copy kept)", f"{duplicates.sum():,}"),
        ("Rides with a power meter", f"{len(power):,}"),
    ], columns=["Item", "Value"])

    power_by_bike = power.fillna({"gear": "(none)"})["gear"].value_counts()
    lines = [
        "# Data inventory",
        "",
        "Generated by `scripts/inventory.py` from the Strava bulk export. "
        "Only aggregates are shown; no GPS coordinates or ride names.",
        "",
        "## Overview",
        "",
        md_table(overview),
        "",
        "![Monthly distance](figures/monthly_distance.png)",
        "",
        "## Rides per year",
        "",
        "A ride *has* a field when more than half of its recorded seconds contain it.",
        "",
        md_table(yearly_table(rides)),
        "",
        "## Recording devices",
        "",
        "Share of rides that recorded each field:",
        "",
        md_table(device_table(rides)),
        "",
        "![Field coverage](figures/field_coverage.png)",
        "",
        "## Bikes",
        "",
        md_table(bike_table(rides)),
        "",
        "## Power meter rides",
        "",
        f"- **{len(power)} rides** with real power data, "
        f"{power['start_time'].min():%Y-%m-%d} – {power['start_time'].max():%Y-%m-%d}.",
        "- By bike: " + ", ".join(f"{bike} {n}" for bike, n in power_by_bike.items()) + ".",
        "- Strava fills *Average Watts* with its own estimate for rides without a "
        "power meter, so the data files (not the CSV) decide which rides have power. "
        f"Strava's weighted-power column agrees with the files for {pct(strava_flag_agrees)} of rides.",
        "",
        "## Data quirks handled during parsing",
        "",
        "- **Malformed FIT files (Xiaomi watch, 2022–2023):** multi-byte fields stored "
        "as big-endian byte arrays; decoded manually.",
        "- **Placeholder values:** sensors that are absent but written as constants "
        "(distance 0, altitude −500 m, heart rate 0) are treated as missing.",
        "- **Split recordings:** one file split into several Strava activities is "
        "cropped to each activity's time window.",
        "- **Auto-pause:** FIT timer stop/start events become the `is_paused` flag, "
        "which is how stops at traffic lights will be measured.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    inv = build_inventory()
    inv.to_parquet(config.INTERIM_DIR / "inventory.parquet", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rides = inv[inv["duplicate_of"].isna()]
    plot_monthly_distance(rides, FIGURES_DIR / "monthly_distance.png")
    plot_field_coverage(rides, FIGURES_DIR / "field_coverage.png")
    (config.REPORTS_DIR / "inventory.md").write_text(render_report(inv), encoding="utf-8")
    print(f"Wrote {config.REPORTS_DIR / 'inventory.md'}")


if __name__ == "__main__":
    main()
