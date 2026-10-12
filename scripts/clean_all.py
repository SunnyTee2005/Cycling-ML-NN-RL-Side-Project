"""Clean every parsed ride and build the master ride table.

Usage:
    python scripts/clean_all.py

Reads  data/interim/inventory.parquet, data/interim/rides/*.parquet
Writes data/processed/rides/<activity_id>.parquet  cleaned 1 Hz rides
       data/processed/rides.parquet                one row per ride (master table)
       data/processed/stops.parquet                one row per stop event
"""

from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from cycling_coach import config
from cycling_coach.cleaning import clean_ride, elevation_gain_m

IN_DIR = config.INTERIM_DIR / "rides"
OUT_DIR = config.PROCESSED_DIR / "rides"


def effective_bike(row) -> str:
    """Strava's gear tag is unreliable for dual recordings; only the TCR has a power meter."""
    if row.has_power_w:
        return "TCR BLACK"
    return row.gear if isinstance(row.gear, str) else "unknown"


def clean_one(task: tuple[int, bool]) -> tuple[dict, pd.DataFrame]:
    activity_id, gps_distance = task
    ride = pd.read_parquet(IN_DIR / f"{activity_id}.parquet")
    clean, stops = clean_ride(ride, gps_distance=gps_distance)
    clean.to_parquet(OUT_DIR / f"{activity_id}.parquet", index=False)
    summary = {
        "activity_id": activity_id,
        "moving_s": int(clean["is_moving"].sum()),
        "stopped_s": int(clean["is_stopped"].sum()),
        "gap_s": int(clean["is_gap"].sum()),
        "n_stops": len(stops),
        "elevation_gain_m": elevation_gain_m(clean["altitude_smooth_m"]),
        "start_lat": clean["lat"].dropna().iloc[0] if clean["lat"].notna().any() else None,
        "start_lon": clean["lon"].dropna().iloc[0] if clean["lon"].notna().any() else None,
        "end_lat": clean["lat"].dropna().iloc[-1] if clean["lat"].notna().any() else None,
        "end_lon": clean["lon"].dropna().iloc[-1] if clean["lon"].notna().any() else None,
    }
    return summary, stops.assign(activity_id=activity_id)


def main() -> None:
    inv = pd.read_parquet(config.INTERIM_DIR / "inventory.parquet")
    rides = inv[inv["duplicate_of"].isna()].copy()
    rides["bike"] = rides.apply(effective_bike, axis=1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(r.activity_id, r.device_group == "GPX export") for r in rides.itertuples()]
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(clean_one, tasks, chunksize=8))

    summaries = pd.DataFrame([s for s, _ in results])
    stops = pd.concat([st for _, st in results if not st.empty], ignore_index=True)
    rides = rides.merge(summaries, on="activity_id")
    rides.to_parquet(config.PROCESSED_DIR / "rides.parquet", index=False)
    stops.to_parquet(config.PROCESSED_DIR / "stops.parquet", index=False)

    agree = (rides["moving_s"] / rides["moving_time_s"]).median()
    print(f"Cleaned {len(rides)} rides, {len(stops)} stop events")
    print(f"Median ratio of our moving time to Strava's: {agree:.3f}")


if __name__ == "__main__":
    main()
