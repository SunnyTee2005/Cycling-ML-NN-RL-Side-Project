"""Parse every ride in the Strava export into data/interim/rides/<activity_id>.parquet.

Usage:
    python scripts/parse_all.py [--workers 8] [--limit 20]

Also writes data/interim/rides_index.parquet: one row per ride with Strava
metadata plus the parse status (ok / error message), device and row count.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from cycling_coach import config
from cycling_coach.ingest import crop_to_activity, load_activity_index, parse_activity_file

RIDES_DIR = config.INTERIM_DIR / "rides"
INDEX_PATH = config.INTERIM_DIR / "rides_index.parquet"


def parse_one(task: tuple[int, Path, Path, pd.Timestamp, float]) -> dict:
    """Parse a single ride and write its parquet. Runs in a worker process."""
    activity_id, src, dst, start, elapsed_s = task
    result = {"activity_id": activity_id, "status": "ok", "device": None, "n_seconds": 0}
    try:
        ride = parse_activity_file(src)
        data = crop_to_activity(ride.data, start, elapsed_s)
        data.to_parquet(dst, index=False)
        result.update(device=ride.device, n_seconds=len(data))
        if ride.warning:
            result["status"] = f"ok ({ride.warning})"
    except Exception as exc:  # keep going; failures are listed in the index
        result["status"] = f"error: {type(exc).__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only parse the first N rides")
    args = parser.parse_args()

    index = load_activity_index(config.RAW_DIR / "activities.csv")
    if args.limit:
        index = index.head(args.limit)
    RIDES_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        (row.activity_id, config.RAW_DIR / row.filename,
         RIDES_DIR / f"{row.activity_id}.parquet", row.start_time, row.elapsed_time_s)
        for row in index.itertuples()
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(parse_one, tasks, chunksize=8))

    status = pd.DataFrame(results)
    index = index.merge(status, on="activity_id", how="left")
    index.to_parquet(INDEX_PATH, index=False)

    errors = index[index["status"].str.startswith("error")]
    print(f"Parsed {len(index)} rides -> {RIDES_DIR}")
    print(index["status"].str.split(":").str[0].value_counts().to_string())
    if not errors.empty:
        print("\nErrors:")
        print(errors[["activity_id", "filename", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
