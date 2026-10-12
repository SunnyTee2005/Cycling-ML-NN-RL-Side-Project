"""Label commute rides (home <-> high school / university).

Usage:
    python scripts/find_commutes.py

Writes data/processed/commutes.parquet and data/processed/places.parquet
(both private: they contain locations).
"""

import pandas as pd

from cycling_coach import config
from cycling_coach.commute import label_commutes


def main() -> None:
    rides = pd.read_parquet(config.PROCESSED_DIR / "rides.parquet")
    rides = rides[rides["in_taiwan"]]
    commutes, centroids = label_commutes(rides)
    names = commutes.groupby("place")["destination"].first()
    places = pd.DataFrame(centroids, columns=["lat", "lon"]).rename_axis("place").reset_index()
    places["name"] = places["place"].map(names).fillna("")
    places.loc[0, "name"] = "home"

    commutes.to_parquet(config.PROCESSED_DIR / "commutes.parquet", index=False)
    places.to_parquet(config.PROCESSED_DIR / "places.parquet", index=False)

    summary = commutes.groupby(["destination", "direction"]).agg(
        rides=("activity_id", "size"),
        first=("depart_local", "min"),
        last=("depart_local", "max"),
        km=("distance_m", lambda s: round(s.median() / 1000, 1)),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
