"""When should I leave? Latest departure time to arrive on time.

Usage:
    python scripts/leave_by.py --arrive 08:10 [--route university] [--direction outbound]
                               [--bike own|youbike] [--days 365]

Requires data/processed/commute_routes.parquet (run scripts/commute_routes.py first).
"""

import argparse

import pandas as pd

from cycling_coach import config
from cycling_coach.routes import leave_by

BIKES = {"own": {"TCR BLACK", "Reacto 400", "MATTS 6.40"}, "youbike": {"Ubike"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arrive", required=True, help="arrival time, HH:MM")
    parser.add_argument("--route", default="university", choices=["university", "high_school"])
    parser.add_argument("--direction", default="outbound", choices=["outbound", "home"])
    parser.add_argument("--bike", default="own", choices=list(BIKES))
    parser.add_argument("--days", type=int, default=365, help="only use the last N days of rides")
    args = parser.parse_args()

    c = pd.read_parquet(config.PROCESSED_DIR / "commute_routes.parquet")
    c = c[(c["destination"] == args.route) & (c["direction"] == args.direction)
          & c["bike"].isin(BIKES[args.bike])]
    c = c[c["depart_local"] >= c["depart_local"].max() - pd.Timedelta(days=args.days)]
    if len(c) < 5:
        raise SystemExit(f"Only {len(c)} matching rides; widen --days or change --bike.")

    print(f"Based on {len(c)} rides ({args.route}, {args.direction}, {args.bike} bike):")
    for confidence in (0.5, 0.8, 0.95):
        print(f"  {confidence:>4.0%} sure to arrive by {args.arrive}: leave at "
              f"{leave_by(c['elapsed_s'], args.arrive, confidence)}")


if __name__ == "__main__":
    main()
