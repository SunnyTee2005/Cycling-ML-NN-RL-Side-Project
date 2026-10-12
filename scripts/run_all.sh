#!/usr/bin/env bash
# Rebuild every data product and report from the raw Strava export, in order.
# Usage: bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

steps=(
  parse_all inventory clean_all report_cleaning find_commutes traffic_lights
  commute_routes physics_report fitness_braking weather_effects find_routes recap
)
for step in "${steps[@]}"; do
  echo "==> ${step}"
  python "scripts/${step}.py" > /dev/null
done
echo "Done. Reports are in reports/."
