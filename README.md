# AI Road-Cycling Coach

A personal cycling analytics project built on **six years of my own ride data**
(~1,800 rides, ~30,000 km, 2020–2026, Taichung, Taiwan).

The long-term goal: **an AI that finds me a faster daily commute.**
Along the way it builds the pieces a real coach would need: clean data,
a physics model of riding, commute and traffic-light analytics, and later
machine learning and reinforcement learning.

## Roadmap

### Phase 1: Data + physics (current)
| # | Step | Status |
|---|------|--------|
| 1 | Project skeleton | ✅ |
| 2 | Parse all rides (`.fit`, `.fit.gz`, `.gpx`, `.gpx.gz`) into 1 Hz tables | ✅ |
| 3 | Data inventory report ([reports/inventory.md](reports/inventory.md)) | ✅ |
| 4 | Cleaning: stops / auto-pause detection, GPS spikes, smoothed elevation and grade ([report](reports/cleaning.md)) | ✅ |
| 5 | Commute detection | ✅ |
| 6 | Traffic-light analytics: per-intersection stats, "red-light forecast" ([report](reports/traffic_lights.md)) | ✅ |
| 7 | Commute time and route comparison, "leave-by" calculator ([report](reports/commutes.md)) | ✅ |
| 8 | Physics model + CdA/Crr calibration ("virtual power meter") ([report](reports/physics.md)) | ✅ |
| 9 | Fitness from commutes, hard-braking hotspots | ⬜ |
| 10 | Weather effects (wind, heat, rain) | ⬜ |
| 11 | Faster-route finder: recombine road segments I have ridden | ⬜ |
| 12 | Yearly recap and final report | ⬜ |

### Phase 2: Machine learning
Predict travel time on roads I have never ridden.

### Phase 3: Reinforcement learning
Learn an adaptive policy at intersections: wait at the red light or turn and detour?

## Project structure

```
├── src/cycling_coach/   # reusable library code
├── scripts/             # thin command-line entry points
├── tests/               # pytest unit tests
├── reports/             # generated reports (no raw GPS)
└── data/                # gitignored: raw and processed ride data
```

## Setup (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Data

The raw data is a Strava bulk export kept in a **separate private repository**.
Clone it into `data/raw/strava` (the whole `data/` folder is gitignored):

```bash
git clone https://github.com/SunnyTee2005/strava-data.git data/raw/strava
```

Only `activities/`, `activities.csv` and `bikes.csv` are read.

## Usage

```bash
python scripts/parse_all.py   # raw files -> data/interim/rides/<activity_id>.parquet
python scripts/inventory.py   # per-ride summary + reports/inventory.md
python scripts/clean_all.py   # cleaned rides, master ride table, stop events
python scripts/report_cleaning.py
python scripts/find_commutes.py   # home <-> high school / university
python scripts/traffic_lights.py
python scripts/commute_routes.py
python scripts/leave_by.py --arrive 08:10   # when to leave home
python scripts/physics_report.py
```

Each parsed ride has one row per second with: `timestamp` (UTC), `lat`, `lon`,
`altitude_m`, `distance_m`, `speed_mps`, `power_w`, `heart_rate_bpm`,
`cadence_rpm`, `temperature_c`, `grade_raw`, plus two flags:
`is_filled` (no sample recorded that second) and `is_paused` (device auto-pause).

## Privacy

GPS traces reveal where someone lives. This repo never contains raw data, and
any published map masks the area around ride start and end points.
