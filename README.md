# AI Road-Cycling Coach

A personal cycling analytics project built on **six years of my own ride data**
(1,769 rides, 29,600 km, 2020–2026, Taichung, Taiwan).

The long-term goal: **an AI that finds me a faster daily commute.** Phase 1 builds the
foundation without machine learning: a clean data pipeline, a calibrated physics model
of riding, traffic-light and commute analytics, and a route finder on a road graph
made only from my own GPS traces.

![Red-light forecast](reports/figures/lights_forecast.png)

## Highlights

| | |
|---|---|
| **Data pipeline** | 2,152 raw files from three devices (Bryton, Garmin, a Xiaomi watch writing malformed FIT files) parsed into one 1 Hz schema; distances agree with Strava within 0.1% |
| **Traffic lights** | 65 lights found automatically from 3,738 stops; per-light chance of red, waiting time and red-phase estimate; a weekly "red-light forecast" |
| **Physics model** | CdA = 0.255 m², Crr = 0.0084 fitted on 228 power-meter rides; held-out error 12% on climbs, 23% on the flat, 5% for 10-minute climbing power |
| **Virtual power meter** | Estimates power for the four years before I owned a power meter: climbing power rose from ~200 W (2020) to ~250–265 W |
| **Wind** | Adding Strava's wind (40% of the 10 m value reaches the rider) cuts the flat error by 5%; 1 m/s of headwind costs ~48 s on my commute |
| **Route finder** | Dijkstra on a 40 m-cell graph of my own rides; suggestions save 1–3 min per commute, confirmed on held-out rides |

Every report states how it was validated, including the results that did not work
(the red-light forecast barely beats predicting the average).

## Reports

| Report | What it covers |
|---|---|
| [Data inventory](reports/inventory.md) | Rides per year, devices, fields, power-meter rides, data quirks |
| [Cleaning](reports/cleaning.md) | GPS spikes, stop and auto-pause detection, smoothed altitude and grade |
| [Traffic lights](reports/traffic_lights.md) | Light detection, worst lights, red-light forecast, best departure time, green waves |
| [Commutes](reports/commutes.md) | Route variants, leave-by calculator, out vs back, YouBike vs own bike |
| [Physics](reports/physics.md) | Power model, CdA/Crr calibration, accuracy, virtual power meter, energy cost of red lights |
| [Fitness and safety](reports/fitness_safety.md) | Fitness from commutes (watts per heartbeat), hard-braking hotspots |
| [Weather](reports/weather.md) | Wind in the physics model, headwind, heat, rain |
| [Route finder](reports/route_finder.md) | Road graph from own rides, fastest paths, honest held-out savings |
| [Recap](reports/recap.md) | Year-by-year numbers, calendar heatmap, "Wrapped" card |

## Methods worth a look

- **Calibration as linear regression.** Rearranging the power model gives
  `η·P − P_gravity − P_accel = CdA·(½ρv³) + Crr·(m g v cos θ)`: two coefficients, no
  intercept, fitted with robust least squares on 30 s windows.
- **Splitting by ride, not by second.** Neighbouring seconds of one ride are near copies;
  a per-second train/test split would leak. All evaluations hold out whole rides or a
  later time period.
- **Mean, not median, for travel times.** A light that is red 40% of the time has a
  median wait of zero but still costs time on average.
- **The optimiser's winner's curse.** The fastest of many noisy paths is biased low, so
  edge times are shrunk toward a prior and savings are re-measured on rides the search
  never saw.
- **Privacy by design.** Home and schools are inferred from the data, never configured;
  committed reports contain no coordinates; maps live in a gitignored folder.

## Roadmap

### Phase 1: Data + physics ✅
| # | Step | Status |
|---|------|--------|
| 1 | Project skeleton | ✅ |
| 2 | Parse all rides (`.fit`, `.fit.gz`, `.gpx`, `.gpx.gz`) into 1 Hz tables | ✅ |
| 3 | Data inventory report | ✅ |
| 4 | Cleaning: stops / auto-pause detection, GPS spikes, smoothed elevation and grade | ✅ |
| 5 | Commute detection | ✅ |
| 6 | Traffic-light analytics: per-intersection stats, "red-light forecast" | ✅ |
| 7 | Commute time and route comparison, "leave-by" calculator | ✅ |
| 8 | Physics model + CdA/Crr calibration ("virtual power meter") | ✅ |
| 9 | Fitness from commutes, hard-braking hotspots | ✅ |
| 10 | Weather effects (wind, heat, rain) | ✅ |
| 11 | Faster-route finder: recombine road segments I have ridden | ✅ |
| 12 | Yearly recap and final report | ✅ |

### Phase 2: Machine learning
Predict travel time on roads I have never ridden, so the route finder can suggest them.

### Phase 3: Reinforcement learning
Learn an adaptive policy at intersections: wait at the red light or turn and detour?
The road graph and traffic-light statistics from phase 1 are the simulator.

## Project structure

```
├── src/cycling_coach/   # library code (readers, cleaning, physics, traffic, routing, ...)
├── scripts/             # one script per pipeline step + run_all.sh
├── tests/               # pytest unit tests on synthetic data
├── reports/             # generated reports and figures (aggregates only)
└── data/                # gitignored: raw export, parsed and cleaned rides
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
bash scripts/run_all.sh                      # rebuild everything (~6 min)
python scripts/leave_by.py --arrive 08:10    # when to leave home to be on time
```

Individual steps, in order: `parse_all`, `inventory`, `clean_all`, `report_cleaning`,
`find_commutes`, `traffic_lights`, `commute_routes`, `physics_report`,
`fitness_braking`, `weather_effects`, `find_routes`, `recap` (all in `scripts/`).

Each parsed ride has one row per second with `timestamp` (UTC), `lat`, `lon`,
`altitude_m`, `distance_m`, `speed_mps`, `power_w`, `heart_rate_bpm`, `cadence_rpm`,
`temperature_c`, `grade_raw`, and the flags `is_filled` (no sample that second) and
`is_paused` (device auto-pause). Cleaning adds `is_moving`, `is_stopped`, `is_gap`,
`altitude_smooth_m` and `grade`.

## Privacy

GPS traces reveal where someone lives. This repo never contains raw data or
coordinates; maps are written to `reports/private/`, which is gitignored.
