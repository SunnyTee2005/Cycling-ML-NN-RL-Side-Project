"""Project paths and rider/bike constants.

Values marked "assumed" are placeholders until measured.
Commute endpoints are deliberately NOT stored here: they are inferred from
the data at runtime so that no home location ends up in the public repo.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "strava"  # clone of the private strava-data repo
INTERIM_DIR = DATA_DIR / "interim"     # parsed rides, 1 Hz, one parquet per ride
PROCESSED_DIR = DATA_DIR / "processed" # cleaned rides
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- Rider -------------------------------------------------------------------
RIDER_MASS_KG = 69.0
RIDER_FTP_W = 232.0
KIT_MASS_KG = 1.5  # clothes, shoes, helmet, bottles (assumed)
# Commute backpack weight varies day to day; handled during calibration (step 8).

# --- Bikes -------------------------------------------------------------------
# Keys match the "Activity Gear" column in Strava's activities.csv.
BIKE_MASS_KG = {
    "TCR BLACK": 8.0,    # Giant TCR Advanced 1, rim brake (assumed)
    "Reacto 400": 9.5,   # Merida Reacto 400, at least 9.5 kg (rider estimate)
    "MATTS 6.40": 13.5,  # Merida Matts mountain bike (assumed)
    "Ubike": 20.0,       # YouBike public bike (assumed)
}


def system_mass_kg(bike: str) -> float:
    """Total moving mass (rider + kit + bike) for a given bike name."""
    return RIDER_MASS_KG + KIT_MASS_KG + BIKE_MASS_KG[bike]
