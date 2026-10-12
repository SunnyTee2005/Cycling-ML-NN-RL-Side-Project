"""Cycling power model.

Power the rider must deliver at the pedals:

    P_legs = (P_gravity + P_rolling + P_aero + P_accel) / eta

    P_gravity = m g v sin(theta)          climbing (negative downhill)
    P_rolling = Crr m g v cos(theta)      tyre deformation
    P_aero    = 0.5 rho CdA v_air^2 v     air drag (v_air = v + headwind)
    P_accel   = m v dv/dt                 change in kinetic energy

theta = atan(grade); eta is drivetrain efficiency (chain, bearings).
"""

from dataclasses import dataclass

import numpy as np

G = 9.80665          # m/s^2
ETA = 0.977          # drivetrain efficiency
R_AIR = 287.05       # J/(kg K), specific gas constant of dry air


@dataclass(frozen=True)
class Setup:
    """Rider + bike parameters."""
    mass_kg: float   # rider + kit + bike
    cda: float       # drag area, m^2
    crr: float       # rolling-resistance coefficient


def air_density(altitude_m, temperature_c=25.0):
    """Air density (kg/m^3) from the standard barometric formula."""
    pressure = 101_325.0 * (1 - 2.25577e-5 * np.asarray(altitude_m)) ** 5.25588
    return pressure / (R_AIR * (np.asarray(temperature_c) + 273.15))


def gravity_power(mass_kg, speed, grade):
    return mass_kg * G * speed * np.sin(np.arctan(grade))


def rolling_power(mass_kg, crr, speed, grade):
    return crr * mass_kg * G * speed * np.cos(np.arctan(grade))


def aero_power(cda, rho, speed, headwind=0.0):
    air_speed = speed + headwind
    return 0.5 * rho * cda * air_speed * np.abs(air_speed) * speed


def accel_power(mass_kg, speed, accel):
    return mass_kg * speed * accel


def power_components(setup: Setup, speed, grade, accel, rho, headwind=0.0) -> dict:
    """Wheel power of each term (W), before drivetrain losses."""
    return {
        "gravity": gravity_power(setup.mass_kg, speed, grade),
        "rolling": rolling_power(setup.mass_kg, setup.crr, speed, grade),
        "aero": aero_power(setup.cda, rho, speed, headwind),
        "accel": accel_power(setup.mass_kg, speed, accel),
    }


def leg_power(setup: Setup, speed, grade, accel, rho, headwind=0.0):
    """Power at the pedals (W). Negative values mean braking or coasting."""
    parts = power_components(setup, speed, grade, accel, rho, headwind)
    return sum(parts.values()) / ETA


def stop_energy_j(mass_kg: float, cruise_speed: float) -> float:
    """Energy to get back to cruising speed after a full stop (kinetic energy / eta)."""
    return 0.5 * mass_kg * cruise_speed**2 / ETA


def joules_to_kcal(joules: float, muscle_efficiency: float = 0.24) -> float:
    """Food energy burned to deliver `joules` at the pedals (~24% muscle efficiency)."""
    return joules / muscle_efficiency / 4184.0
