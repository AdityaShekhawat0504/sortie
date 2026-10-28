"""Fault injection on real flight logs.

Real logs with labelled faults are rare, so Sortie measures its own detection
quality by injecting known faults into real (healthy) flights and checking
whether the right check fires, how fast, and what else fires by mistake.

Every injector works on canonical signals, so it exercises exactly the code
path a real fault would. It returns the modified copy plus the ground truth.

Two "control" injectors change the data in ways that look dramatic but are
healthy (an aggressive manoeuvre the vehicle tracks, a load step that the
battery handles). They must *not* trigger anything; they measure specificity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .ingest import Flight


@dataclass
class Injection:
    fault: str
    expected_check: str | None  # None for control cases
    t_start: float
    t_end: float
    params: dict


def _mask(df: pd.DataFrame, t0: float, t1: float) -> np.ndarray:
    idx = df.index.to_numpy()
    return (idx >= t0) & (idx <= t1)


def _ramp(df: pd.DataFrame, t0: float, t1: float, rise: float = 0.3) -> np.ndarray:
    """0..1 envelope: ramps up over ``rise`` s, holds, ramps down."""
    t = df.index.to_numpy()
    up = np.clip((t - t0) / rise, 0, 1)
    down = np.clip((t1 - t) / rise, 0, 1)
    return np.where((t >= t0) & (t <= t1), np.minimum(up, down), 0.0)


# --------------------------------------------------------------------------- #
# faults


def gps_jump(f: Flight, t0: float, t1: float, rng, mag: float | None = None) -> dict:
    g = f.signals["gps"]
    meters = float(rng.uniform(60, 250)) if mag is None else mag
    bearing = float(rng.uniform(0, 2 * np.pi))
    m = _mask(g, t0, t1)
    dlat = meters * np.cos(bearing) / 111_320.0
    dlon = meters * np.sin(bearing) / (111_320.0 * np.cos(np.radians(g["lat"].mean())))
    g.loc[m, "lat"] += dlat
    g.loc[m, "lon"] += dlon
    return {"jump_m": round(meters, 1)}


def gps_degradation(f: Flight, t0: float, t1: float, rng, mag: float | None = None) -> dict:
    g = f.signals["gps"]
    m = _mask(g, t0, t1)
    eph = float(rng.uniform(5, 15)) if mag is None else mag
    g.loc[m, "eph"] = eph
    if "sats" in g and mag is None:
        g.loc[m, "sats"] = float(rng.integers(3, 6))
    return {"eph_m": round(eph, 1)}


def vibration_burst(f: Flight, t0: float, t1: float, rng, mag: float | None = None) -> dict:
    a = f.signals["accel"]
    m = _mask(a, t0, t1)
    sigma = float(rng.uniform(6, 14)) if mag is None else mag
    noise = rng.normal(0, sigma / np.sqrt(3), size=(int(m.sum()), 3))
    a.loc[m, ["ax", "ay", "az"]] += noise
    return {"sigma_ms2": round(sigma, 1)}


def battery_sag(f: Flight, t0: float, t1: float, rng, mag: float | None = None) -> dict:
    from .checks import cell_count

    b = f.signals["battery"]
    cells = cell_count(b["voltage"])
    per_cell = float(rng.uniform(0.35, 0.8)) if mag is None else mag
    b["voltage"] -= per_cell * cells * _ramp(b, t0, t1, rise=0.5)
    return {"drop_v_per_cell": round(per_cell, 2), "cells": cells}


def motor_failure(f: Flight, t0: float, t1: float, rng, mag: float | None = None) -> dict:
    mo = f.signals["motors"]
    m = _mask(mo, t0, t1)
    dead = str(rng.choice(mo.columns))
    mo.loc[m, dead] = 0.0
    others = [c for c in mo.columns if c != dead]
    mo.loc[m, others] = (mo.loc[m, others] + 0.25).clip(upper=1.0).clip(lower=0.45)
    return {"motor": dead}


def motor_saturation(f: Flight, t0: float, t1: float, rng, mag: float | None = None) -> dict:
    mo = f.signals["motors"]
    m = _mask(mo, t0, t1)
    sat = str(rng.choice(mo.columns))
    level = 1.0 if mag is None else mag
    mo.loc[m, sat] = level
    return {"motor": sat, "level": level}


def attitude_divergence(f: Flight, t0: float, t1: float, rng, mag: float | None = None) -> dict:
    a = f.signals["att"]
    deg = float(rng.uniform(15, 45)) if mag is None else mag
    axis = str(rng.choice(["roll", "pitch"]))
    a[axis] += np.radians(deg) * _ramp(a, t0, t1, rise=0.4)
    return {"axis": axis, "deg": round(deg, 1)}


def ekf_rejection(f: Flight, t0: float, t1: float, rng, mag: float | None = None) -> dict:
    e = f.signals["ekf"]
    col = str(rng.choice([c for c in ("vel", "pos", "hgt") if c in e.columns] or list(e.columns)))
    ratio = float(rng.uniform(1.3, 3.0)) if mag is None else mag
    e.loc[_mask(e, t0, t1), col] = ratio
    return {"sensor": col, "ratio": round(ratio, 2)}


# --------------------------------------------------------------------------- #
# healthy controls: must NOT trigger


def control_tracked_manoeuvre(f: Flight, t0: float, t1: float, rng, mag: float | None = None) -> dict:
    """Aggressive roll that the vehicle follows: setpoint and attitude move together."""
    deg = float(rng.uniform(20, 45))
    a, sp = f.signals["att"], f.signals["att_sp"]
    a["roll"] += np.radians(deg) * _ramp(a, t0, t1, rise=0.4)
    sp["roll"] += np.radians(deg) * _ramp(sp, t0, t1, rise=0.4)
    return {"deg": round(deg, 1)}


def control_load_step(f: Flight, t0: float, t1: float, rng, mag: float | None = None) -> dict:
    """Current step with the voltage drop a healthy pack shows (V = V0 - R*I)."""
    b = f.signals["battery"]
    if "current" not in b:
        b["current"] = 10.0
    amps = float(rng.uniform(15, 40))
    env = _ramp(b, t0, t1, rise=0.5)
    r_int = 0.008 * max(1, int(np.ceil(b["voltage"].quantile(0.98) / 4.25)))  # ~8 mOhm per cell, a healthy LiPo
    b["current"] += amps * env
    b["voltage"] -= amps * r_int * env
    return {"amps": round(amps, 1)}


INJECTORS: dict[str, tuple[Callable, str | None, tuple[str, ...]]] = {
    # name: (function, check that should fire, signals required)
    "gps_jump": (gps_jump, "gps", ("gps",)),
    "gps_degradation": (gps_degradation, "gps", ("gps",)),
    "vibration_burst": (vibration_burst, "vibration", ("accel",)),
    "battery_sag": (battery_sag, "battery", ("battery",)),
    "motor_failure": (motor_failure, "motors", ("motors",)),
    "motor_saturation": (motor_saturation, "motors", ("motors",)),
    "attitude_divergence": (attitude_divergence, "attitude", ("att", "att_sp")),
    "ekf_rejection": (ekf_rejection, "estimator", ("ekf",)),
    "control_tracked_manoeuvre": (control_tracked_manoeuvre, None, ("att", "att_sp")),
    "control_load_step": (control_load_step, None, ("battery",)),
}


# Magnitude sweeps for the sensitivity study: (parameter name, low, high).
SWEEPS = {
    "gps_jump": ("jump [m]", 2.0, 120.0),
    "gps_degradation": ("eph [m]", 1.0, 8.0),
    "vibration_burst": ("noise sigma [m/s^2]", 0.5, 12.0),
    "battery_sag": ("drop [V/cell]", 0.02, 0.6),
    "attitude_divergence": ("error [deg]", 2.0, 35.0),
    "ekf_rejection": ("test ratio", 0.5, 2.5),
}


def inject(
    flight: Flight, fault: str, t0: float, duration: float, rng, mag: float | None = None
) -> tuple[Flight, Injection]:
    fn, expected, _ = INJECTORS[fault]
    f = flight.copy()
    params = fn(f, t0, t0 + duration, rng, mag)
    return f, Injection(fault, expected, t0, t0 + duration, params)


def applicable(flight: Flight, fault: str) -> bool:
    return flight.has(*INJECTORS[fault][2])
