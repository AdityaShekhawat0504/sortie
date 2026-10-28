"""Rule-based post-flight checks.

Each check takes a :class:`~sortie.ingest.Flight` and returns a list of
:class:`~sortie.findings.Finding`. Checks only look at the armed part of the
flight (whole log for bench logs that never arm) and say so when the data
needed to assess something is missing, instead of silently passing.

Thresholds live in :data:`DEFAULTS` so they can be tuned per airframe. Where a
threshold mirrors a PX4 parameter default, the parameter is named.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from .findings import Finding, episodes, within
from .ingest import Flight, on_ground

DEFAULTS = {
    # vibration: RMS of high-frequency accel noise (all axes combined)
    "vib_warn_ms2": 4.0,
    "vib_crit_ms2": 8.0,
    "vib_min_rate_hz": 20.0,
    # GPS (EKF2_REQ_EPH = 3 m, EKF2_REQ_NSATS = 6 in PX4)
    "gps_eph_m": 3.0,
    "gps_min_sats": 6,
    "gps_jump_min_ms": 30.0,
    "gps_jump_min_m": 10.0,
    # EKF innovation test ratio: > 1 means the measurement was rejected
    "ekf_ratio": 1.0,
    "ekf_ratio_crit": 2.0,
    # battery, per cell under load
    "cell_warn_v": 3.4,
    "cell_crit_v": 3.2,
    "sag_min_v_per_cell": 0.15,
    # attitude tracking error
    "att_warn_deg": 10.0,
    "att_crit_deg": 25.0,
    "att_min_s": 0.5,
    # motors (normalised 0..1)
    "motor_sat": 0.97,
    "motor_sat_min_s": 0.5,
    "motor_imbalance": 0.12,
    "motor_dead": 0.05,
}


def _scope(flight: Flight, df: pd.DataFrame) -> pd.DataFrame:
    start, end = flight.armed_window()
    return within(df, start, end)


def _rate_hz(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return 0.0
    dt = np.median(np.diff(df.index.to_numpy()))
    return float(1.0 / dt) if dt > 0 else 0.0


def _missing(check: str, what: str) -> Finding:
    return Finding(check, "info", 0.0, 0.0, f"Not assessed: {what} not in log.")


# --------------------------------------------------------------------------- #


def check_vibration(flight: Flight, cfg: dict = DEFAULTS) -> list[Finding]:
    if not flight.has("accel"):
        return [_missing("vibration", "accelerometer data")]
    acc = _scope(flight, flight.signals["accel"])
    out: list[Finding] = []

    if "clip" in acc and len(acc) > 1:
        clips = acc["clip"].diff().fillna(0) > 0
        for s, e in episodes(clips, merge_gap=2.0):
            n = int(within(acc, s, e)["clip"].diff().clip(lower=0).sum())
            out.append(
                Finding(
                    "vibration",
                    "critical" if n > 50 else "warning",
                    s,
                    e,
                    f"Accelerometer clipping ({n} samples). The IMU saturated, so "
                    "the estimator was fed wrong accelerations.",
                    {"clipped_samples": n},
                )
            )

    rate = _rate_hz(acc)
    if rate < cfg["vib_min_rate_hz"]:
        out.append(
            Finding(
                "vibration",
                "info",
                0.0,
                0.0,
                f"Vibration level not assessed: accelerometer logged at {rate:.1f} Hz, "
                f"need >= {cfg['vib_min_rate_hz']:.0f} Hz.",
            )
        )
        return out

    win = max(int(rate * 0.5), 5)
    axes = acc[["ax", "ay", "az"]]
    hf = axes - axes.rolling(win * 2, center=True, min_periods=3).mean()
    level = np.sqrt((hf**2).rolling(win, center=True, min_periods=3).mean().sum(axis=1))
    for sev, thr in (("critical", cfg["vib_crit_ms2"]), ("warning", cfg["vib_warn_ms2"])):
        for s, e in episodes(level > thr, min_duration=0.3, merge_gap=1.0):
            if any(f.check == "vibration" and f.overlaps(s, e) and f.severity == "critical" for f in out):
                continue
            peak = float(within(level.to_frame("l"), s, e)["l"].max())
            out.append(
                Finding(
                    "vibration",
                    sev,
                    s,
                    e,
                    f"High-frequency vibration {peak:.1f} m/s^2 RMS (limit {thr:.0f}). "
                    "Check prop balance, motor bearings and flight-controller mounting.",
                    {"peak_rms_ms2": peak, "threshold": thr},
                )
            )
    return out


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def check_gps(flight: Flight, cfg: dict = DEFAULTS) -> list[Finding]:
    if not flight.has("gps"):
        return [_missing("gps", "GPS data")]
    g = _scope(flight, flight.signals["gps"])
    if g.empty:
        return []
    out: list[Finding] = []

    if "eph" in g:
        for s, e in episodes(g["eph"] > cfg["gps_eph_m"], merge_gap=2.0, min_samples=2):
            peak = float(within(g, s, e)["eph"].max())
            out.append(
                Finding(
                    "gps",
                    "critical" if peak > 3 * cfg["gps_eph_m"] else "warning",
                    s,
                    e,
                    f"GPS horizontal accuracy degraded to {peak:.1f} m "
                    f"(EKF2_REQ_EPH default {cfg['gps_eph_m']:.0f} m). Possible "
                    "interference, multipath or antenna issue.",
                    {"peak_eph_m": peak},
                )
            )
    if "sats" in g:
        for s, e in episodes(g["sats"] < cfg["gps_min_sats"], merge_gap=2.0, min_samples=2):
            low = int(within(g, s, e)["sats"].min())
            out.append(
                Finding(
                    "gps",
                    "warning",
                    s,
                    e,
                    f"Only {low} satellites used (EKF2_REQ_NSATS default {cfg['gps_min_sats']}).",
                    {"min_sats": low},
                )
            )

    # Position jumps: distance between fixes that implies an impossible speed.
    if len(g) > 2:
        lat, lon = g["lat"].to_numpy(), g["lon"].to_numpy()
        dist = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dt = np.diff(g.index.to_numpy())
        speed = np.divide(dist, dt, out=np.zeros_like(dist), where=dt > 0)
        reported = np.hypot(g.get("vn", 0 * g["lat"]), g.get("ve", 0 * g["lat"])).to_numpy()[1:]
        limit = np.maximum(cfg["gps_jump_min_ms"], 3 * reported + 10)
        # At 10-20 Hz GPS a 2 m wobble already "implies" 40 m/s, so also
        # require a jump bigger than the receiver's own accuracy estimate.
        eph = g["eph"].to_numpy()[1:] if "eph" in g else np.zeros_like(dist)
        min_jump = np.maximum(cfg["gps_jump_min_m"], 3 * np.nan_to_num(eph))
        jump = pd.Series((speed > limit) & (dist > min_jump), index=g.index[1:])
        for s, e in episodes(jump, merge_gap=1.0):
            seg = jump.index[(jump.index >= s) & (jump.index <= e)]
            k = int(np.argmax(dist[np.isin(g.index[1:], seg)]))
            jmp = float(dist[np.isin(g.index[1:], seg)][k])
            out.append(
                Finding(
                    "gps",
                    "critical",
                    s,
                    e,
                    f"GPS position jumped {jmp:.0f} m between fixes, faster than the "
                    "vehicle can move. Typical of spoofing, jamming recovery or a receiver glitch.",
                    {"jump_m": jmp},
                )
            )
    return out


def check_estimator(flight: Flight, cfg: dict = DEFAULTS) -> list[Finding]:
    if not flight.has("ekf"):
        return [_missing("estimator", "EKF innovation test ratios")]
    ekf = _scope(flight, flight.signals["ekf"])
    out: list[Finding] = []
    names = {"vel": "velocity", "pos": "position", "hgt": "height", "mag": "magnetometer/heading"}
    for col in ekf.columns:
        series = ekf[col].fillna(0)
        for s, e in episodes(series > cfg["ekf_ratio"], merge_gap=1.0, min_samples=2):
            peak = float(within(series.to_frame("r"), s, e)["r"].max())
            sev = "critical" if peak > cfg["ekf_ratio_crit"] or e - s > 5 else "warning"
            out.append(
                Finding(
                    "estimator",
                    sev,
                    s,
                    e,
                    f"EKF rejected {names.get(col, col)} measurements (test ratio {peak:.2f} > 1). "
                    "The estimator disagreed with that sensor.",
                    {"sensor": col, "peak_ratio": peak},
                )
            )
    return out


def cell_count(voltage: pd.Series) -> int:
    """Estimate LiPo cell count from the highest observed pack voltage."""
    vmax = float(voltage.quantile(0.98))
    return max(1, int(math.ceil(vmax / 4.25)))


def check_battery(flight: Flight, cfg: dict = DEFAULTS) -> list[Finding]:
    if not flight.has("battery"):
        return [_missing("battery", "battery data")]
    full = flight.signals["battery"]
    b = _scope(flight, full)
    if len(b) < 3:
        return []
    cells = cell_count(full["voltage"])
    vcell = b["voltage"] / cells
    out: list[Finding] = []

    for sev, thr in (("critical", cfg["cell_crit_v"]), ("warning", cfg["cell_warn_v"])):
        for s, e in episodes(vcell < thr, merge_gap=2.0, min_samples=2):
            if any(f.overlaps(s, e) and f.summary.startswith("Cell") for f in out):
                continue
            low = float(within(vcell.to_frame("v"), s, e)["v"].min())
            out.append(
                Finding(
                    "battery",
                    sev,
                    s,
                    e,
                    f"Cell voltage down to {low:.2f} V under load ({cells}S pack, limit {thr:.1f} V).",
                    {"min_cell_v": low, "cells": cells},
                )
            )

    # Sag not explained by load: robustly fit V = a - R*I + c*t + d*t^2 (load
    # plus slow discharge) and look for large negative residuals. A healthy
    # pack follows the model; a weak cell, loose connector or brownout does not.
    resid, evidence, thr = battery_residual(b, cells, cfg)
    for s, e in episodes(resid < -thr, merge_gap=2.0, min_samples=2):
        drop = float(-within(resid.to_frame("r"), s, e)["r"].min())
        out.append(
            Finding(
                "battery",
                "critical" if drop > 2 * thr else "warning",
                s,
                e,
                f"Voltage dropped {drop:.2f} V more than the load explains. "
                "Check for a weak cell, loose connector or brownout.",
                {"unexplained_drop_v": drop, **evidence},
            )
        )
    return out


def battery_residual(b: pd.DataFrame, cells: int, cfg: dict = DEFAULTS):
    """Robust load + discharge model of pack voltage. Returns residual, evidence, threshold."""
    t = b.index.to_numpy(dtype=float)
    tn = (t - t.mean()) / (t.std() or 1.0)
    cols = [np.ones_like(tn), tn, tn**2]
    use_current = "current" in b and float(b["current"].std()) > 0.5
    if use_current:
        cols.insert(1, b["current"].to_numpy(dtype=float))
    X = np.column_stack(cols)
    y = b["voltage"].to_numpy(dtype=float)
    inlier = np.ones(len(y), dtype=bool)
    for _ in range(3):  # iteratively drop outliers so a sag cannot bend the fit
        coef, *_ = np.linalg.lstsq(X[inlier], y[inlier], rcond=None)
        r = y - X @ coef
        mad = float(np.median(np.abs(r[inlier] - np.median(r[inlier])))) or 1e-3
        inlier = np.abs(r) < 4 * 1.4826 * mad + 1e-6
        if inlier.sum() < max(5, len(y) // 3):
            inlier[:] = True
            break
    resid = pd.Series(r, index=b.index)
    evidence = {"internal_resistance_ohm": float(-coef[1])} if use_current else {}
    thr = max(cfg["sag_min_v_per_cell"] * cells, 6 * 1.4826 * mad)
    return resid, evidence, thr


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def check_attitude(flight: Flight, cfg: dict = DEFAULTS) -> list[Finding]:
    if not flight.has("att", "att_sp"):
        return [_missing("attitude", "attitude or attitude setpoint")]
    att = _scope(flight, flight.signals["att"])
    sp = flight.signals["att_sp"]
    if len(att) < 3 or len(sp) < 3:
        return []
    t = att.index.to_numpy()
    err = pd.Series(0.0, index=att.index)
    for axis in ("roll", "pitch"):
        sp_i = np.interp(t, sp.index.to_numpy(), sp[axis].to_numpy())
        err = np.maximum(err, np.degrees(np.abs(_wrap(att[axis].to_numpy() - sp_i))))
    # On the ground the vehicle cannot follow a tilt setpoint (it is resting
    # on its legs), so only judge tracking while the land detector says airborne.
    err = np.where(on_ground(flight, t), 0.0, err)
    err = pd.Series(err, index=att.index)
    out = []
    for sev, thr in (("critical", cfg["att_crit_deg"]), ("warning", cfg["att_warn_deg"])):
        for s, e in episodes(err > thr, min_duration=cfg["att_min_s"], merge_gap=0.5, min_samples=2):
            if any(f.overlaps(s, e) for f in out):
                continue
            peak = float(within(err.to_frame("e"), s, e)["e"].max())
            out.append(
                Finding(
                    "attitude",
                    sev,
                    s,
                    e,
                    f"Vehicle did not follow its attitude setpoint: error up to {peak:.0f} deg "
                    f"for {e - s:.1f} s. Points to a control, tuning or actuator problem.",
                    {"peak_error_deg": peak},
                )
            )
    return out


def check_motors(flight: Flight, cfg: dict = DEFAULTS) -> list[Finding]:
    if not flight.has("motors"):
        return [_missing("motors", "motor outputs")]
    m = _scope(flight, flight.signals["motors"])
    if len(m) < 3 or m.to_numpy().max() < 0.1:
        return []  # motors never spun in this window (bench log)
    out = []

    sat = (m >= cfg["motor_sat"]).any(axis=1)
    for s, e in episodes(sat, min_duration=cfg["motor_sat_min_s"], merge_gap=0.5, min_samples=2):
        seg = within(m, s, e)
        which = [c for c in m.columns if (seg[c] >= cfg["motor_sat"]).mean() > 0.5]
        out.append(
            Finding(
                "motors",
                "critical" if e - s > 3 else "warning",
                s,
                e,
                f"Motor(s) {', '.join(which) or '?'} saturated for {e - s:.1f} s. "
                "The controller ran out of thrust authority.",
                {"motors": which},
            )
        )

    flying = m.mean(axis=1).between(0.2, 0.9)
    if flying.sum() >= 3:
        mf = m[flying]
        others = mf.sum(axis=1).to_numpy()[:, None] - mf.to_numpy()
        others = others / (mf.shape[1] - 1)
        dead = pd.Series(((mf.to_numpy() < cfg["motor_dead"]) & (others > 0.3)).any(axis=1), index=mf.index)
        for s, e in episodes(dead, merge_gap=0.5, min_samples=2):
            seg = within(mf, s, e)
            which = seg.mean().idxmin()
            out.append(
                Finding(
                    "motors",
                    "critical",
                    s,
                    e,
                    f"Motor {which} output near zero while the others work hard: "
                    "motor or ESC failure.",
                    {"motor": which},
                )
            )
        dev = (mf - mf.mean(axis=1).to_numpy()[:, None]).mean()
        worst = dev.abs().idxmax()
        if abs(dev[worst]) > cfg["motor_imbalance"] and not any(f.summary.startswith("Motor m") for f in out):
            out.append(
                Finding(
                    "motors",
                    "warning",
                    float(mf.index[0]),
                    float(mf.index[-1]),
                    f"Motor {worst} runs {dev[worst]:+.2f} away from the average over the "
                    "whole flight. Typical of a CG offset, bent prop or tired motor.",
                    {"motor": worst, "mean_deviation": float(dev[worst])},
                )
            )
    return out


CHECKS: dict[str, Callable[[Flight, dict], list[Finding]]] = {
    "vibration": check_vibration,
    "gps": check_gps,
    "estimator": check_estimator,
    "battery": check_battery,
    "attitude": check_attitude,
    "motors": check_motors,
}


# Checks that only make sense while the vehicle is armed and flying.
FLIGHT_ONLY = {"attitude", "motors", "battery", "vibration"}


def run_checks(flight: Flight, cfg: dict | None = None) -> list[Finding]:
    cfg = {**DEFAULTS, **(cfg or {})}
    findings: list[Finding] = []
    bench = "status" in flight.signals and not flight.ever_armed
    if bench:
        findings.append(
            Finding(
                "flight",
                "info",
                0.0,
                flight.duration_s,
                "Vehicle never armed (bench or ground log): flight checks "
                f"({', '.join(sorted(FLIGHT_ONLY))}) skipped.",
            )
        )
    for name, fn in CHECKS.items():
        if bench and name in FLIGHT_ONLY:
            continue
        findings.extend(fn(flight, cfg))
    return sorted(findings, key=lambda f: (f.t_start, f.check))
