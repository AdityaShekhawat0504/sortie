"""Unsupervised second opinion: which seconds of this flight look least like the rest?

Rules only find what someone wrote a rule for. This layer summarises every
second of the flight into a small feature vector, scales each feature by the
flight's own median and MAD (so a noisy airframe is compared with itself, not
with a fleet average), scores windows with an Isolation Forest and explains
each flagged window by the features that deviate most.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .ingest import Flight

FEATURES = {
    "vib_rms": "accelerometer high-frequency RMS",
    "gyro_std": "gyro rate variability",
    "gps_eph": "GPS horizontal accuracy",
    "gps_speed_gap": "GPS position vs. velocity mismatch",
    "ekf_max": "worst EKF test ratio",
    "batt_v_drop": "pack voltage drop vs. window start",
    "att_err": "attitude tracking error",
    "motor_max": "highest motor command",
    "motor_spread": "spread between motors",
}


@dataclass
class AnomalousWindow:
    t_start: float
    t_end: float
    score: float
    top_features: list[tuple[str, float]]  # (feature, robust z)

    def explain(self) -> str:
        def fmt(z: float) -> str:
            return f"{z:+.1f}" if abs(z) < 100 else (">+99" if z > 0 else "<-99")

        parts = [f"{FEATURES.get(n, n)} (z={fmt(z)})" for n, z in self.top_features]
        return "; ".join(parts)


def _agg(df: pd.DataFrame | None, edges: np.ndarray, fn) -> np.ndarray:
    out = np.full(len(edges) - 1, np.nan)
    if df is None or df.empty:
        return out
    idx = np.searchsorted(edges, df.index.to_numpy(), side="right") - 1
    for i in range(len(out)):
        sel = df[idx == i]
        if len(sel):
            out[i] = fn(sel)
    return out


def window_features(flight: Flight, window_s: float = 1.0) -> pd.DataFrame:
    start, end = flight.armed_window()
    edges = np.arange(start, end + window_s, window_s)
    if len(edges) < 4:
        return pd.DataFrame()
    s = flight.signals
    feats: dict[str, np.ndarray] = {}

    acc = s.get("accel")
    if acc is not None:
        hf = acc[["ax", "ay", "az"]] - acc[["ax", "ay", "az"]].rolling(25, center=True, min_periods=3).mean()
        feats["vib_rms"] = _agg(hf, edges, lambda d: float(np.sqrt((d**2).mean().sum())))
    if s.get("gyro") is not None:
        feats["gyro_std"] = _agg(s["gyro"], edges, lambda d: float(d.std().sum()))
    g = s.get("gps")
    if g is not None:
        feats["gps_eph"] = _agg(g, edges, lambda d: float(d["eph"].max()))
        if {"vn", "ve"} <= set(g.columns):
            lat0 = np.radians(g["lat"].mean())
            n = g["lat"].diff() * 111_320.0
            e = g["lon"].diff() * 111_320.0 * np.cos(lat0)
            dt = pd.Series(g.index, index=g.index).diff()
            gap = (np.hypot(n, e) / dt - np.hypot(g["vn"], g["ve"])).abs()
            feats["gps_speed_gap"] = _agg(gap.to_frame("g").dropna(), edges, lambda d: float(d["g"].max()))
    if s.get("ekf") is not None:
        feats["ekf_max"] = _agg(s["ekf"], edges, lambda d: float(np.nanmax(d.to_numpy())))
    if s.get("battery") is not None:
        b = s["battery"]["voltage"].to_frame("v")
        feats["batt_v_drop"] = _agg(b, edges, lambda d: float(d["v"].iloc[0] - d["v"].min()))
    if s.get("att") is not None and s.get("att_sp") is not None:
        att, sp = s["att"], s["att_sp"]
        t = att.index.to_numpy()
        err = np.maximum(
            np.abs(att["roll"].to_numpy() - np.interp(t, sp.index, sp["roll"])),
            np.abs(att["pitch"].to_numpy() - np.interp(t, sp.index, sp["pitch"])),
        )
        feats["att_err"] = _agg(pd.DataFrame({"e": np.degrees(err)}, index=att.index), edges, lambda d: float(d["e"].max()))
    if s.get("motors") is not None:
        m = s["motors"]
        feats["motor_max"] = _agg(m, edges, lambda d: float(d.to_numpy().max()))
        feats["motor_spread"] = _agg(m, edges, lambda d: float((d.max(axis=1) - d.min(axis=1)).mean()))

    df = pd.DataFrame(feats, index=pd.Index(edges[:-1], name="t"))
    return df.dropna(axis=1, how="all")


def robust_z(df: pd.DataFrame) -> pd.DataFrame:
    med = df.median()
    mad = (df - med).abs().median() * 1.4826
    # Floor the scale so a perfectly flat signal does not explode on tiny changes.
    floor = (df.abs().median() * 0.05).clip(lower=1e-3)
    scale = np.maximum(mad, floor)
    return ((df - med) / scale).fillna(0.0)


def score(flight: Flight, top_k: int = 3, window_s: float = 1.0, seed: int = 0) -> list[AnomalousWindow]:
    feats = window_features(flight, window_s)
    if len(feats) < 6 or feats.shape[1] == 0:
        return []
    z = robust_z(feats)
    model = IsolationForest(n_estimators=200, random_state=seed, contamination="auto")
    model.fit(z.to_numpy())
    raw = -model.score_samples(z.to_numpy())  # higher = more anomalous
    order = np.argsort(raw)[::-1][:top_k]
    out = []
    for i in order:
        row = z.iloc[i]
        top = row.abs().sort_values(ascending=False).head(3)
        out.append(
            AnomalousWindow(
                t_start=float(feats.index[i]),
                t_end=float(feats.index[i] + window_s),
                score=round(float(raw[i]), 3),
                top_features=[(name, float(row[name])) for name in top.index],
            )
        )
    return out
