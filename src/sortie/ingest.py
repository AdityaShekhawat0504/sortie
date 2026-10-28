"""Turn a PX4 ULog file into a set of canonical, schema-independent signals.

PX4 renames and moves fields between releases (``lat`` as int 1e7 vs
``latitude_deg``, vibration metrics in ``estimator_status.vibe[]`` vs not at
all, test ratios in ``estimator_status`` vs ``estimator_innovation_test_ratios``
and so on). Every check in Sortie works on the canonical signals defined here,
so supporting a new PX4 version means touching one adapter, not every check.

Canonical signals (all indexed by ``t``, seconds since log start):

==============  ===========================================================
``accel``       ax, ay, az [m/s^2]; optional ``clip`` (cumulative clip count)
``gyro``        gx, gy, gz [rad/s]
``gps``         lat, lon [deg], eph, epv [m], sats, fix, optional ``jam``
``lpos``        x, y, z, vx, vy, vz [m, m/s] (NED, local frame)
``battery``     voltage [V], current [A], optional remaining [0..1]
``att``         roll, pitch, yaw [rad]
``att_sp``      roll, pitch [rad]
``motors``      m0..mN normalised to [0, 1]
``ekf``         one column per innovation test ratio (1.0 = rejection limit)
``status``      armed [bool], nav_state, failsafe
``land``        on_ground [bool] (land detector: landed or ground contact)
==============  ===========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# PX4 arming_state value for "armed" (stable across releases).
ARMED = 2


@dataclass
class Flight:
    """A parsed flight: metadata plus canonical signals."""

    name: str
    meta: dict = field(default_factory=dict)
    signals: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(self.meta.get("duration_s", 0.0))

    def has(self, *names: str) -> bool:
        return all(n in self.signals and not self.signals[n].empty for n in names)

    @property
    def ever_armed(self) -> bool:
        st = self.signals.get("status")
        return st is not None and not st.empty and bool(st["armed"].any())

    def armed_window(self) -> tuple[float, float]:
        """First arm to last disarm. Falls back to the whole log."""
        st = self.signals.get("status")
        if st is not None and not st.empty and st["armed"].any():
            armed_t = st.index[st["armed"].to_numpy()]
            end = self.duration_s
            # last disarm after the last armed sample
            after = st[(st.index > armed_t[-1]) & (~st["armed"])]
            if not after.empty:
                end = float(after.index[0])
            return float(armed_t[0]), float(end)
        return 0.0, self.duration_s

    def copy(self) -> "Flight":
        return Flight(
            name=self.name,
            meta=dict(self.meta),
            signals={k: v.copy() for k, v in self.signals.items()},
        )


# --------------------------------------------------------------------------- #
# helpers


def _frame(data: dict, t0: int, cols: dict[str, str]) -> pd.DataFrame:
    """Build a DataFrame from a ULog data dict with renamed columns."""
    out = {}
    for canon, src in cols.items():
        if src in data:
            out[canon] = np.asarray(data[src], dtype=float)
    if not out:
        return pd.DataFrame()
    t = (np.asarray(data["timestamp"], dtype=np.int64) - t0) / 1e6
    df = pd.DataFrame(out, index=pd.Index(t, name="t"))
    return df[~df.index.duplicated()].sort_index()


def _first(ulog, name: str):
    """Instance 0 of a topic, or None."""
    for d in ulog.data_list:
        if d.name == name and d.multi_id == 0:
            return d.data
    return None


def quat_to_euler(q0, q1, q2, q3):
    """PX4 quaternion (w, x, y, z) to roll, pitch, yaw in radians."""
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    pitch = np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1.0, 1.0))
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return roll, pitch, yaw


# --------------------------------------------------------------------------- #
# adapters: one per canonical signal


def _accel_gyro(ulog, t0, sig):
    d = _first(ulog, "sensor_combined")
    if d is None:
        return
    sig["accel"] = _frame(
        d,
        t0,
        {
            "ax": "accelerometer_m_s2[0]",
            "ay": "accelerometer_m_s2[1]",
            "az": "accelerometer_m_s2[2]",
            "clip": "accelerometer_clipping",
        },
    )
    if "clip" in sig["accel"]:
        # Newer PX4 logs a bitmask per sample; count samples that clipped.
        sig["accel"]["clip"] = (sig["accel"]["clip"] > 0).cumsum().astype(float)
    sig["gyro"] = _frame(d, t0, {"gx": "gyro_rad[0]", "gy": "gyro_rad[1]", "gz": "gyro_rad[2]"})


def _gps(ulog, t0, sig):
    d = _first(ulog, "vehicle_gps_position")
    if d is None:
        return
    if "latitude_deg" in d:  # PX4 >= 1.14
        cols = {"lat": "latitude_deg", "lon": "longitude_deg"}
        scale = 1.0
    else:  # older: degrees * 1e7 as int
        cols = {"lat": "lat", "lon": "lon"}
        scale = 1e-7
    cols.update(
        {
            "eph": "eph",
            "epv": "epv",
            "sats": "satellites_used",
            "fix": "fix_type",
            "jam": "jamming_indicator",
            "vn": "vel_n_m_s",
            "ve": "vel_e_m_s",
        }
    )
    df = _frame(d, t0, cols)
    if df.empty:
        return
    df["lat"] *= scale
    df["lon"] *= scale
    sig["gps"] = df


def _lpos(ulog, t0, sig):
    d = _first(ulog, "vehicle_local_position")
    if d is None:
        return
    sig["lpos"] = _frame(d, t0, {k: k for k in ("x", "y", "z", "vx", "vy", "vz")})


def _battery(ulog, t0, sig):
    d = _first(ulog, "battery_status")
    if d is None:
        return
    df = _frame(d, t0, {"voltage": "voltage_v", "current": "current_a", "remaining": "remaining"})
    if df.empty or "voltage" not in df:
        return
    df = df[df["voltage"] > 1.0]  # sensor not connected yet
    # Uncalibrated power modules report nonsense (hundreds of volts); a
    # drone pack is at most 14S (~59 V).
    if not df.empty and df["voltage"].median() < 60:
        sig["battery"] = df


def _attitude(ulog, t0, sig):
    d = _first(ulog, "vehicle_attitude")
    if d is not None and "q[0]" in d:
        q = [np.asarray(d[f"q[{i}]"], dtype=float) for i in range(4)]
        r, p, y = quat_to_euler(*q)
        t = (np.asarray(d["timestamp"], dtype=np.int64) - t0) / 1e6
        df = pd.DataFrame({"roll": r, "pitch": p, "yaw": y}, index=pd.Index(t, name="t"))
        sig["att"] = df[~df.index.duplicated()].sort_index()
    d = _first(ulog, "vehicle_attitude_setpoint")
    if d is not None:
        df = _frame(d, t0, {"roll": "roll_body", "pitch": "pitch_body"})
        if not df.empty:
            sig["att_sp"] = df


def _motors(ulog, t0, sig):
    d = _first(ulog, "actuator_motors")
    if d is not None:  # PX4 >= 1.13: already normalised 0..1, NaN when unused
        cols = {f"m{i}": f"control[{i}]" for i in range(12)}
        df = _frame(d, t0, cols).dropna(axis=1, how="all")
        if not df.empty:
            sig["motors"] = df.clip(0, 1)
            return
    d = _first(ulog, "actuator_outputs")
    if d is None:
        return
    n = int(np.nanmax(d.get("noutputs", [4]))) or 4
    n = min(n, 8)
    df = _frame(d, t0, {f"m{i}": f"output[{i}]" for i in range(n)})
    if df.empty:
        return
    if df.to_numpy().max() > 10:  # PWM microseconds
        df = (df - 1000.0) / 1000.0
    df = df.clip(0, 1)
    # actuator_outputs mixes motors with servos (VTOL, fixed wing). A motor
    # idles at ~0 when disarmed and moves; a servo rests near mid-travel; an
    # unused channel never moves. Keep only channels that look like motors.
    is_motor = (df.min() <= 0.05) & (df.max() > 0.05) & (df.std() > 1e-6)
    df = df.loc[:, is_motor]
    if not df.empty:
        sig["motors"] = df


def _ekf(ulog, t0, sig):
    d = _first(ulog, "estimator_status")
    if d is not None and "vel_test_ratio" in d:
        cols = {k: f"{k}_test_ratio" for k in ("vel", "pos", "hgt", "mag")}
        df = _frame(d, t0, cols)
        if not df.empty:
            sig["ekf"] = df
            if "vibe[0]" in d:
                sig["vibe"] = _frame(d, t0, {"coning": "vibe[0]", "hf_dang": "vibe[1]", "hf_dvel": "vibe[2]"})
            return
    d2 = _first(ulog, "estimator_innovation_test_ratios")
    if d2 is not None:
        # Per-axis ratios; reduce to one column per sensor (worst axis).
        groups = {
            "vel": ["gps_hvel[0]", "gps_hvel[1]", "gps_vvel"],
            "pos": ["gps_hpos[0]", "gps_hpos[1]"],
            "hgt": ["baro_vpos", "gps_vpos"],
            "mag": ["mag_field[0]", "mag_field[1]", "mag_field[2]", "heading"],
        }
        t = (np.asarray(d2["timestamp"], dtype=np.int64) - t0) / 1e6
        out = {}
        for g, keys in groups.items():
            arrs = [np.asarray(d2[k], dtype=float) for k in keys if k in d2]
            if arrs:
                out[g] = np.nanmax(np.vstack(arrs), axis=0)
        if out:
            df = pd.DataFrame(out, index=pd.Index(t, name="t"))
            sig["ekf"] = df[~df.index.duplicated()].sort_index()
    if d is not None and "vibe[0]" in d:
        sig["vibe"] = _frame(d, t0, {"coning": "vibe[0]", "hf_dang": "vibe[1]", "hf_dvel": "vibe[2]"})


def _status(ulog, t0, sig):
    d = _first(ulog, "vehicle_status")
    if d is None:
        return
    df = _frame(d, t0, {"arming_state": "arming_state", "nav_state": "nav_state", "failsafe": "failsafe"})
    if df.empty:
        return
    df["armed"] = df["arming_state"] == ARMED
    sig["status"] = df


def _land(ulog, t0, sig):
    d = _first(ulog, "vehicle_land_detected")
    if d is None:
        return
    df = _frame(d, t0, {"landed": "landed", "ground_contact": "ground_contact", "maybe_landed": "maybe_landed"})
    if df.empty:
        return
    sig["land"] = pd.DataFrame({"on_ground": df.fillna(0).astype(bool).any(axis=1)}, index=df.index)


def on_ground(flight: "Flight", t: np.ndarray) -> np.ndarray:
    """Land-detector state (landed / ground contact) at times ``t``; False if unknown."""
    land = flight.signals.get("land")
    if land is None or land.empty:
        return np.zeros(len(t), dtype=bool)
    idx = np.searchsorted(land.index.to_numpy(), t, side="right") - 1
    vals = land["on_ground"].to_numpy()
    return np.where(idx >= 0, vals[np.clip(idx, 0, None)], False)


ADAPTERS = (_accel_gyro, _gps, _lpos, _battery, _attitude, _motors, _ekf, _status, _land)


def load(path: str | Path) -> Flight:
    """Parse a ``.ulg`` file into a :class:`Flight`."""
    from pyulog import ULog  # imported lazily so the rest works without it

    path = Path(path)
    ulog = ULog(str(path))
    t0 = ulog.start_timestamp
    sig: dict[str, pd.DataFrame] = {}
    for adapter in ADAPTERS:
        try:
            adapter(ulog, t0, sig)
        except (KeyError, ValueError) as exc:  # pragma: no cover - defensive
            sig.setdefault("_errors", pd.DataFrame())
            print(f"[sortie] adapter {adapter.__name__} skipped: {exc}")
    sig = {k: v for k, v in sig.items() if not k.startswith("_") and not v.empty}

    info = ulog.msg_info_dict
    # ulog.last_timestamp can be far beyond the data (seen on real logs), so
    # use the extent of the signals actually present.
    duration = max((float(v.index.max()) for v in sig.values()), default=0.0)
    meta = {
        "file": path.name,
        "duration_s": round(duration, 2),
        "hardware": info.get("ver_hw", "unknown"),
        "software": info.get("ver_sw_release_str") or info.get("ver_sw", "unknown"),
        "system": info.get("sys_name", "unknown"),
        "dropouts": len(ulog.dropouts),
        "dropout_ms": int(sum(d.duration for d in ulog.dropouts)),
        "topics": len({d.name for d in ulog.data_list}),
    }
    return Flight(name=path.stem, meta=meta, signals=sig)
