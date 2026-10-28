"""Fault-injection benchmark: how well do the checks actually work?

For every flight that was armed long enough and every applicable fault, inject
the fault at random times (seeded), run all checks and score:

* **detected**: a warning/critical finding from the expected check overlaps
  the injected window (0.5 s slack for window edges).
* **latency**: seconds from fault onset to the start of that finding.
* **false alarms**: new warning/critical findings (not present on the clean
  flight) that do not overlap the injected window.
* **controls** (healthy-but-dramatic changes) count as wrong if *anything*
  new fires inside the window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .checks import FLIGHT_ONLY, run_checks
from .findings import Finding
from .ingest import Flight
from .ingest import on_ground
from .inject import INJECTORS, applicable, inject


def _usable(flight: Flight, fault: str) -> bool:
    """Can this fault be injected meaningfully into this flight?"""
    expected = INJECTORS[fault][1]
    if not applicable(flight, fault):
        return False
    if not flight.ever_armed and (expected in FLIGHT_ONLY or expected is None):
        return False
    if fault in ("attitude_divergence", "control_tracked_manoeuvre"):
        s, e = flight.armed_window()
        t = np.linspace(s, e, 50)
        if on_ground(flight, t).mean() > 0.5:  # tracking is not judged on the ground
            return False
    return True


def _onset(rng, start, end, dur, baseline, expected, tries=60):
    """Random onset whose window does not overlap an existing finding of the
    expected check (there a new fault cannot be told apart from the old one)."""
    for _ in range(tries):
        t0 = float(rng.uniform(start + 0.5, end - dur - 0.25))
        if not any(b.check == expected and b.overlaps(t0, t0 + dur, SLACK_S) for b in baseline):
            return t0
    return None

SLACK_S = 0.5


@dataclass
class Trial:
    flight: str
    fault: str
    t_start: float
    duration: float
    detected: bool
    latency_s: float | None
    false_alarms: int
    params: dict


def _alerts(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity in ("warning", "critical")]


def _new(findings: list[Finding], baseline: list[Finding]) -> list[Finding]:
    """Alerts not explained by a finding already present in the clean flight.

    A baseline finding explains an alert if it covers the alert's whole span
    (0.5 s tolerance). An alert that grew out of a baseline finding into new
    time therefore still counts as new.
    """
    return [
        f
        for f in _alerts(findings)
        if not any(
            b.check == f.check and b.t_start <= f.t_start + 0.5 and b.t_end >= f.t_end - 0.5 for b in baseline
        )
    ]


def run_trials(
    flights: list[Flight],
    n_per_fault: int = 25,
    min_dur: float = 1.5,
    max_dur: float = 3.5,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trials: list[Trial] = []
    for flight in flights:
        # Bench logs (never armed) are still real sensor data: use them for
        # the sensor-level faults that do not need the vehicle to fly.
        start, end = flight.armed_window()
        if end - start < max_dur + 1.5:
            continue
        baseline = _alerts(run_checks(flight))
        for fault in INJECTORS:
            if not _usable(flight, fault):
                continue
            for _ in range(n_per_fault):
                dur = float(rng.uniform(min_dur, max_dur))
                t0 = _onset(rng, start, end, dur, baseline, INJECTORS[fault][1])
                if t0 is None:
                    continue
                faulty, inj = inject(flight, fault, t0, dur, rng)
                new = _new(run_checks(faulty), baseline)
                in_window = [f for f in new if f.overlaps(inj.t_start, inj.t_end, SLACK_S)]
                if inj.expected_check is None:  # control: nothing may fire
                    hits = []
                    detected = len(in_window) > 0  # "detected" == wrongly flagged
                else:
                    hits = [f for f in in_window if f.check == inj.expected_check]
                    detected = bool(hits)
                latency = min(max(0.0, h.t_start - inj.t_start) for h in hits) if hits else None
                false_alarms = len([f for f in new if f not in in_window])
                trials.append(
                    Trial(flight.name, fault, round(t0, 2), round(dur, 2), detected, latency, false_alarms, inj.params)
                )
    return pd.DataFrame([t.__dict__ for t in trials])


def summarise(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return trials
    rows = []
    for fault, g in trials.groupby("fault", sort=False):
        control = INJECTORS[fault][1] is None
        lat = g["latency_s"].dropna()
        rows.append(
            {
                "fault": fault,
                "type": "control (must stay quiet)" if control else f"fault -> {INJECTORS[fault][1]}",
                "trials": len(g),
                "detection_rate" if not control else "wrongly_flagged": round(g["detected"].mean(), 3),
                "median_latency_s": round(float(lat.median()), 2) if len(lat) else None,
                "false_alarms_per_trial": round(g["false_alarms"].mean(), 3),
            }
        )
    out = pd.DataFrame(rows)
    return out


def markdown_table(summary: pd.DataFrame) -> str:
    lines = [
        "| Injected | Expected | Trials | Detected | Median latency | Other false alarms / trial |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        control = "wrongly_flagged" in r and not pd.isna(r.get("wrongly_flagged"))
        rate = r["wrongly_flagged"] if control else r["detection_rate"]
        exp = "nothing (healthy control)" if control else r["type"].split("-> ")[-1]
        det = f"{rate:.0%} flagged" if control else f"{rate:.0%}"
        lat = "n/a" if pd.isna(r["median_latency_s"]) else f"{r['median_latency_s']:.2f} s"
        lines.append(f"| `{r['fault']}` | {exp} | {r['trials']} | {det} | {lat} | {r['false_alarms_per_trial']:.2f} |")
    return "\n".join(lines)


def sensitivity(
    flights: list[Flight],
    levels: int = 12,
    n_per_level: int = 12,
    duration: float = 2.5,
    seed: int = 11,
) -> pd.DataFrame:
    """Detection rate as a function of fault magnitude.

    Answers the question a rule-based detector usually dodges: *how small a
    fault can it see?* Returns one row per (fault, magnitude) with the
    detection rate across flights and random onset times.
    """
    from .inject import SWEEPS

    rng = np.random.default_rng(seed)
    rows = []
    for fault, (label, lo, hi) in SWEEPS.items():
        expected = INJECTORS[fault][1]
        pool = []
        for f in flights:
            if not _usable(f, fault):
                continue
            s, e = f.armed_window()
            if e - s >= duration + 1.5:
                pool.append((f, s, e, _alerts(run_checks(f))))
        if not pool:
            continue
        for mag in np.linspace(lo, hi, levels):
            hits = tried = 0
            for i in range(n_per_level * 3):
                if tried >= n_per_level:
                    break
                f, s, e, base = pool[i % len(pool)]
                t0 = _onset(rng, s, e, duration, base, expected)
                if t0 is None:
                    continue
                tried += 1
                faulty, inj = inject(f, fault, t0, duration, rng, float(mag))
                new = _new(run_checks(faulty), base)
                hits += any(x.check == expected and x.overlaps(inj.t_start, inj.t_end, SLACK_S) for x in new)
            if tried:
                rows.append({"fault": fault, "parameter": label, "magnitude": round(float(mag), 3),
                             "detection_rate": hits / tried, "trials": tried})
    return pd.DataFrame(rows)


def detection_threshold(sens: pd.DataFrame, rate: float = 0.5) -> pd.DataFrame:
    """Smallest swept magnitude with detection rate >= ``rate`` per fault."""
    out = []
    for fault, g in sens.groupby("fault", sort=False):
        ok = g[g["detection_rate"] >= rate]
        out.append({
            "fault": fault,
            "parameter": g["parameter"].iloc[0],
            f"min_detectable@{int(rate * 100)}%": None if ok.empty else float(ok["magnitude"].min()),
        })
    return pd.DataFrame(out)


def anomaly_recall(flights: list[Flight], n_per_fault: int = 15, top_k: int = 3, seed: int = 5) -> pd.DataFrame:
    """How often does the unsupervised layer rank the injected fault in its top-k windows?

    Random ranking would hit with probability ~ (k + fault_len) / n_windows,
    so that baseline is reported next to each result.
    """
    from .anomaly import score

    rng = np.random.default_rng(seed)
    rows = []
    for fault, (_, expected, _) in INJECTORS.items():
        if expected is None:
            continue
        hits = tried = 0
        chance = []
        for flight in flights:
            if not _usable(flight, fault):
                continue
            s, e = flight.armed_window()
            if e - s < 8:
                continue
            for _ in range(n_per_fault):
                dur = 2.0
                t0 = float(rng.uniform(s + 0.5, e - dur - 0.25))
                faulty, inj = inject(flight, fault, t0, dur, rng)
                top = score(faulty, top_k=top_k)
                tried += 1
                hits += any(w.t_start <= inj.t_end and w.t_end >= inj.t_start for w in top)
                n_win = max(1, int(e - s))
                chance.append(min(1.0, (top_k + dur + 1) / n_win))
        if tried:
            rows.append({
                "fault": fault,
                "trials": tried,
                f"in_top{top_k}": round(hits / tried, 3),
                "random_baseline": round(float(np.mean(chance)), 3),
            })
    return pd.DataFrame(rows)
