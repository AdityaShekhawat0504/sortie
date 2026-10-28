"""Finding model and the interval helper every check uses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


@dataclass
class Finding:
    """One thing a reviewer should look at, with the time window to look at."""

    check: str
    severity: str  # info | warning | critical
    t_start: float
    t_end: float
    summary: str
    evidence: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    def overlaps(self, start: float, end: float, slack: float = 0.0) -> bool:
        return self.t_start <= end + slack and self.t_end >= start - slack

    def to_dict(self) -> dict:
        d = asdict(self)
        d["t_start"] = round(self.t_start, 2)
        d["t_end"] = round(self.t_end, 2)
        d["evidence"] = {k: _jsonable(v) for k, v in self.evidence.items()}
        return d


def _jsonable(v):
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def episodes(
    mask: pd.Series,
    min_duration: float = 0.0,
    merge_gap: float = 1.0,
    min_samples: int = 1,
) -> list[tuple[float, float]]:
    """Turn a boolean time series into ``(start, end)`` intervals.

    Consecutive true samples form an interval; intervals closer than
    ``merge_gap`` seconds are merged; intervals shorter than ``min_duration``
    seconds *and* with fewer than ``min_samples`` samples are dropped. Using
    both criteria keeps the checks usable on low-rate logs (1 Hz topics) and
    high-rate logs (250 Hz) alike.
    """
    if mask.empty or not mask.any():
        return []
    t = mask.index.to_numpy(dtype=float)
    m = mask.to_numpy(dtype=bool)
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    raw = [(t[s], t[e], e - s + 1) for s, e in zip(starts, ends)]

    merged: list[list[float]] = []
    for s, e, n in raw:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
            merged[-1][2] += n
        else:
            merged.append([s, e, n])
    return [(float(s), float(e)) for s, e, n in merged if (e - s) >= min_duration and n >= min_samples]


def within(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index <= end)]
