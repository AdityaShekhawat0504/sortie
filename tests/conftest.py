from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sortie.ingest import Flight

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def _df(t, **cols):
    return pd.DataFrame(cols, index=pd.Index(t, name="t"))


def make_flight(duration: float = 60.0, seed: int = 0) -> Flight:
    """A clean, armed, airborne synthetic hover: every check should stay quiet."""
    rng = np.random.default_rng(seed)
    t200 = np.arange(0, duration, 1 / 200)
    t10 = np.arange(0, duration, 1 / 10)
    t50 = np.arange(0, duration, 1 / 50)
    n200, n10, n50 = len(t200), len(t10), len(t50)
    sig = {
        "accel": _df(t200, ax=rng.normal(0, 0.3, n200), ay=rng.normal(0, 0.3, n200),
                     az=rng.normal(-9.81, 0.3, n200)),
        "gyro": _df(t200, gx=rng.normal(0, 0.01, n200), gy=rng.normal(0, 0.01, n200), gz=rng.normal(0, 0.01, n200)),
        "gps": _df(t10, lat=47.85 + np.cumsum(rng.normal(0, 1e-7, n10)), lon=12.12 + np.cumsum(rng.normal(0, 1e-7, n10)),
                   eph=np.full(n10, 0.8), epv=np.full(n10, 1.2), sats=np.full(n10, 14.0), fix=np.full(n10, 3.0),
                   vn=np.zeros(n10), ve=np.zeros(n10)),
        "battery": _df(t10, voltage=16.4 - 0.01 * t10 + rng.normal(0, 0.02, n10),
                       current=np.full(n10, 12.0) + rng.normal(0, 0.8, n10)),
        "att": _df(t50, roll=rng.normal(0, 0.01, n50), pitch=rng.normal(0, 0.01, n50), yaw=np.zeros(n50)),
        "att_sp": _df(t50, roll=np.zeros(n50), pitch=np.zeros(n50)),
        "motors": _df(t50, **{f"m{i}": 0.55 + rng.normal(0, 0.02, n50) for i in range(4)}),
        "ekf": _df(t10, vel=np.full(n10, 0.2), pos=np.full(n10, 0.3), hgt=np.full(n10, 0.25), mag=np.full(n10, 0.1)),
        "status": _df(np.array([0.0, 1.0, duration - 1, duration]), arming_state=[1, 2, 2, 1],
                      nav_state=[0, 0, 0, 0], failsafe=[0, 0, 0, 0], armed=[False, True, True, False]),
    }
    return Flight(name="synthetic", meta={"duration_s": duration, "hardware": "test"}, signals=sig)


@pytest.fixture
def flight():
    return make_flight()


@pytest.fixture
def samples():
    return SAMPLES
