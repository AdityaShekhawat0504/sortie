import numpy as np
import pandas as pd
import pytest

from sortie.checks import cell_count, run_checks
from sortie.inject import INJECTORS, inject


def alerts(findings, check=None):
    return [f for f in findings if f.severity != "info" and (check is None or f.check == check)]


def test_clean_flight_is_quiet(flight):
    assert alerts(run_checks(flight)) == []


@pytest.mark.parametrize("fault", [k for k, v in INJECTORS.items() if v[1] is not None])
def test_each_fault_triggers_its_check(flight, fault):
    rng = np.random.default_rng(1)
    faulty, inj = inject(flight, fault, 20.0, 3.0, rng)
    hits = [f for f in alerts(run_checks(faulty), inj.expected_check) if f.overlaps(inj.t_start, inj.t_end, 0.5)]
    assert hits, f"{fault} not detected"


@pytest.mark.parametrize("fault", [k for k, v in INJECTORS.items() if v[1] is None])
def test_healthy_controls_stay_quiet(flight, fault):
    rng = np.random.default_rng(2)
    faulty, _ = inject(flight, fault, 20.0, 3.0, rng)
    assert alerts(run_checks(faulty)) == []


def test_bench_log_skips_flight_checks(flight):
    st = flight.signals["status"]
    st["armed"] = False
    names = {f.check for f in run_checks(flight)}
    assert "flight" in names and "attitude" not in names and "motors" not in names


def test_attitude_not_judged_on_ground(flight):
    rng = np.random.default_rng(3)
    faulty, _ = inject(flight, "attitude_divergence", 20.0, 3.0, rng, mag=30.0)
    t = np.array([0.0, 59.0])
    faulty.signals["land"] = pd.DataFrame({"on_ground": [True, True]}, index=pd.Index(t, name="t"))
    assert alerts(run_checks(faulty), "attitude") == []


def test_missing_signal_reported_not_silent(flight):
    del flight.signals["gps"]
    gps = [f for f in run_checks(flight) if f.check == "gps"]
    assert len(gps) == 1 and gps[0].severity == "info" and "Not assessed" in gps[0].summary


def test_gps_wobble_on_fast_receiver_is_not_a_jump(flight):
    g = flight.signals["gps"]
    g.loc[g.index > 30, "lat"] += 2.0 / 111_320  # 2 m step at 10 Hz = 20 m/s implied
    assert alerts(run_checks(flight), "gps") == []


def test_cell_count():
    assert cell_count(pd.Series([16.8, 15.0])) == 4
    assert cell_count(pd.Series([25.2, 22.0])) == 6
