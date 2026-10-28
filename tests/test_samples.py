"""End-to-end on real PX4 logs from four schema generations."""

import pytest

from sortie import analyze
from sortie.ingest import load


def test_all_samples_parse(samples):
    logs = sorted(samples.glob("*.ulg"))
    assert len(logs) >= 5
    for log in logs:
        f = load(log)
        assert f.duration_s > 0
        assert "accel" in f.signals and "att" in f.signals


def test_vtol_servos_are_not_motors(samples):
    f = load(samples / "cubeorange_vtol_6s.ulg")
    assert list(f.signals["motors"].columns) == ["m4", "m5", "m6", "m7"]


def test_uncalibrated_power_module_ignored(samples):
    assert "battery" not in load(samples / "fmuv4_115s.ulg").signals


@pytest.mark.parametrize("name,verdict", [("sitl_quad_31s", "PASS"), ("auav_x21_69s", "PASS")])
def test_verdicts(samples, name, verdict):
    a = analyze(samples / f"{name}.ulg")
    assert a.verdict == verdict
    d = a.to_dict()
    assert d["flight"] == name and isinstance(d["findings"], list)
    assert "<html>" in a.html().lower()


def test_gps_converging_at_startup_is_flagged(samples):
    a = analyze(samples / "sitl_bench_11s.ulg", anomalies=False)
    assert any(f.check == "gps" and f.severity == "critical" for f in a.findings)
