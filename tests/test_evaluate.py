from sortie.evaluate import detection_threshold, run_trials, sensitivity, summarise
from sortie.ingest import load


def test_benchmark_on_real_flight(samples):
    flights = [load(samples / "sitl_quad_31s.ulg")]
    trials = run_trials(flights, n_per_fault=3, seed=1)
    s = summarise(trials).set_index("fault")
    assert s.loc["gps_jump", "detection_rate"] == 1.0
    assert s.loc["control_tracked_manoeuvre", "wrongly_flagged"] == 0.0
    assert (trials["false_alarms"] == 0).all()


def test_sensitivity_is_monotonic_enough(samples):
    flights = [load(samples / "sitl_quad_31s.ulg")]
    sens = sensitivity(flights, levels=5, n_per_level=3)
    thr = detection_threshold(sens).set_index("fault")
    # the gps jump rule needs > 10 m: tiny jumps must not be "detected"
    assert thr.loc["gps_jump"].iloc[-1] > 2.0
