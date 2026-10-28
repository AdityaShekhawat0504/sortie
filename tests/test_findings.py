import pandas as pd

from sortie.findings import Finding, episodes


def series(values, dt=0.1):
    return pd.Series(values, index=[i * dt for i in range(len(values))])


def r(eps):
    return [(round(a, 6), round(b, 6)) for a, b in eps]


def test_episodes_basic():
    got = episodes(series([0, 1, 1, 1, 0, 0, 1, 0]), merge_gap=0.1)
    assert r(got) == [(0.1, 0.3), (0.6, 0.6)]
    # with a wider merge gap the two bursts become one episode
    assert r(episodes(series([0, 1, 1, 1, 0, 0, 1, 0]), merge_gap=0.5)) == [(0.1, 0.6)]


def test_episodes_merge_and_min_samples():
    s = series([1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    assert r(episodes(s, merge_gap=0.25)) == [(0.0, 0.4), (1.4, 1.4)]
    assert r(episodes(s, merge_gap=0.25, min_samples=2)) == [(0.0, 0.4)]


def test_episodes_empty():
    assert episodes(series([0, 0, 0])) == []
    assert episodes(pd.Series([], dtype=bool)) == []


def test_finding_overlap_and_dict():
    f = Finding("gps", "warning", 10.0, 12.0, "x", {"v": 1.23456})
    assert f.overlaps(11, 15) and f.overlaps(12.4, 13, slack=0.5) and not f.overlaps(13, 14)
    assert f.to_dict()["evidence"]["v"] == 1.2346
