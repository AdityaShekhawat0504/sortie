import json

from fastapi.testclient import TestClient

from sortie.cli import main


def test_cli_analyze_json_and_html(samples, tmp_path, capsys):
    out = tmp_path / "r.html"
    assert main(["analyze", str(samples / "sitl_quad_31s.ulg"), "--json", "--html", str(out)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"] == "PASS" and out.read_text().startswith("<!doctype html>")


def test_cli_strict_exit_code(samples):
    assert main(["analyze", str(samples / "sitl_bench_11s.ulg"), "--no-anomaly", "--strict"]) == 2


def test_api_roundtrip(samples, tmp_path, monkeypatch):
    monkeypatch.setenv("SORTIE_DATA", str(tmp_path))
    import importlib

    import sortie.api as api

    importlib.reload(api)
    client = TestClient(api.app)
    with open(samples / "sitl_quad_31s.ulg", "rb") as fh:
        r = client.post("/flights", files={"file": ("quad.ulg", fh, "application/octet-stream")})
    assert r.status_code == 200, r.text
    fid = r.json()["id"]
    assert client.get("/flights").json()[0]["id"] == fid
    assert client.get(f"/flights/{fid}/report").status_code == 200
    assert client.get("/flights/doesnotexist/report").status_code == 404
    bad = client.post("/flights", files={"file": ("x.txt", b"hello", "text/plain")})
    assert bad.status_code == 400
