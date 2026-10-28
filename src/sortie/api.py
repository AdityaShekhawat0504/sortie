"""HTTP API: upload a log, get findings or the HTML report back.

    POST /flights            multipart upload of a .ulg -> analysis JSON
    GET  /flights            all analysed flights (fleet store)
    GET  /flights/{id}       findings of one flight
    GET  /flights/{id}/report  HTML report
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from . import __version__
from .analysis import analyze
from .store import FleetStore, file_id

DATA_DIR = Path(os.environ.get("SORTIE_DATA", "./sortie-data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
MAX_BYTES = int(os.environ.get("SORTIE_MAX_MB", "200")) * 1024 * 1024

app = FastAPI(title="Sortie", version=__version__, description="Post-flight forensics for PX4 logs")


def _store() -> FleetStore:
    return FleetStore(DATA_DIR / "fleet.sqlite")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/flights")
async def upload(file: UploadFile = File(...)) -> dict:
    if not (file.filename or "").endswith(".ulg"):
        raise HTTPException(400, "expected a PX4 .ulg file")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "log too large")
    with tempfile.NamedTemporaryFile(suffix=".ulg", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        fid = file_id(tmp_path)
        try:
            a = analyze(tmp_path)
        except Exception as exc:  # corrupt or non-PX4 file
            raise HTTPException(422, f"could not parse log: {exc}") from exc
        a.flight.name = Path(file.filename).stem
        _store().add(fid, a)
        (DATA_DIR / f"{fid}.html").write_text(a.html(), encoding="utf-8")
        return {"id": fid, **a.to_dict()}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/flights")
def flights() -> list[dict]:
    return _store().flights()


@app.get("/flights/{flight_id}")
def flight(flight_id: str) -> dict:
    found = _store().findings(flight_id)
    if not found:
        raise HTTPException(404, "unknown flight")
    return {"id": flight_id, "findings": found}


@app.get("/flights/{flight_id}/report", response_class=HTMLResponse)
def report(flight_id: str) -> str:
    path = DATA_DIR / f"{flight_id}.html"
    if not path.exists() or not flight_id.isalnum():
        raise HTTPException(404, "unknown flight")
    return path.read_text(encoding="utf-8")
