"""Fleet store: every analysed flight and finding in one SQLite file.

SQLite keeps the tool dependency-free; the schema is plain enough to move to
Postgres or DuckDB unchanged.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from .analysis import Analysis

SCHEMA = """
CREATE TABLE IF NOT EXISTS flights (
    id TEXT PRIMARY KEY,           -- sha256 of the log file
    name TEXT, hardware TEXT, software TEXT,
    duration_s REAL, armed_s REAL, verdict TEXT,
    n_critical INTEGER, n_warning INTEGER,
    analysed_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS findings (
    flight_id TEXT REFERENCES flights(id) ON DELETE CASCADE,
    check_name TEXT, severity TEXT, t_start REAL, t_end REAL,
    summary TEXT, evidence TEXT
);
CREATE INDEX IF NOT EXISTS idx_findings_check ON findings(check_name, severity);
"""


def file_id(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class FleetStore:
    def __init__(self, path: str | Path = "sortie.sqlite"):
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)

    def add(self, flight_id: str, a: Analysis) -> None:
        start, end = a.flight.armed_window()
        alerts = [f for f in a.findings if f.severity != "info"]
        with self.db:
            self.db.execute("DELETE FROM findings WHERE flight_id = ?", (flight_id,))
            self.db.execute(
                "INSERT OR REPLACE INTO flights (id, name, hardware, software, duration_s, armed_s,"
                " verdict, n_critical, n_warning) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    flight_id,
                    a.flight.name,
                    str(a.flight.meta.get("hardware")),
                    str(a.flight.meta.get("software")),
                    a.flight.duration_s,
                    round(end - start, 2) if a.flight.ever_armed else 0.0,
                    a.verdict,
                    sum(f.severity == "critical" for f in alerts),
                    sum(f.severity == "warning" for f in alerts),
                ),
            )
            self.db.executemany(
                "INSERT INTO findings VALUES (?,?,?,?,?,?,?)",
                [
                    (flight_id, f.check, f.severity, f.t_start, f.t_end, f.summary, json.dumps(f.to_dict()["evidence"]))
                    for f in a.findings
                ],
            )

    def flights(self) -> list[dict]:
        cur = self.db.execute("SELECT * FROM flights ORDER BY analysed_at DESC")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def findings(self, flight_id: str) -> list[dict]:
        cur = self.db.execute("SELECT * FROM findings WHERE flight_id = ? ORDER BY t_start", (flight_id,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def fleet_summary(self) -> list[dict]:
        """Which checks fire most across the fleet: where to look first."""
        cur = self.db.execute(
            "SELECT check_name, severity, COUNT(*) AS n, COUNT(DISTINCT flight_id) AS flights "
            "FROM findings WHERE severity != 'info' GROUP BY check_name, severity ORDER BY n DESC"
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
