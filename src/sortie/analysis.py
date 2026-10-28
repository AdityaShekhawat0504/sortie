"""One call from a log file to findings, anomalies and a report."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .anomaly import AnomalousWindow, score
from .checks import run_checks
from .findings import Finding
from .ingest import Flight, load


@dataclass
class Analysis:
    flight: Flight
    findings: list[Finding]
    anomalies: list[AnomalousWindow] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        from .report import verdict

        return verdict(self.findings)

    def to_dict(self) -> dict:
        start, end = self.flight.armed_window()
        return {
            "flight": self.flight.name,
            "verdict": self.verdict,
            "meta": self.flight.meta,
            "armed_window_s": [round(start, 2), round(end, 2)],
            "signals": sorted(self.flight.signals),
            "findings": [f.to_dict() for f in self.findings],
            "anomalies": [
                {
                    "t_start": round(w.t_start, 2),
                    "t_end": round(w.t_end, 2),
                    "score": w.score,
                    "explanation": w.explain(),
                }
                for w in self.anomalies
            ],
        }

    def html(self) -> str:
        from .report import render

        return render(self.flight, self.findings, self.anomalies)


def analyze(source: str | Path | Flight, config: dict | None = None, anomalies: bool = True) -> Analysis:
    flight = source if isinstance(source, Flight) else load(source)
    findings = run_checks(flight, config)
    windows = score(flight) if anomalies else []
    return Analysis(flight, findings, windows)
