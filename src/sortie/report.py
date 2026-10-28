"""Self-contained HTML post-flight report (plots embedded as PNG)."""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .anomaly import AnomalousWindow  # noqa: E402
from .findings import SEVERITY_ORDER, Finding  # noqa: E402
from .ingest import Flight  # noqa: E402

COLORS = {"critical": "#c0392b", "warning": "#e67e22", "info": "#7f8c8d"}
CHECK_ROWS = ["vibration", "gps", "estimator", "battery", "attitude", "motors"]


def verdict(findings: list[Finding]) -> str:
    worst = max((SEVERITY_ORDER[f.severity] for f in findings), default=0)
    return {0: "PASS", 1: "REVIEW", 2: "FAIL"}[worst]


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _shade(ax, findings: list[Finding], checks: set[str]):
    for f in findings:
        if f.check in checks and f.severity != "info":
            ax.axvspan(f.t_start, max(f.t_end, f.t_start + 0.2), color=COLORS[f.severity], alpha=0.18, lw=0)


def timeline_png(flight: Flight, findings: list[Finding], anomalies: list[AnomalousWindow]) -> str:
    fig, ax = plt.subplots(figsize=(10, 2.6))
    rows = CHECK_ROWS + ["anomaly"]
    for i, check in enumerate(rows):
        ax.axhline(i, color="#eee", lw=8, zorder=0)
        for f in findings:
            if f.check == check and f.severity != "info":
                ax.plot([f.t_start, max(f.t_end, f.t_start + 0.3)], [i, i], lw=8,
                        color=COLORS[f.severity], solid_capstyle="butt")
    for w in anomalies:
        ax.plot([w.t_start, w.t_end], [len(rows) - 1] * 2, lw=8, color="#8e44ad", solid_capstyle="butt")
    s, e = flight.armed_window()
    ax.axvspan(s, e, color="#2ecc71", alpha=0.06)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlim(0, flight.duration_s)
    ax.set_xlabel("time since log start [s]  (green = armed)")
    ax.invert_yaxis()
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return _png(fig)


def signals_png(flight: Flight, findings: list[Finding]) -> str:
    s = flight.signals
    panels = []
    if "lpos" in s:
        panels.append(("altitude [m]", lambda ax: ax.plot(s["lpos"].index, -s["lpos"]["z"], lw=1), set()))
    if "accel" in s:
        def acc(ax):
            a = s["accel"][["ax", "ay", "az"]]
            hf = a - a.rolling(25, center=True, min_periods=3).mean()
            ax.plot(a.index, np.sqrt((hf**2).sum(axis=1)), lw=0.6)
        panels.append(("accel HF |a| [m/s²]", acc, {"vibration"}))
    if "gps" in s and "eph" in s["gps"]:
        panels.append(("GPS eph [m]", lambda ax: ax.plot(s["gps"].index, s["gps"]["eph"], lw=1), {"gps"}))
    if "ekf" in s:
        def ekf(ax):
            for c in s["ekf"].columns:
                ax.plot(s["ekf"].index, s["ekf"][c], lw=0.9, label=c)
            ax.axhline(1.0, color="k", ls="--", lw=0.7)
            ax.legend(fontsize=7, ncol=4, loc="upper right")
        panels.append(("EKF test ratio", ekf, {"estimator"}))
    if "battery" in s:
        panels.append(("battery [V]", lambda ax: ax.plot(s["battery"].index, s["battery"]["voltage"], lw=1), {"battery"}))
    if "att" in s and "att_sp" in s:
        def att(ax):
            ax.plot(s["att"].index, np.degrees(s["att"]["roll"]), lw=0.9, label="roll")
            ax.plot(s["att_sp"].index, np.degrees(s["att_sp"]["roll"]), lw=0.9, ls="--", label="roll sp")
            ax.legend(fontsize=7, loc="upper right")
        panels.append(("roll [deg]", att, {"attitude"}))
    if "motors" in s:
        def mot(ax):
            for c in s["motors"].columns:
                ax.plot(s["motors"].index, s["motors"][c], lw=0.8, label=c)
            ax.legend(fontsize=7, ncol=4, loc="upper right")
        panels.append(("motor command [0-1]", mot, {"motors"}))
    if not panels:
        return ""
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 1.7 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (label, draw, checks) in zip(axes, panels):
        draw(ax)
        _shade(ax, findings, checks)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("time since log start [s]")
    return _png(fig)


TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>Sortie report: {name}</title>
<style>
body{{font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222}}
h1{{font-size:22px;margin:0}} .muted{{color:#777}}
.verdict{{display:inline-block;padding:4px 12px;border-radius:4px;color:#fff;font-weight:600;margin-left:8px}}
.PASS{{background:#27ae60}} .REVIEW{{background:#e67e22}} .FAIL{{background:#c0392b}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} td,th{{border-bottom:1px solid #eee;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#fafafa}} .sev{{font-weight:600}} .critical{{color:#c0392b}} .warning{{color:#e67e22}} .info{{color:#7f8c8d}}
img{{max-width:100%}} code{{background:#f4f4f4;padding:1px 4px;border-radius:3px}}
</style></head><body>
<h1>Post-flight report: {name}<span class="verdict {verdict}">{verdict}</span></h1>
<p class="muted">{meta_line}</p>
<h2>Timeline</h2><img src="data:image/png;base64,{timeline}">
<h2>Findings ({n_alerts} alerts)</h2>
<table><tr><th>Time [s]</th><th>Check</th><th>Severity</th><th>What happened</th></tr>{rows}</table>
<h2>Unsupervised second opinion</h2>
<p class="muted">Seconds of this flight that look least like the rest (Isolation Forest on per-second features, scaled to this flight).</p>
<table><tr><th>Time [s]</th><th>Score</th><th>Why</th></tr>{anom_rows}</table>
<h2>Signals</h2><img src="data:image/png;base64,{signals}">
<p class="muted">Generated by sortie {version} on {generated}. Thresholds are defaults; tune them per airframe.</p>
</body></html>"""


def render(flight: Flight, findings: list[Finding], anomalies: list[AnomalousWindow]) -> str:
    from . import __version__

    rows = "".join(
        f"<tr><td>{f.t_start:.1f} - {f.t_end:.1f}</td><td>{f.check}</td>"
        f"<td class='sev {f.severity}'>{f.severity}</td><td>{html.escape(f.summary)}</td></tr>"
        for f in sorted(findings, key=lambda x: (-SEVERITY_ORDER[x.severity], x.t_start))
    )
    anom_rows = "".join(
        f"<tr><td>{w.t_start:.1f} - {w.t_end:.1f}</td><td>{w.score:.2f}</td><td>{html.escape(w.explain())}</td></tr>"
        for w in anomalies
    ) or "<tr><td colspan=3>Flight too short to score.</td></tr>"
    m = flight.meta
    start, end = flight.armed_window()
    meta_line = (
        f"{html.escape(str(m.get('hardware')))} · PX4 {html.escape(str(m.get('software'))[:12])} · "
        f"{m.get('duration_s', 0):.0f} s logged · armed {start:.1f}-{end:.1f} s · "
        f"{m.get('dropouts', 0)} dropouts ({m.get('dropout_ms', 0)} ms)"
    )
    return TEMPLATE.format(
        name=html.escape(flight.name),
        verdict=verdict(findings),
        meta_line=meta_line,
        timeline=timeline_png(flight, findings, anomalies),
        n_alerts=sum(f.severity != "info" for f in findings),
        rows=rows,
        anom_rows=anom_rows,
        signals=signals_png(flight, findings),
        version=__version__,
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
