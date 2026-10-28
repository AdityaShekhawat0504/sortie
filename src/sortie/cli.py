"""Command line: ``sortie analyze``, ``sortie fleet``, ``sortie bench``, ``sortie serve``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ICON = {"critical": "!!", "warning": " !", "info": " ."}


def _logs(path: str) -> list[Path]:
    p = Path(path)
    return sorted(p.rglob("*.ulg")) if p.is_dir() else [p]


def cmd_analyze(args) -> int:
    from .analysis import analyze

    a = analyze(args.log, anomalies=not args.no_anomaly)
    if args.json:
        print(json.dumps(a.to_dict(), indent=2))
    else:
        start, end = a.flight.armed_window()
        armed = f"armed {start:.1f}-{end:.1f} s" if a.flight.ever_armed else "never armed"
        print(f"{a.flight.name}  [{a.verdict}]  {a.flight.meta['hardware']}  "
              f"{a.flight.duration_s:.0f} s logged, {armed}")
        for f in a.findings:
            print(f"  {ICON[f.severity]} {f.t_start:7.1f}s  {f.check:<10} {f.summary}")
        for w in a.anomalies:
            print(f"  ~~ {w.t_start:7.1f}s  anomaly    score {w.score:.2f}: {w.explain()}")
    if args.html:
        Path(args.html).write_text(a.html(), encoding="utf-8")
        print(f"report written to {args.html}", file=sys.stderr)
    return 2 if a.verdict == "FAIL" and args.strict else 0


def cmd_fleet(args) -> int:
    from .analysis import analyze
    from .store import FleetStore, file_id

    store = FleetStore(args.db)
    for log in _logs(args.path):
        a = analyze(log, anomalies=False)
        store.add(file_id(log), a)
        print(f"{a.verdict:<7} {log.name}")
    print("\nMost frequent findings across the fleet:")
    for row in store.fleet_summary():
        print(f"  {row['check_name']:<10} {row['severity']:<8} {row['n']:>4}x in {row['flights']} flight(s)")
    return 0


def cmd_bench(args) -> int:
    from .evaluate import (anomaly_recall, detection_threshold, markdown_table, run_trials,
                           sensitivity, summarise)
    from .ingest import load

    flights = [load(p) for p in _logs(args.path)]
    trials = run_trials(flights, n_per_fault=args.n, seed=args.seed)
    print("## Detection on injected faults\n")
    print(markdown_table(summarise(trials)))
    sens = sensitivity(flights, seed=args.seed)
    print("\n## Smallest fault detected in >= 50% of trials\n")
    print("| Fault | Parameter | Detected at >= |\n|---|---|---:|")
    for _, r in detection_threshold(sens).iterrows():
        v = r.iloc[2]
        print(f"| `{r['fault']}` | {r['parameter']} | {'not reached' if v != v or v is None else f'{v:.2f}'} |")
    if not args.no_anomaly:
        print("\n## Unsupervised layer: injected fault among top-3 anomalous seconds\n")
        print(anomaly_recall(flights, seed=args.seed).to_string(index=False))
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        trials.to_csv(Path(args.out) / "trials.csv", index=False)
        sens.to_csv(Path(args.out) / "sensitivity.csv", index=False)
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("sortie.api:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sortie", description="Post-flight forensics for PX4 logs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyse one .ulg file")
    a.add_argument("log")
    a.add_argument("--html", help="write a self-contained HTML report")
    a.add_argument("--json", action="store_true", help="print findings as JSON")
    a.add_argument("--no-anomaly", action="store_true", help="skip the unsupervised layer")
    a.add_argument("--strict", action="store_true", help="exit 2 when the verdict is FAIL (for CI/pipelines)")
    a.set_defaults(fn=cmd_analyze)

    f = sub.add_parser("fleet", help="analyse a folder of logs into a SQLite fleet store")
    f.add_argument("path")
    f.add_argument("--db", default="sortie.sqlite")
    f.set_defaults(fn=cmd_fleet)

    b = sub.add_parser("bench", help="fault-injection benchmark on a folder of logs")
    b.add_argument("path")
    b.add_argument("-n", type=int, default=25, help="trials per fault and flight")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--out", help="folder for trials.csv and sensitivity.csv")
    b.add_argument("--no-anomaly", action="store_true")
    b.set_defaults(fn=cmd_bench)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
