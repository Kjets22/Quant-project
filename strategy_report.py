"""
strategy_report.py — per-strategy P&L on demand: realized + open marks + the
hypothetical outage/old-rule credits, reconciled against broker equity.

  python strategy_report.py            full report
  python strategy_report.py --today    add a "today" column

Read-only; places no orders. (Lives in the repo rather than a scratch dir because
this is the question that gets asked most.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

import alpaca_api as broker

STOCK = ["v3", "v4", "v6", "v7", "vC", "vQ", "vQ2", "vA", "vP", "vR", "vS", "vM"]
OPTS = ["vC-OPT-2W", "vM-OPT-0DTE", "v6-OPT-6W"]
LEGACY = {"vCO": "vC-OPT-2W", "vMO": "vM-OPT-0DTE"}
START = 100_000.0


def oname(x):
    s = x.get("strat", "vCO")
    return LEGACY.get(s, s)


def main():
    led = json.loads(Path("runs/alpaca2_ledger.json").read_text())
    rf = Path("runs/recovered_trades.json")
    rec = json.loads(rf.read_text()) if rf.exists() else []
    pos = broker.positions()
    spx = {p["symbol"]: float(p["current_price"]) for p in pos}
    opl = {p["symbol"]: float(p.get("unrealized_pl") or 0) for p in pos}
    today = str(pd.Timestamp.now(tz="America/New_York").date())

    rows = []
    for st in STOCK:
        cl = [x for x in led["closed"]
              if x["strat"] == st and x["outcome"] != "MISSED"]
        op = [x for x in led["open"] if x["strat"] == st]
        r = sum(x.get("pnl") or 0 for x in cl)
        tod = sum(x.get("pnl") or 0 for x in cl if (x.get("xts") or "") >= today)
        u = 0.0
        for x in op:
            sign = -1 if x.get("side") == "short" else 1
            u += sign * (spx.get(x["tk"], x["fill"]) - x["fill"]) * x["qty"]
        w = sum(1 for x in cl if (x.get("pnl") or 0) > 0)
        rows.append([st, r, u, r + u, len(cl), w, len(op), tod, op])
    for on in OPTS:
        oc = [x for x in led.get("opt_closed", []) if oname(x) == on]
        oo = [x for x in led.get("opt_open", []) if oname(x) == on]
        if not (oc or oo):
            continue
        r = sum(x.get("pnl") or 0 for x in oc)
        tod = sum(x.get("pnl") or 0 for x in oc if (x.get("xts") or "") >= today)
        u = sum(opl.get(x["occ"], 0.0) for x in oo)
        w = sum(1 for x in oc if (x.get("pnl") or 0) > 0)
        rows.append([on, r, u, r + u, len(oc), w, len(oo), tod, oo])

    rows.sort(key=lambda z: -z[3])
    show_today = "--today" in sys.argv
    hdr = (f"{'strategy':>12} {'realized':>10} {'open':>10} {'TOTAL':>10} "
           f"{'closed':>7} {'win%':>6} {'open':>5}")
    if show_today:
        hdr += f" {'today':>9}"
    print(hdr)
    print("-" * (len(hdr) + 2))
    tr = tu = 0.0
    for st, r, u, t, n, w, no, tod, _ in rows:
        tr += r
        tu += u
        line = (f"{st:>12} {r:>+10.2f} {u:>+10.2f} {t:>+10.2f} {n:>7} "
                f"{(f'{w / n * 100:.0f}%' if n else '-'):>6} {no:>5}")
        if show_today:
            line += f" {tod:>+9.2f}"
        print(line)
    print("-" * (len(hdr) + 2))
    print(f"{'ALL':>12} {tr:>+10.2f} {tu:>+10.2f} {tr + tu:>+10.2f}")

    print("\nopen positions by strategy:")
    any_open = False
    for st, r, u, t, n, w, no, tod, op in rows:
        for x in op:
            any_open = True
            if "occ" in x:
                print(f"  {st:>12}  {x['occ']} x{x['qty']} @ {x.get('fill')} "
                      f"mark {opl.get(x['occ'], 0):+.2f}")
            else:
                sign = -1 if x.get("side") == "short" else 1
                pl = sign * (spx.get(x["tk"], x["fill"]) - x["fill"]) * x["qty"]
                side = "SHORT" if x.get("side") == "short" else "long "
                print(f"  {st:>12}  {x['style']} {side} {x['tk']:>5} {x['qty']:>3}sh "
                      f"@ {x['fill']:>8.2f}  {pl:>+8.2f}")
    if not any_open:
        print("  (none)")

    if rec:
        by = {}
        for x in rec:
            by[x["strat"]] = by.get(x["strat"], 0) + x["pnl"]
        tot = sum(by.values())
        print(f"\nhypothetical credits (outage / old cross-block misses), "
              f"NOT in the balance: {tot:+.2f}")
        print("  " + "  ".join(f"{k} {v:+.2f}" for k, v in sorted(by.items())))

    a = broker.account()
    eq = float(a["equity"])
    print(f"\nbroker equity ${eq:,.2f} | TOTAL P&L incl. open {eq - START:+,.2f} "
          f"| today {eq - float(a['last_equity']):+,.2f} "
          f"| halted {led['state']['halted']}")


if __name__ == "__main__":
    main()

