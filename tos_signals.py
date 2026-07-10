"""
tos_signals.py — the thinkorswim ORDER SHEET.

Runs the five frozen strategies on the latest data and prints exactly what to enter into
thinkorswim paperMoney: ticker, shares (for a $1,000 position), entry reference, target,
stop, and the time-exit deadline. Enter each as a bracket order (BUY + "1st trgs OCO":
SELL LMT at target / SELL STP at stop) — see TOS_GUIDE.md.

Shows POSITIONS TO BE HOLDING NOW (signals still open) with an actionability check:
  ENTER  — price is still near the signal entry (within 30% of the way to target)
  LATE   — price has run; skip it (the geometry is gone)
  MANAGE — you should already be in from a prior day; just keep the bracket working

  python tos_signals.py            # scans the last 7 days, prints the sheet
Research/paper tool — it prints orders for YOU to place in paperMoney. It never trades.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from paper_ensemble_v3 import candidates, CONFIGS, _C5

NOTIONAL = 1_000.0
LOOKBACK_DAYS = 7


def main():
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    fetch_end = str(end.date())
    rows = []
    for cfg in CONFIGS:
        rows += [t for t in candidates(*cfg, start, end, fetch_end) if t["outcome"] == "OPEN"]
        print(f"  [{cfg[0]} scanned]", flush=True)
    if not rows:
        print("\nNo open signals right now. Re-run after the next session.")
        return
    rows.sort(key=lambda t: t["ets"], reverse=True)
    today = pd.Timestamp.today().normalize()
    hbar_hours = {"v3": 12, "v4": 6, "v6": 96, "v7": 96, "vC": 96}   # clock in TRADING hours

    print(f"\n=== thinkorswim ORDER SHEET — {today.date()} ===")
    print("Enter each as: BUY shares w/ '1st trgs OCO' -> SELL LMT @target + SELL STP @stop (GTC)\n")
    print(f"{'strat':>5} {'tk':>5} {'signal time':>16} {'shares':>7} {'entry ref':>10} "
          f"{'TARGET':>9} {'STOP':>9} {'mark':>9} {'action':>7} {'time-exit after':>16}")
    for t in rows:
        mark = float(_C5[t["tk"]]["close"].iloc[-1])
        sh = round(NOTIONAL / t["entry"], 1)
        fresh_today = t["ets"].normalize() >= today - pd.Timedelta(days=1)
        progress = (mark - t["entry"]) / (t["tgt"] - t["entry"]) if t["tgt"] > t["entry"] else 1
        if mark <= t["stop"]:
            action = "SKIP"                       # already through the stop
        elif fresh_today and progress < 0.30:
            action = "ENTER"
        elif fresh_today:
            action = "LATE"
        else:
            action = "MANAGE"
        print(f"{t['strat']:>5} {t['tk']:>5} {str(t['ets'])[5:16]:>16} {sh:>7} "
              f"{t['entry']:>10.2f} {t['tgt']:>9.2f} {t['stop']:>9.2f} {mark:>9.2f} "
              f"{action:>7} {'~'+str(hbar_hours[t['strat']])+'h trading':>16}")
    print("\nRules: ENTER only rows marked ENTER (price still near signal). MANAGE = keep the")
    print("bracket working if already in. If neither leg fills by the time-exit, close manually")
    print("at market. $1,000/position, max 10 per strategy account. Paper only.")


if __name__ == "__main__":
    main()
