"""
sharpe_paper.py — Sharpe per strategy from the live paper ledger (runs/paper_v4_ledger.json).

Builds each account's DAILY realized P&L series over the paper window (business days,
zero-filled for flat days), converts to daily returns on the $10,000 account, and
annualizes: Sharpe = mean/std * sqrt(252). ALSO reports the number of daily observations —
with ~1-2 weeks of data these Sharpes are noisy estimates, not statistics.
"""

from __future__ import annotations

import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

ACCOUNT = 10_000.0
led = json.load(open("runs/paper_v4_ledger.json"))

start = min(pd.Timestamp(t["ets"]) for v in led.values() for t in v).normalize()
end = max(pd.Timestamp(t["xts"]) for v in led.values() for t in v
          if t["xts"] is not None).normalize()
days = pd.bdate_range(start, end)

print(f"PAPER SHARPE — window {start.date()} .. {end.date()} ({len(days)} trading days)\n")
print(f"  {'strat':>6} {'days':>5} {'realized$':>10} {'unreal$':>9} {'daily mean$':>12} "
      f"{'daily std$':>11} {'Sharpe(ann)':>12}")
rows = []
for name, trades in led.items():
    daily = pd.Series(0.0, index=days)
    unreal = 0.0
    for t in trades:
        if t["pnl"] is not None and t["xts"] is not None:
            d = pd.Timestamp(t["xts"]).normalize()
            if d in daily.index:
                daily[d] += t["pnl"]
        elif t.get("unreal") is not None:
            unreal += t["unreal"]
    r = daily / ACCOUNT
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else float("nan")
    rows.append(daily)
    print(f"  {name:>6} {len(days):>5} {daily.sum():>+10.2f} {unreal:>+9.2f} "
          f"{daily.mean():>+12.2f} {daily.std():>11.2f} {sharpe:>12.2f}")
comb = sum(rows)
rc = comb / (ACCOUNT * len(led))
sh = rc.mean() / rc.std() * np.sqrt(252) if rc.std() > 0 else float("nan")
print(f"  {'ALL':>6} {len(days):>5} {comb.sum():>+10.2f} {'':>9} {comb.mean():>+12.2f} "
      f"{comb.std():>11.2f} {sh:>12.2f}")
print("\n  CAVEAT: computed from only ~2 weeks of daily P&L — direction is meaningful,")
print("  the precise numbers are noise. The reliable Sharpes are the multi-year backtests.")
