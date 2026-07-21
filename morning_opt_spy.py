"""
morning_opt_spy.py — SPY baseline for the vM options overlay: replay vM's SPY stock
trades (exact champion params) as REAL option trades over the usable Polygon window
(2024-07-22..now). Cells: 0dte/atm, 0dte/otm1, 1-3d/atm. Same conventions as the QQQ
replay (1%/side on premium, unfillable rows excluded from P&L and reported).
Output: runs/morning_options_spy.txt
"""

from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from morning_qqq import days
from morning_qqq3 import add_nr
from morning_validate import load_ticker
from morning_opt import orb2s_trade, replay, summarize

START = "2024-07-22"
CELLS = [("0dte", "atm"), ("0dte", "otm1"), ("1-3d", "atm")]


def main():
    import pandas as pd
    dd = days(load_ticker("SPY"))
    add_nr(dd)
    s = pd.Timestamp(START).date()
    trades = []
    for d in dd:
        if d["date"] < s:
            continue
        t = orb2s_trade(d, 5, 2.0, nr=0.3, last_entry=690, sides="both")
        if t is not None:
            trades.append(t)
    stock = np.array([t["stock_ret_net"] for t in trades])
    lines = [f"SPY vM options replay — {len(trades)} trades {START}..now",
             f"stock (2bps): tot {stock.sum()*100:+.2f}%  win {(stock>0).mean():.0%}",
             ""]
    for bucket, mode in CELLS:
        res = replay(trades, bucket, mode, underlying="SPY")
        su = summarize(res)
        # era split at 2025-07-14 (gate-era vs final-era)
        import datetime as dt
        cut = dt.date(2025, 7, 14)
        f1 = [r["opt_ret"] for r in res if r["status"] == "filled" and r["date"] < cut]
        f2 = [r["opt_ret"] for r in res if r["status"] == "filled" and r["date"] >= cut]
        line = (f"{bucket}/{mode}: n={su['n']} filled={su['filled']} "
                f"(unfillable rate {su['unfillable_rate']:.0%})")
        if su["filled"]:
            line += (f" | win {su['win']:.0%} avg {su['avg_ret']*100:+.1f}% "
                     f"med {su['med_ret']*100:+.1f}% tot {su['tot_ret']*100:+.1f}%"
                     f" | era1 {np.sum(f1)*100:+.0f}% (n={len(f1)}) / "
                     f"era2 {np.sum(f2)*100:+.0f}% (n={len(f2)})")
        lines.append(line)
        print(line, flush=True)
    out = "\n".join(lines) + "\n"
    from pathlib import Path
    Path("runs/morning_options_spy.txt").write_text(out, encoding="utf-8")
    print("\nsaved runs/morning_options_spy.txt")


if __name__ == "__main__":
    main()
