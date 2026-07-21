"""
vm_universe.py — exact vM params (two-sided 25-min ORB, rr=2, NR<=0.3, entries<=11:30,
noon exit, 2bps) applied unchanged to every cached ticker. Per ticker: gate/final ladder
windows, full-period total, win rate, monthly Sharpe, maxDD. Plus an equal-weight
portfolio of all final-positive index ETFs and of the full universe.
QQQ was the development ticker (in-sample-ish); SPY/IWM were already checked; the rest
are fresh. Read this as a transfer test, not new validation.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from morning_qqq import COST, SUBS, GATE, FINAL, days, window, run
from morning_qqq3 import add_nr
from morning_qqq4 import orb2s
from morning_validate import load_ticker, CHAMP

TICKERS = ["QQQ", "SPY", "IWM", "AAPL", "MSFT", "NVDA", "META",
           "JPM", "XLE", "XOM", "TLT", "GLD", "KO"]


def champ(day):
    return orb2s(day, **CHAMP)


def monthly_sharpe(rets):
    m = rets.resample("ME").sum()
    return float(m.mean() / m.std() * np.sqrt(12)) if len(m) > 6 and m.std() > 0 else float("nan")


def main():
    rows, series = [], {}
    for tk in TICKERS:
        try:
            dd = days(load_ticker(tk))
        except FileNotFoundError:
            print(f"  {tk}: no data file, skipped")
            continue
        add_nr(dd)
        trades = [(d["date"], champ(d)) for d in dd]
        trades = [(dt, r) for dt, r in trades if r is not None]
        if len(trades) < 30:
            print(f"  {tk}: too few trades ({len(trades)}), skipped")
            continue
        rets = pd.Series([r for _, r in trades],
                         index=pd.to_datetime([dt for dt, _ in trades]))
        series[tk] = rets
        g = run(window(dd, *GATE), champ)
        f = run(window(dd, *FINAL), champ)
        arena_worst = min(
            (float(run(window(dd, lo, hi), champ).sum() * 100)
             if len(run(window(dd, lo, hi), champ)) >= 8 else float("nan"))
            for lo, hi in SUBS)
        eq = (1 + rets).cumprod()
        mdd = float((eq / eq.cummax() - 1).min())
        rows.append({
            "tk": tk, "n": len(rets), "win": float((rets > 0).mean()),
            "total": float(rets.sum() * 100), "avg_bp": float(rets.mean() * 1e4),
            "sharpe": monthly_sharpe(rets), "maxDD": mdd,
            "arena_worst": arena_worst,
            "gate": float(g.sum() * 100) if len(g) >= 10 else float("nan"),
            "final": float(f.sum() * 100) if len(f) >= 10 else float("nan"),
        })

    print("=== vM (unchanged params) across the universe | 2bps | full period "
          "2021-06..now ===")
    print(f"  {'tk':<5} {'n':>4} {'win':>5} {'avg':>7} {'total':>8} {'Sharpe':>7} "
          f"{'maxDD':>7} {'arenaW':>8} {'gate':>7} {'final':>7}")
    for r in sorted(rows, key=lambda x: -x["final"] if np.isfinite(x["final"]) else 99):
        print(f"  {r['tk']:<5} {r['n']:>4} {r['win']:>5.0%} {r['avg_bp']:>+6.1f}bp "
              f"{r['total']:>+7.2f}% {r['sharpe']:>7.2f} {r['maxDD']:>7.2%} "
              f"{r['arena_worst']:>+7.2f}% {r['gate']:>+6.2f}% {r['final']:>+6.2f}%")

    # ---- portfolios: equal-weight daily average of member returns ----
    def portfolio(name, members):
        mem = {t: series[t] for t in members if t in series}
        if len(mem) < 2:
            return
        df = pd.DataFrame({t: s.groupby(s.index).sum() for t, s in mem.items()})
        port = df.mean(axis=1, skipna=True)          # equal weight across active names
        port = port[df.notna().any(axis=1)]
        m = port.resample("ME").sum()
        sh = float(m.mean() / m.std() * np.sqrt(12)) if m.std() > 0 else float("nan")
        eq = (1 + port).cumprod()
        mdd = float((eq / eq.cummax() - 1).min())
        print(f"  {name:<28} days={len(port):>4} total={port.sum()*100:+7.2f}% "
              f"Sharpe={sh:5.2f} maxDD={mdd:7.2%}")

    print("\n=== equal-weight portfolios (daily avg of active members) ===")
    portfolio("INDEX ETFs (QQQ+SPY)", ["QQQ", "SPY"])
    portfolio("ALL final-positive tickers",
              [r["tk"] for r in rows if np.isfinite(r["final"]) and r["final"] > 0])
    portfolio("FULL universe", [r["tk"] for r in rows])


if __name__ == "__main__":
    main()
