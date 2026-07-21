"""
morning_validate.py — VALIDATION of the round-4 ladder survivor:
  CHAMPION "vM": QQQ two-sided 25-min ORB, target 2 x risk, stop = other side of the
  opening range, NR filter (prior-day range in bottom 30% of trailing 7 days),
  entries until 11:30, hard exit 12:00, 2bps.

Three tests (no re-selection — the champion stays the champion regardless of outcome):
  1. NEIGHBORHOOD map: 12 pre-registered neighbors (rr, NR cutoff, OR length, deadline)
     shown on arena/gate/final. A real edge sits on a plateau, not a spike.
  2. FRESH TICKERS: exact champion params on SPY and IWM — never used in any morning
     round. Structural edges travel; overfit ones don't.
  3. Champion full history: yearly totals, equity path, maxDD, monthly Sharpe.
"""

from __future__ import annotations

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from morning_qqq import COST, SUBS, GATE, FINAL, load, days, window, run, OPEN
from morning_qqq3 import add_nr
from morning_qqq4 import orb2s

_ET = ZoneInfo("America/New_York")


def load_ticker(tk):
    base = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv",
                       parse_dates=["timestamp"])
    parts = [base]
    for p in sorted(Path("data_cache").glob(f"{tk}_recent_*.csv")):
        parts.append(pd.read_csv(p, parse_dates=["timestamp"]))
    df = (pd.concat(parts, ignore_index=True)
          .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
          .reset_index(drop=True))
    ts = pd.DatetimeIndex(df["timestamp"]).tz_localize("UTC").tz_convert(_ET)
    df["date"] = ts.date
    df["min"] = ts.hour * 60 + ts.minute
    return df


CHAMP = dict(or_bars=5, rr=2.0, nr=0.3)


def champ(day):
    return orb2s(day, **CHAMP)


NEIGHBORS = [
    ("rr 1.5",        dict(or_bars=5, rr=1.5, nr=0.3)),
    ("rr 3.0",        dict(or_bars=5, rr=3.0, nr=0.3)),
    ("rr None(noon)", dict(or_bars=5, rr=None, nr=0.3)),
    ("NR .25",        dict(or_bars=5, rr=2.0, nr=0.25)),
    ("NR .35",        dict(or_bars=5, rr=2.0, nr=0.35)),
    ("NR .40",        dict(or_bars=5, rr=2.0, nr=0.4)),
    ("OR 20min",      dict(or_bars=4, rr=2.0, nr=0.3)),
    ("OR 30min",      dict(or_bars=6, rr=2.0, nr=0.3)),
    ("dl 11:00",      dict(or_bars=5, rr=2.0, nr=0.3, last_entry=660)),
    ("dl 12:00",      dict(or_bars=5, rr=2.0, nr=0.3, last_entry=715)),
    ("long only",     dict(or_bars=5, rr=2.0, nr=0.3, sides="long")),
    ("short only",    dict(or_bars=5, rr=2.0, nr=0.3, sides="short")),
]


def ladder_line(dd, fn, name):
    subs = []
    for lo, hi in SUBS:
        r = run(window(dd, lo, hi), fn)
        subs.append(float(r.sum() * 100) if len(r) >= 8 else float("nan"))
    g = run(window(dd, *GATE), fn)
    f = run(window(dd, *FINAL), fn)
    gt = g.sum() * 100 if len(g) else float("nan")
    ft = f.sum() * 100 if len(f) else float("nan")
    t = (f.mean() / f.std() * np.sqrt(len(f))
         if len(f) > 5 and f.std() > 0 else float("nan"))
    print(f"  {name:<14} arena_worst={np.nanmin(subs):+6.2f}%  gate={gt:+6.2f}% "
          f"(n={len(g)})  final={ft:+6.2f}% (n={len(f)}, t={t:+.2f})")
    return ft


def main():
    df = load()
    dd = days(df)
    add_nr(dd)

    print("=== 1) NEIGHBORHOOD MAP (champion + 12 pre-registered neighbors) ===")
    ladder_line(dd, champ, "CHAMPION")
    pos = 0
    for name, kw in NEIGHBORS:
        ft = ladder_line(dd, lambda d, kw=kw: orb2s(d, **kw), name)
        pos += (not np.isnan(ft)) and ft > 0
    print(f"  -> {pos}/{len(NEIGHBORS)} neighbors final-positive")

    print("\n=== 2) FRESH TICKERS (exact champion params, never used in dev) ===")
    for tk in ("SPY", "IWM"):
        tdf = load_ticker(tk)
        tdd = days(tdf)
        add_nr(tdd)
        print(f"  {tk}:")
        ladder_line(tdd, champ, f"  {tk}")

    print("\n=== 3) CHAMPION FULL HISTORY (QQQ) ===")
    trades = [(d["date"], champ(d)) for d in dd]
    trades = [(dt, r) for dt, r in trades if r is not None]
    rets = pd.Series([r for _, r in trades],
                     index=pd.to_datetime([dt for dt, _ in trades]))
    print(f"  trades={len(rets)}  win={float((rets > 0).mean()):.1%}  "
          f"avg={rets.mean()*1e4:+.1f}bp  total={rets.sum()*100:+.2f}%")
    eq = (1 + rets).cumprod()
    dd_ser = eq / eq.cummax() - 1
    monthly = rets.resample("ME").sum()
    sharpe = monthly.mean() / monthly.std() * np.sqrt(12) if monthly.std() > 0 else 0
    print(f"  maxDD={dd_ser.min():.2%}  monthly Sharpe={sharpe:.2f}  "
          f"months+={float((monthly > 0).mean()):.0%}")
    print("  yearly:")
    for y, r in rets.groupby(rets.index.year):
        print(f"    {y}: {r.sum()*100:+6.2f}%  (n={len(r)}, win={(r > 0).mean():.0%})")


if __name__ == "__main__":
    main()
