"""
morning_hf_validate.py — VALIDATION of the morning_hf.py F1 sweep champion:
  "O4 Rn nr.7 e660": orb2s(or_bars=4, rr=None, nr=0.7, last_entry=660, sides='both')
  20-min OR, first close beyond either extreme by 11:00 enters (stop = far extreme,
  risk cap 1.2%), NO target (noon exit), prior-day NR rank <= 0.7, 2bps round trip.

No re-selection — the champion stays the champion regardless of outcome.
  1. NEIGHBORHOOD map: 12 pre-registered neighbors (each param nudged one step,
     both directions where meaningful) on arena_worst/gate/final.
  2. FRESH TICKERS: exact params on SPY and IWM (never used in the HF sweep).
  3. FULL HISTORY on QQQ: yearly totals 2021-2026, maxDD, monthly Sharpe,
     final at 0/2/5bps.
  4. FREQUENCY CHECK: recompute trade-day fraction overall/gate/final vs the
     sweep's claim (floor 0.48).

Usage:  python morning_hf_validate.py
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

from morning_qqq import COST, SUBS, GATE, FINAL, load, days, window, run
from morning_qqq3 import add_nr
from morning_qqq4 import orb2s

_ET = ZoneInfo("America/New_York")
OVERALL = ("2022-01-14", "2099-01-01")     # frequency span used by morning_hf.py
FREQ_MIN = 0.48


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


CHAMP = dict(or_bars=4, rr=None, nr=0.7, last_entry=660, sides="both")


def champ(day):
    return orb2s(day, **CHAMP)


# 12 pre-registered neighbors: each param nudged one step on the sweep's own grid
# (or_bars 3/4/5/6, rr 1.5/2/3/None, nr None/.5/.7/.85, last_entry 660/690),
# both directions where meaningful; sides long/short as the structural nudge.
NEIGHBORS = [
    ("or_bars 3",   dict(or_bars=3, rr=None, nr=0.7, last_entry=660)),
    ("or_bars 5",   dict(or_bars=5, rr=None, nr=0.7, last_entry=660)),
    ("or_bars 6",   dict(or_bars=6, rr=None, nr=0.7, last_entry=660)),
    ("rr 2.0",      dict(or_bars=4, rr=2.0, nr=0.7, last_entry=660)),
    ("rr 3.0",      dict(or_bars=4, rr=3.0, nr=0.7, last_entry=660)),
    ("nr .5",       dict(or_bars=4, rr=None, nr=0.5, last_entry=660)),
    ("nr .85",      dict(or_bars=4, rr=None, nr=0.85, last_entry=660)),
    ("nr None",     dict(or_bars=4, rr=None, nr=None, last_entry=660)),
    ("dl 10:30",    dict(or_bars=4, rr=None, nr=0.7, last_entry=630)),
    ("dl 11:30",    dict(or_bars=4, rr=None, nr=0.7, last_entry=690)),
    ("long only",   dict(or_bars=4, rr=None, nr=0.7, last_entry=660, sides="long")),
    ("short only",  dict(or_bars=4, rr=None, nr=0.7, last_entry=660, sides="short")),
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
    print(f"  {name:<12} arena_worst={np.nanmin(subs):+6.2f}% "
          f"subs={[round(s, 1) for s in subs]} gate={gt:+6.2f}% (n={len(g)})  "
          f"final={ft:+6.2f}% (n={len(f)}, t={t:+.2f})")
    return ft, gt


def freq_over(dd, lo, hi):
    w = window(dd, lo, hi)
    hits = sum(champ(d) is not None for d in w)
    return hits, len(w), (hits / len(w) if w else 0.0)


def main():
    df = load()
    dd = days(df)
    add_nr(dd)

    print("=== 1) NEIGHBORHOOD MAP (champion + 12 pre-registered neighbors) ===")
    ladder_line(dd, champ, "CHAMPION")
    pos = 0
    fts = []
    for name, kw in NEIGHBORS:
        ft, _ = ladder_line(dd, lambda d, kw=kw: orb2s(d, **kw), name)
        fts.append((name, ft))
        pos += (not np.isnan(ft)) and ft > 0
    print(f"  -> {pos}/{len(NEIGHBORS)} neighbors final-positive "
          f"(range {min(f for _, f in fts):+.2f}% .. {max(f for _, f in fts):+.2f}%)")

    print("\n=== 2) FRESH TICKERS (exact champion params) ===")
    for tk in ("SPY", "IWM"):
        tdf = load_ticker(tk)
        tdd = days(tdf)
        add_nr(tdd)
        print(f"  {tk}: {len(tdd)} days {tdd[0]['date']}..{tdd[-1]['date']}")
        ladder_line(tdd, champ, f"  {tk}")

    print("\n=== 3) CHAMPION FULL HISTORY (QQQ, net 2bps) ===")
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
    print("  yearly (2bps):")
    for y, r in rets.groupby(rets.index.year):
        print(f"    {y}: {r.sum()*100:+6.2f}%  (n={len(r)}, win={(r > 0).mean():.0%}, "
              f"avg={r.mean()*1e4:+.1f}bp)")
    f = run(window(dd, *FINAL), champ)
    f0, f5 = f + COST, f + COST - 5e-4
    tstat = f.mean() / f.std() * np.sqrt(len(f)) if f.std() > 0 else 0.0
    print(f"  FINAL {FINAL[0]}..now: n={len(f)} win={(f > 0).mean():.0%} "
          f"t={tstat:+.2f} | 0bps {f0.sum()*100:+.2f}% / 2bps {f.sum()*100:+.2f}% / "
          f"5bps {f5.sum()*100:+.2f}%")

    print("\n=== 4) FREQUENCY CHECK (trade-day fraction, floor "
          f"{FREQ_MIN:.2f}) ===")
    claims = {"overall": 0.62, "gate": 0.65, "final": 0.61}
    spans = [("overall", OVERALL), ("gate", GATE), ("final", FINAL)]
    all_ok = True
    for tag, (lo, hi) in spans:
        hits, n, fq = freq_over(dd, lo, hi)
        ok = fq >= FREQ_MIN
        all_ok &= ok
        print(f"  {tag:<8} {lo}..{hi if hi != '2099-01-01' else 'now'}: "
              f"{hits}/{n} = {fq:.3f}  (sweep claimed {claims[tag]:.2f})  "
              f"{'floor-OK' if ok else 'BELOW FLOOR'}")
    print(f"  -> frequency floor {'CONFIRMED' if all_ok else 'REFUTED'}")


if __name__ == "__main__":
    main()
