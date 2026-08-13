"""
vpt_champion.py — the vPT champion's ONE-SHOT test + full backtest proof.

Champion (fixed before this file was ever run; selection history in
runs/vx_lab_log.txt and MINDMAP 10j):
  hourly bars, ZigZag k=2.0 x ATR, tol 0.5%, DB + iHS patterns only,
  LONG only, measured-move targets, pattern-invalidation stops, H=48,
  8-ticker basket, equal $1k/trade, 5 bps costs.

Protocol: OPT 2023-01..2024-07 (searched), VAL 2024-07..2025-07 (confirmed,
Sharpe ~1.3), TEST 2025-07..now — touched exactly ONCE, by this script.
Output: per-window stats, monthly P&L, drawdown, and the equity curve
(runs/vpt_champion_curve.csv) as the working-proof artifact.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

import vpt_geometric as G
from triple_barrier_breadth import TICKERS

PATS = {"DB", "iHS"}
CFG = dict(tf=60, k=2.0, tol=0.005, H=48)
WINDOWS = [("OPT ", "2023-01-01", "2024-07-01"),
           ("VAL ", "2024-07-01", "2025-07-01"),
           ("TEST", "2025-07-01", "2099-01-01")]


def collect(lo, hi):
    rows = []
    for tk in TICKERS:
        df = G.bars(tk, CFG["tf"])
        sub = df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)] \
            .reset_index(drop=True)
        st = G.detect(sub, k=CFG["k"], tol=CFG["tol"], sides="long",
                      patterns=PATS)
        r, t, names = G.simulate(sub, st, H=CFG["H"])
        for ret, tt, nm in zip(r, t, names):
            rows.append((pd.Timestamp(tt), tk, nm, ret))
    return pd.DataFrame(rows, columns=["t", "tk", "pattern", "ret"]) \
        .sort_values("t")


def stats(df):
    s = pd.Series(df["ret"].to_numpy(), index=pd.to_datetime(df["t"]))
    d = s.resample("D").sum()
    d = d[d.index.dayofweek < 5]
    span = pd.date_range(d.index.min(), d.index.max(), freq="B")
    d = d.reindex(span, fill_value=0.0)
    shp = float(d.mean() / d.std() * np.sqrt(252)) if d.std() else 0
    eq = (d * 1000).cumsum()
    peak = eq.cummax()
    mdd = float((peak - eq).max())
    r = df["ret"].to_numpy()
    return dict(shp=shp, tot=float(r.sum() * 100), n=len(r),
                win=float((r > 0).mean()), avg_bp=float(r.mean() * 1e4),
                mdd=mdd, daily=d, eq=eq)


def main():
    curves = {}
    print("vPT CHAMPION — hourly DB+iHS long, 8 tickers, $1k/trade, 5bps\n")
    print(f"{'window':>6} {'Sharpe':>7} {'total':>8} {'n':>5} {'win%':>5} "
          f"{'avg':>7} {'maxDD($1k/trade)':>17}")
    for name, lo, hi in WINDOWS:
        df = collect(lo, hi)
        st = stats(df)
        curves[name.strip()] = st
        print(f"{name:>6} {st['shp']:>7.2f} {st['tot']:>+7.1f}% {st['n']:>5} "
              f"{st['win']:>5.0%} {st['avg_bp']:>+6.1f}bp {st['mdd']:>13,.0f}")
    t = curves["TEST"]
    print("\nTEST monthly P&L ($ at $1k/trade):")
    mo = t["daily"].resample("ME").sum() * 1000
    for ts, x in mo.items():
        print(f"  {ts:%Y-%m}: {x:+9.2f}")
    pd.concat({k: v["eq"] for k, v in curves.items()}, axis=0) \
        .to_csv("runs/vpt_champion_curve.csv")
    print("\nequity curves -> runs/vpt_champion_curve.csv")


if __name__ == "__main__":
    main()
