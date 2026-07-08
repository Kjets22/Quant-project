"""
last_week_test.py — a true forward test of the FROZEN strategy on last week only.

Pulls fresh 5-min bars (through today) from Polygon, stitches them onto the cache,
retrains the frozen config (base+S/R, 1:1 bracket, top-7% selectivity, 3 bps) on
everything BEFORE last week, then walks it forward over the last ~7 days and prints
the actual trade blotter. One week = a handful of trades = NOISE, not validation —
this is a "watch it run live" sanity check, not evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import lightgbm as lgb

from triple_barrier_ml import atr, features, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features
from basket import ticker_cfg
from data import fetch_polygon

COST_BPS = 3.0
SEL_Q = 0.93        # top ~7% conviction (threshold from TRAIN only)
TP, SL = 1.5, 1.0
HBAR = 24
FWD_DAYS = 7        # "last week"
FETCH_START = "2026-05-20"
FETCH_END = "2026-06-28"


def fetch_recent(tk):
    """Fresh 5-min bars for the recent window, cached to avoid re-hitting the API."""
    p = Path(f"data_cache/{tk}_recent_{FETCH_START}_{FETCH_END}.csv")
    if p.exists():
        return pd.read_csv(p, parse_dates=["timestamp"])
    cfg = ticker_cfg(tk)
    cfg.data.start_date = FETCH_START
    cfg.data.end_date = FETCH_END
    cfg.data.multiplier = 5
    cfg.data.timespan = "minute"
    df = fetch_polygon(cfg)
    df.to_csv(p, index=False)
    return df


def hourly_extended(tk):
    cache = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv",
                        parse_dates=["timestamp"])
    recent = fetch_recent(tk)
    df = (pd.concat([cache, recent], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp"))
    g = df.set_index("timestamp").resample("60min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.reset_index()


def run_ticker(tk, cutoff):
    d = hourly_extended(tk)
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, TP, SL)
    fv = X.notna().all(axis=1).to_numpy()
    n = len(c)

    tr_idx = np.where(fv & np.isfinite(y) & (ts < cutoff))[0]
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=40, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr_idx], y[tr_idx].astype(int))
    thr = np.quantile(clf.predict_proba(X.iloc[tr_idx])[:, 1], SEL_Q)

    fwd = np.where(fv & (ts >= cutoff))[0]
    if len(fwd) == 0:
        return []
    proba = clf.predict_proba(X.iloc[fwd])[:, 1]
    pmap = {int(ix): float(pr) for ix, pr in zip(fwd, proba)}

    trades = []
    i, last = int(fwd[0]), int(fwd[-1])
    while i <= last:
        if pmap.get(i, -1.0) < thr:
            i += 1
            continue
        a = A[i]
        up, dn = c[i] + TP * a, c[i] - SL * a
        res, j = None, i + 1
        while j < min(i + HBAR + 1, n):
            if l[j] <= dn:
                res = 0; break
            if h[j] >= up:
                res = 1; break
            j += 1
        open_trade = False
        if res is None:
            if j >= n:                       # ran out of data -> still open
                open_trade = True
                exit_j = n - 1
            else:
                res = 1 if c[min(j, n - 1)] > c[i] else 0
                exit_j = min(j, n - 1)
        else:
            exit_j = j
        ret = (np.nan if open_trade else
               ((a if res == 1 else -a) / c[i] - COST_BPS / 1e4))
        trades.append({
            "tk": tk, "entry": str(pd.to_datetime(ts[i])), "entry_px": round(float(c[i]), 2),
            "target": round(float(up), 2), "stop": round(float(dn), 2),
            "exit": str(pd.to_datetime(ts[exit_j])), "bars": int(exit_j - i),
            "outcome": ("OPEN" if open_trade else ("TARGET" if res == 1 else "STOP")),
            "ret": (None if open_trade else round(float(ret) * 100, 2)),
        })
        i = (exit_j if open_trade else j) + 1
    return trades


def main():
    # cutoff = 7 calendar days before the latest bar we can fetch
    latest = None
    for tk in TICKERS:
        try:
            d = hourly_extended(tk)
            mx = pd.to_datetime(d["timestamp"]).max()
            latest = mx if latest is None else max(latest, mx)
        except Exception as e:
            print(f"  [fetch warn] {tk}: {e}")
    if latest is None:
        print("No data could be fetched."); return
    cutoff = (latest.normalize() - pd.Timedelta(days=FWD_DAYS))
    print(f"latest bar available: {latest}")
    print(f"forward (last-week) window: {cutoff.date()} .. {latest.date()}\n")

    all_trades = []
    for tk in TICKERS:
        try:
            all_trades += run_ticker(tk, np.datetime64(cutoff))
        except Exception as e:
            print(f"  [skip] {tk}: {e}")
    all_trades.sort(key=lambda t: t["entry"])

    if not all_trades:
        print("No signals fired last week (top-7% conviction bar not met).")
        return

    print(f"{'ticker':>6} {'entry (UTC)':>16} {'in':>8} {'tgt':>8} {'stop':>8} "
          f"{'bars':>4} {'outcome':>7} {'ret%':>7}")
    for t in all_trades:
        entry_short = t["entry"][5:16]
        rs = "  open" if t["ret"] is None else f"{t['ret']:+.2f}"
        print(f"{t['tk']:>6} {entry_short:>16} {t['entry_px']:>8} {t['target']:>8} "
              f"{t['stop']:>8} {t['bars']:>4} {t['outcome']:>7} {rs:>7}")

    closed = [t for t in all_trades if t["ret"] is not None]
    wins = sum(t["outcome"] == "TARGET" for t in closed)
    tot = sum(t["ret"] for t in closed)
    n_open = sum(t["outcome"] == "OPEN" for t in all_trades)
    print(f"\n  closed trades = {len(closed)}  wins = {wins}  "
          f"win% = {wins/len(closed):.0%}" if closed else "\n  no closed trades")
    if closed:
        print(f"  total return (1 unit/trade) = {tot:+.2f}%   "
              f"mean/trade = {tot/len(closed):+.2f}%   open = {n_open}")
    Path("runs").mkdir(exist_ok=True)
    Path("runs/last_week.json").write_text(json.dumps(all_trades, indent=2))


if __name__ == "__main__":
    main()

