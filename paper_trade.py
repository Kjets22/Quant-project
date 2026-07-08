"""
paper_trade.py — live paper-trading engine for the frozen 30-min / 1.5:1 strategy.

Run it any time (e.g. once after each trading day this week). Each run:
  1. pulls the latest 5-min bars from Polygon, stitches onto the cache, resamples to 30 min,
  2. trains the model on everything BEFORE the paper start date (frozen, no look-ahead),
  3. walks the model forward over the paper window, opening 1.5:1 bracket paper-trades on
     top-7% signals and closing them at target / stop / time,
  4. prints the blotter + running P&L and writes runs/paper_ledger.json.

It is idempotent: it recomputes the whole week from the frozen model + latest data each run,
so just re-run it to refresh. Standalone — touches no frozen snapshot or working file.
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

PAPER_START = pd.Timestamp("2026-06-29")     # this coming week
MIN = 30
TP, SL = 1.5, 1.0
SEL_Q = 0.93
HBAR = 24
COST_BPS = 3.0
FETCH_START = "2026-06-01"


def recent(tk, end):
    cfg = ticker_cfg(tk)
    cfg.data.start_date, cfg.data.end_date = FETCH_START, end
    cfg.data.multiplier, cfg.data.timespan = 5, "minute"
    return fetch_polygon(cfg)


def bars30(tk, end):
    cache = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv",
                        parse_dates=["timestamp"])
    df = (pd.concat([cache, recent(tk, end)], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
    g = df.set_index("timestamp").resample(f"{MIN}min").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna()
    return g.reset_index()


def process(tk, end):
    d = bars30(tk, end)
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, TP, SL)
    fv = X.notna().all(axis=1).to_numpy()
    n = len(c)
    start = np.datetime64(PAPER_START)
    tr = np.where(fv & np.isfinite(y) & (ts < start))[0]
    if len(tr) < 500:
        return []
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr].astype(int))
    thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SEL_Q)
    fwd = np.where(fv & (ts >= start))[0]
    if len(fwd) == 0:
        return []
    proba = {int(ix): float(p) for ix, p in zip(fwd, clf.predict_proba(X.iloc[fwd])[:, 1])}
    trades = []
    i, last = int(fwd[0]), int(fwd[-1])
    while i <= last:
        if proba.get(i, -1) < thr:
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
        if res is None and j >= n:                       # ran out of data -> still open
            outcome, ex_j, ex_px, ret = "OPEN", n - 1, c[n - 1], None
        else:
            if res is None:                              # full horizon elapsed -> time exit
                res = 1 if c[min(j, n - 1)] > c[i] else 0
            ex_j = j if j < n else n - 1
            outcome = "TARGET" if res == 1 else "STOP"
            ex_px = up if res == 1 else dn
            ret = ((TP if res == 1 else -SL) * a) / c[i] * 100 - COST_BPS / 100
        trades.append({
            "ticker": tk, "entry_ts": str(pd.Timestamp(ts[i])), "entry": round(float(c[i]), 2),
            "target": round(float(up), 2), "stop": round(float(dn), 2),
            "exit_ts": str(pd.Timestamp(ts[ex_j])), "exit": round(float(ex_px), 2),
            "outcome": outcome, "ret_pct": (None if ret is None else round(float(ret), 2)),
        })
        i = (ex_j if outcome == "OPEN" else j) + 1
    return trades


def main():
    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"PAPER TRADING  |  30-min / 1.5:1 / top-7% / 3 bps  |  8-name basket")
    print(f"week starts {PAPER_START.date()}  |  data through {end}\n")
    trades = []
    for tk in TICKERS:
        try:
            trades += process(tk, end)
        except Exception as e:
            print(f"  [skip {tk}] {e}")
    trades.sort(key=lambda t: t["entry_ts"])
    if not trades:
        print("No paper trades yet — no top-7% signal has fired since the week opened.")
        print("Re-run after more 30-min bars print (e.g. end of each trading day).")
        return
    print(f"{'tk':>5} {'entry (UTC)':>16} {'in':>8} {'tgt':>8} {'stop':>8} {'outcome':>7} {'ret%':>7}")
    for t in trades:
        rs = "  open" if t["ret_pct"] is None else f"{t['ret_pct']:+.2f}"
        print(f"{t['ticker']:>5} {t['entry_ts'][5:16]:>16} {t['entry']:>8} {t['target']:>8} "
              f"{t['stop']:>8} {t['outcome']:>7} {rs:>7}")
    closed = [t for t in trades if t["ret_pct"] is not None]
    wins = sum(t["outcome"] == "TARGET" for t in closed)
    total = sum(t["ret_pct"] for t in closed)
    n_open = sum(t["outcome"] == "OPEN" for t in trades)
    print()
    if closed:
        print(f"  closed={len(closed)}  wins={wins}  win%={wins/len(closed):.0%}  "
              f"total (1 unit/trade)={total:+.2f}%  mean/trade={total/len(closed):+.2f}%  open={n_open}")
    else:
        print(f"  closed=0  open={n_open}  (positions live, no exits yet)")
    led = {
        "config": "30min 1.5:1 top-7% base+S/R, 8 names, 3 bps",
        "paper_start": str(PAPER_START.date()), "as_of": end,
        "open": [t for t in trades if t["outcome"] == "OPEN"],
        "closed": closed,
        "summary": {"closed": len(closed), "wins": wins,
                    "win_pct": round(wins / len(closed) * 100, 1) if closed else None,
                    "total_ret_pct": round(total, 2) if closed else 0.0,
                    "open": n_open},
    }
    Path("runs").mkdir(exist_ok=True)
    Path("runs/paper_ledger.json").write_text(json.dumps(led, indent=2))
    print("  ledger -> runs/paper_ledger.json")


if __name__ == "__main__":
    main()
