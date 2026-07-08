"""
last_week_chart.py — last week's trades at 1.5:1, WITH candle data for plotting.
Trains on everything before last week, runs forward, and exports per-ticker hourly
OHLC candles + the entry/target/stop/exit of each trade to runs/last_week_chart.json.
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

SEL_Q = 0.93
TP, SL = 1.5, 1.0     # the better ratio
HBAR = 24
FWD_DAYS = 7
FETCH_START, FETCH_END = "2026-05-20", "2026-06-28"


def recent(tk):
    p = Path(f"data_cache/{tk}_recent_{FETCH_START}_{FETCH_END}.csv")
    if p.exists():
        return pd.read_csv(p, parse_dates=["timestamp"])
    cfg = ticker_cfg(tk)
    cfg.data.start_date, cfg.data.end_date = FETCH_START, FETCH_END
    cfg.data.multiplier, cfg.data.timespan = 5, "minute"
    df = fetch_polygon(cfg)
    df.to_csv(p, index=False)
    return df


def hourly_ohlc(tk):
    cache = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv",
                        parse_dates=["timestamp"])
    df = (pd.concat([cache, recent(tk)], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
    g = df.set_index("timestamp").resample("60min").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna()
    return g.reset_index()


def run_ticker(tk, cutoff):
    d = hourly_ohlc(tk)
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    o, h, l, c, v = (d[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, TP, SL)
    fv = X.notna().all(axis=1).to_numpy()
    n = len(c)
    tr = np.where(fv & np.isfinite(y) & (ts < cutoff))[0]
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr].astype(int))
    thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SEL_Q)
    fwd = np.where(fv & (ts >= cutoff))[0]
    if len(fwd) == 0:
        return None
    proba = {int(ix): float(p) for ix, p in zip(fwd, clf.predict_proba(X.iloc[fwd])[:, 1])}
    trades = []
    i, last = int(fwd[0]), int(fwd[-1])
    while i <= last:
        if proba.get(i, -1) < thr:
            i += 1; continue
        a = A[i]
        up, dn = c[i] + TP * a, c[i] - SL * a
        res, j = None, i + 1
        while j < min(i + HBAR + 1, n):
            if l[j] <= dn:
                res = 0; break
            if h[j] >= up:
                res = 1; break
            j += 1
        if res is None and j >= n:
            outcome, ex_j, ex_px = "OPEN", n - 1, c[n - 1]
        elif res == 1:
            outcome, ex_j, ex_px = "TARGET", j, up
        elif res == 0:
            outcome, ex_j, ex_px = "STOP", j, dn
        else:
            ex_j = min(j, n - 1)
            outcome, ex_px = ("TIME+" if c[ex_j] > c[i] else "TIME-"), c[ex_j]
        trades.append({
            "entry_ts": str(pd.Timestamp(ts[i])), "entry_px": round(float(c[i]), 2),
            "target": round(float(up), 2), "stop": round(float(dn), 2),
            "exit_ts": str(pd.Timestamp(ts[ex_j])), "exit_px": round(float(ex_px), 2),
            "outcome": outcome,
        })
        i = (ex_j if outcome == "OPEN" else j) + 1
    if not trades:
        return None
    # candles from a bit before the first entry to a bit after the last exit
    win = d[pd.to_datetime(d["timestamp"]) >= (cutoff - np.timedelta64(0, "D"))]
    candles = [[str(pd.Timestamp(r.timestamp)), round(r.open, 2), round(r.high, 2),
                round(r.low, 2), round(r.close, 2)] for r in win.itertuples()]
    return {"candles": candles, "trades": trades}


def main():
    latest = max(pd.to_datetime(hourly_ohlc(tk)["timestamp"]).max() for tk in TICKERS)
    cutoff = latest.normalize() - pd.Timedelta(days=FWD_DAYS)
    print(f"forward window {cutoff.date()} .. {latest.date()}  (target:stop = {TP}:{SL})\n")
    out = {}
    for tk in TICKERS:
        try:
            r = run_ticker(tk, np.datetime64(cutoff))
        except Exception as e:
            print(f"  [skip {tk}] {e}"); continue
        if r:
            out[tk] = r
            for t in r["trades"]:
                print(f"  {tk:>5} {t['entry_ts'][5:16]} buy {t['entry_px']:>8}  "
                      f"tgt {t['target']:>8}  stop {t['stop']:>8}  -> {t['outcome']:>6} "
                      f"@ {t['exit_ts'][5:16]} ({t['exit_px']})")
    Path("runs").mkdir(exist_ok=True)
    Path("runs/last_week_chart.json").write_text(json.dumps(out, indent=1))
    print(f"\n  tickers with trades: {list(out.keys())}")
    print("  saved runs/last_week_chart.json")


if __name__ == "__main__":
    main()
