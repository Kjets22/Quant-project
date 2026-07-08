"""
last_week_all.py — how did each validated config trade LAST WEEK (Mon 2026-06-22 .. Fri 06-26)?
For every config it trains a frozen model on all data BEFORE last week, then walks it forward
over last week (non-overlapping bracket trades), and reports trades / win% / P&L side by side.
Standalone; touches no frozen snapshot.
"""

from __future__ import annotations

import sys

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
HBAR = 24
COST_BPS = 3.0
LWS = pd.Timestamp("2026-06-22")          # last-week start (Monday)
LWE = pd.Timestamp("2026-06-29")          # this Monday (exclusive end)
FETCH_START, FETCH_END = "2026-05-20", "2026-06-28"
CONFIGS = [("v1", 60, 1.0, 1.0), ("v2", 60, 1.5, 1.0),
           ("v3", 30, 1.5, 1.0), ("v4", 15, 4.0, 1.0)]
_CACHE5 = {}


def load5(tk):
    if tk in _CACHE5:
        return _CACHE5[tk]
    cache = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    from pathlib import Path
    p = Path(f"data_cache/{tk}_recent_{FETCH_START}_{FETCH_END}.csv")
    if p.exists():
        rec = pd.read_csv(p, parse_dates=["timestamp"])
    else:
        cfg = ticker_cfg(tk); cfg.data.start_date, cfg.data.end_date = FETCH_START, FETCH_END
        cfg.data.multiplier, cfg.data.timespan = 5, "minute"
        rec = fetch_polygon(cfg); rec.to_csv(p, index=False)
    df = (pd.concat([cache, rec], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
    _CACHE5[tk] = df
    return df


def resample(df, mins):
    g = df.set_index("timestamp").resample(f"{mins}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.reset_index()


def run_config(mins, tp, sl):
    trades = []
    for tk in TICKERS:
        try:
            d = resample(load5(tk), mins)
        except Exception as e:
            print(f"  [skip {tk}] {e}"); continue
        ts = pd.to_datetime(d["timestamp"]).to_numpy()
        h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
        A = atr(h, l, c)
        X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                       sr_features(d).reset_index(drop=True)], axis=1)
        y = label(h, l, c, A, tp, sl)
        fv = X.notna().all(axis=1).to_numpy()
        n = len(c)
        tr = np.where(fv & np.isfinite(y) & (ts < np.datetime64(LWS)))[0]
        if len(tr) < 500:
            continue
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, verbose=-1)
        clf.fit(X.iloc[tr], y[tr].astype(int))
        thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SEL_Q)
        fwd = np.where(fv & (ts >= np.datetime64(LWS)) & (ts < np.datetime64(LWE)))[0]
        if len(fwd) == 0:
            continue
        proba = {int(ix): float(p) for ix, p in zip(fwd, clf.predict_proba(X.iloc[fwd])[:, 1])}
        i, last = int(fwd[0]), int(fwd[-1])
        while i <= last:
            if proba.get(i, -1) < thr:
                i += 1; continue
            a = A[i]; up, dn = c[i] + tp * a, c[i] - sl * a
            res, j = None, i + 1
            while j < min(i + HBAR + 1, n):
                if l[j] <= dn:
                    res = 0; break
                if h[j] >= up:
                    res = 1; break
                j += 1
            if res is None and j >= n:
                trades.append((tk, str(pd.Timestamp(ts[i]))[5:16], "OPEN", None)); i = n; continue
            if res is None:
                res = 1 if c[min(j, n - 1)] > c[i] else 0
            ret = ((tp if res == 1 else -sl) * a) / c[i] * 100 - COST_BPS / 100
            trades.append((tk, str(pd.Timestamp(ts[i]))[5:16], "TARGET" if res == 1 else "STOP", ret))
            i = j + 1
    return trades


def main():
    print(f"LAST WEEK ({LWS.date()} .. {(LWE - pd.Timedelta(days=3)).date()}) — each config, 8-name basket\n")
    print(f"  {'ver':>4} {'config':>14} {'trades':>7} {'wins':>5} {'win%':>6} {'total%':>8} {'open':>5}")
    summary = {}
    for ver, mins, tp, sl in CONFIGS:
        t = run_config(mins, tp, sl)
        summary[ver] = t
        closed = [x for x in t if x[3] is not None]
        wins = sum(x[2] == "TARGET" for x in closed)
        tot = sum(x[3] for x in closed)
        nopen = sum(x[2] == "OPEN" for x in t)
        wp = f"{wins/len(closed):.0%}" if closed else "-"
        print(f"  {ver:>4} {f'{tp:g}:{sl:g}/{mins}min':>14} {len(closed):>7} {wins:>5} {wp:>6} "
              f"{tot:>+8.2f} {nopen:>5}")
    for ver, mins, tp, sl in CONFIGS:
        t = summary[ver]
        if not t:
            continue
        print(f"\n  --- {ver} ({tp:g}:{sl:g}/{mins}min) trades ---")
        for tk, when, outcome, ret in t:
            rs = "open" if ret is None else f"{ret:+.2f}%"
            print(f"    {tk:>5} {when}  {outcome:>7}  {rs}")


if __name__ == "__main__":
    main()
