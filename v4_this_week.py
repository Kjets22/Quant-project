"""
v4_this_week.py — how is v4 (15-min / 4:1) doing THIS week (starting Mon 2026-06-29)?
Fetches the latest bars, trains v4 on everything before this week, walks forward over the
week so far, and prints the blotter + P&L. Standalone; touches no frozen snapshot.
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

MIN, TP, SL = 15, 4.0, 1.0
SEL_Q, HBAR, COST_BPS = 0.93, 24, 3.0
WEEK_START = pd.Timestamp("2026-06-29")
END = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)


def bars(tk):
    cache = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    cfg = ticker_cfg(tk)
    cfg.data.start_date, cfg.data.end_date = "2026-06-01", str(END.date())
    cfg.data.multiplier, cfg.data.timespan = 5, "minute"
    rec = fetch_polygon(cfg)
    df = (pd.concat([cache, rec], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
    g = df.set_index("timestamp").resample(f"{MIN}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.reset_index()


def run(tk):
    d = bars(tk)
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, TP, SL)
    fv = X.notna().all(axis=1).to_numpy()
    n = len(c)
    tr = np.where(fv & np.isfinite(y) & (ts < np.datetime64(WEEK_START)))[0]
    if len(tr) < 500:
        return []
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr].astype(int))
    thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SEL_Q)
    fwd = np.where(fv & (ts >= np.datetime64(WEEK_START)) & (ts < np.datetime64(END)))[0]
    if len(fwd) == 0:
        return []
    proba = {int(ix): float(p) for ix, p in zip(fwd, clf.predict_proba(X.iloc[fwd])[:, 1])}
    out, i, last = [], int(fwd[0]), int(fwd[-1])
    while i <= last:
        if proba.get(i, -1) < thr:
            i += 1; continue
        a = A[i]; up, dn = c[i] + TP * a, c[i] - SL * a
        res, j = None, i + 1
        while j < min(i + HBAR + 1, n):
            if l[j] <= dn:
                res = 0; break
            if h[j] >= up:
                res = 1; break
            j += 1
        if res is None and j >= n:
            out.append((tk, str(pd.Timestamp(ts[i]))[5:16], round(c[i], 2), round(up, 2),
                        round(dn, 2), "OPEN", None)); i = n; continue
        if res is None:
            res = 1 if c[min(j, n - 1)] > c[i] else 0
        ret = ((TP if res == 1 else -SL) * a) / c[i] * 100 - COST_BPS / 100
        out.append((tk, str(pd.Timestamp(ts[i]))[5:16], round(c[i], 2), round(up, 2),
                    round(dn, 2), "TARGET" if res == 1 else "STOP", round(ret, 2)))
        i = j + 1
    return out


def main():
    print(f"v4 (15-min / 4:1) THIS WEEK ({WEEK_START.date()} .. data through {END.date()})\n")
    trades = []
    for tk in TICKERS:
        try:
            trades += run(tk)
        except Exception as e:
            print(f"  [skip {tk}] {e}")
    trades.sort(key=lambda t: t[1])
    if not trades:
        print("  No v4 signals yet this week (week just started / no top-7% bar met).")
        return
    print(f"  {'tk':>5} {'entry (UTC)':>16} {'in':>8} {'tgt':>8} {'stop':>8} {'outcome':>7} {'ret%':>7}")
    for tk, when, ein, tg, st, oc, r in trades:
        rs = "  open" if r is None else f"{r:+.2f}"
        print(f"  {tk:>5} {when:>16} {ein:>8} {tg:>8} {st:>8} {oc:>7} {rs:>7}")
    closed = [t for t in trades if t[6] is not None]
    wins = sum(t[5] == "TARGET" for t in closed)
    tot = sum(t[6] for t in closed)
    nop = sum(t[5] == "OPEN" for t in trades)
    print()
    if closed:
        print(f"  closed={len(closed)}  wins={wins}  win%={wins/len(closed):.0%}  "
              f"total (1 unit/trade)={tot:+.2f}%  open={nop}")
    else:
        print(f"  closed=0  open={nop}  (positions live, no exits yet)")


if __name__ == "__main__":
    main()
