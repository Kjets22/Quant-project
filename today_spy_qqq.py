"""
today_spy_qqq.py — what did the frozen 30-min / 1.5:1 model do on SPY & QQQ today?
Trains on everything before the paper-start date, then prints today's 30-min bars
with the model's conviction (proba) vs its top-7% threshold, marks any signals, and
tracks the resulting bracket paper-trades. Standalone; touches no frozen snapshot.
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
from sr_features import sr_features
from basket import ticker_cfg
from data import fetch_polygon

PAPER_START = pd.Timestamp("2026-06-29")
NAMES = ["SPY", "QQQ"]
MIN = 30
TP, SL = 1.5, 1.0
SEL_Q = 0.93
HBAR = 24
COST_BPS = 3.0


def bars30(tk, end):
    cache = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv",
                        parse_dates=["timestamp"])
    cfg = ticker_cfg(tk)
    cfg.data.start_date, cfg.data.end_date = "2026-06-01", end
    cfg.data.multiplier, cfg.data.timespan = 5, "minute"
    rec = fetch_polygon(cfg)
    df = (pd.concat([cache, rec], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
    g = df.set_index("timestamp").resample(f"{MIN}min").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna()
    return g.reset_index()


def run(tk, end):
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
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr].astype(int))
    thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SEL_Q)
    fwd = np.where(fv & (ts >= start))[0]
    print(f"===== {tk} =====  top-7% conviction threshold = {thr:.3f}")
    if len(fwd) == 0:
        print("  no completed 30-min bars yet today.\n")
        return
    proba = {int(ix): float(p) for ix, p in zip(fwd, clf.predict_proba(X.iloc[fwd])[:, 1])}
    print(f"  {'bar (UTC)':>16} {'close':>9} {'conviction':>10} {'signal?':>8}")
    for ix in fwd:
        p = proba[int(ix)]
        sig = "<< SIGNAL" if p >= thr else ""
        print(f"  {str(pd.Timestamp(ts[ix]))[5:16]:>16} {c[ix]:>9.2f} {p:>10.3f} {sig:>8}")
    mx = max(proba.values())
    print(f"  today's max conviction = {mx:.3f}  (needs >= {thr:.3f} to fire)")
    # reconstruct trades
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
        if res is None and j >= n:
            out, exj, expx, ret = "OPEN", n - 1, c[n - 1], None
        else:
            if res is None:
                res = 1 if c[min(j, n - 1)] > c[i] else 0
            exj = j if j < n else n - 1
            out = "TARGET" if res == 1 else "STOP"
            expx = up if res == 1 else dn
            ret = ((TP if res == 1 else -SL) * a) / c[i] * 100 - COST_BPS / 100
        trades.append((tk, str(pd.Timestamp(ts[i]))[5:16], round(c[i], 2), round(up, 2),
                       round(dn, 2), out, ret, str(pd.Timestamp(ts[exj]))[5:16]))
        i = (exj if out == "OPEN" else j) + 1
    if trades:
        print("  --- trades today ---")
        for t in trades:
            rs = "open" if t[6] is None else f"{t[6]:+.2f}%"
            print(f"    buy {t[2]} @ {t[1]}  tgt {t[3]} stop {t[4]}  -> {t[5]} @ {t[7]} ({rs})")
    else:
        print("  --- no trade fired today ---")
    print()


def main():
    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"SPY & QQQ — frozen 30-min / 1.5:1 model, paper day {PAPER_START.date()} (data through {end})\n")
    for tk in NAMES:
        try:
            run(tk, end)
        except Exception as e:
            print(f"  [skip {tk}] {e}\n")


if __name__ == "__main__":
    main()
