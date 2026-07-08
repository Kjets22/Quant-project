"""
options_30min.py — express the VALIDATED 30-min / 1.5:1 stock edge with options.

Replays every 30-min/1.5:1 trade and reprices it as a call (Black-Scholes, per-ticker
realized vol as IV proxy), BUYING THE ASK and SELLING THE BID. Tests deep-ITM weekly
calls on all signals and on the big-move subset, at tight (SPY/QQQ) and blended spreads.
Reuses the pricing from options_strategy.py so it's identical to the earlier options work.
Standalone; touches no frozen snapshot.
"""

from __future__ import annotations

import math
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
from options_strategy import opt_ret, evaluate   # same BS pricer + ask/bid crossing

FRESH = ["IWM", "GLD", "META", "XOM", "KO"]
NAMES = FRESH if (len(sys.argv) > 1 and "fresh" in sys.argv) else TICKERS
MIN = 30
TP, SL = 1.5, 1.0
SEL_Q = 0.93
HBAR = 24


def bars(tk):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv",
                     parse_dates=["timestamp"])
    g = df.set_index("timestamp").resample(f"{MIN}min").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna()
    return g.reset_index()


def trades_of(ticker):
    d = bars(ticker)
    ts = pd.to_datetime(d["timestamp"])
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    daily = d.set_index("timestamp")["close"].resample("1D").last().dropna()
    sig = min(max(float(np.log(daily / daily.shift(1)).std() * math.sqrt(252)), 0.08), 0.80)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, TP, SL)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
    tsv = ts.to_numpy()[idx]
    n = len(idx)
    K = 5
    bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
    out = []
    for k in range(K):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        clf.fit(Xv.iloc[:bnds[k]], yv[:bnds[k]])
        thr = np.quantile(clf.predict_proba(Xv.iloc[:bnds[k]])[:, 1], SEL_Q)
        proba = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
        i = bnds[k]
        while i < bnds[k + 1] - 1:
            if proba[i - bnds[k]] < thr:
                i += 1
                continue
            a = Av[i]
            up, dn = cv[i] + TP * a, cv[i] - SL * a
            res, j = None, i + 1
            while j < min(i + HBAR + 1, n):
                if lv[j] <= dn:
                    res = 0; break
                if hv[j] >= up:
                    res = 1; break
                j += 1
            exit_j = j if res is not None else min(j, n - 1)
            if res == 1:
                Sx, win = cv[i] + TP * a, 1
            elif res == 0:
                Sx, win = cv[i] - SL * a, 0
            else:
                win = 1 if cv[exit_j] > cv[i] else 0
                Sx = cv[exit_j]
            dt = (pd.Timestamp(tsv[exit_j]) - pd.Timestamp(tsv[i])).total_seconds() / 86400.0
            out.append({"S0": cv[i], "Sx": Sx, "a": a, "sig": sig,
                        "dt": max(dt, 1 / 24), "win": win})
            i = j + 1
    return out


def main():
    trades = []
    for tk in NAMES:
        try:
            trades += trades_of(tk)
        except Exception as e:
            print(f"  [skip {tk}] {e}")
    print(f"[ticker set: {NAMES}]")
    win = np.mean([t["win"] for t in trades])
    a_over_s = np.array([t["a"] / t["S0"] for t in trades])
    # stock baseline at 1.5:1
    sret = np.array([((TP if t["win"] else -SL) * t["a"]) / t["S0"] - 3 / 1e4 for t in trades])
    print(f"30-min / 1.5:1 — {len(trades)} trades, underlying win% {win:.1%}, "
          f"STOCK total {sret.sum()*100:+.0f}% (+{sret.mean()*1e4:.1f} bps/trade)\n")

    for hs, lab in ((0.005, "tight SPY/QQQ-like, 1% round-trip"),
                    (0.010, "blended basket, 2% round-trip")):
        print(f"OPTIONS — buy ask / sell bid @ {lab}:")
        rows = [evaluate(trades, "call", 1.00, 7, hs, "ATM weekly, all signals"),
                evaluate(trades, "call", 0.95, 7, hs, "deep-ITM weekly, all signals")]
        thr10 = np.quantile(a_over_s, 0.90)
        thr03 = np.quantile(a_over_s, 0.97)
        sub10 = [t for t, x in zip(trades, a_over_s) if x >= thr10]
        sub03 = [t for t, x in zip(trades, a_over_s) if x >= thr03]
        rows.append(evaluate(sub10, "call", 0.95, 7, hs, "deep-ITM weekly, top-10% move"))
        rows.append(evaluate(sub03, "call", 0.95, 7, hs, "deep-ITM weekly, top-3% move"))
        print(f"    {'structure':>32} {'n':>5} {'win%':>6} {'gross%':>7} {'tax%':>6} {'net/trade%':>11} {'total%':>8}")
        for r in rows:
            print(f"    {r['tag']:>32} {r['n']:>5} {r['win']:>6.1%} {r['gross']:>+7.2f} "
                  f"{r['tax']:>6.2f} {r['mean']:>+11.2f} {r['total']:>+8.0f}")
        print()
    print("READ: deep-ITM weekly = least option tax. Positive net/trade only where the move")
    print("is big enough to cover theta+spread -- i.e. the big-move subset, same as 60-min.")


if __name__ == "__main__":
    main()
