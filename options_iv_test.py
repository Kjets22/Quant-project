"""
options_iv_test.py — does a WIDER target (4:1) make BUYING calls work, once we price at
REALISTIC implied vol (the fix for last turn's bug)? Compares 1.5:1 vs 2:1 vs 4:1 on the
30-min engine, deep-ITM weekly calls, buy ask / sell bid, implied vol = realized x VRP.

The question: a wider target gives a bigger winning move (good for options) BUT a lower
win rate (bad for option BUYERS, who pay theta+spread on every loser). Which wins?
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
from options_strategy import opt_ret

MIN = 30
SEL_Q = 0.93
HBAR = 24
VRP = 1.25            # implied vol ~ realized x 1.25 (the premium option buyers overpay)
HS = 0.005           # tight SPY/QQQ-like half-spread


def bars(tk):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    g = df.set_index("timestamp").resample(f"{MIN}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.reset_index()


def trades_of(ticker, tp, sl):
    d = bars(ticker)
    ts = pd.to_datetime(d["timestamp"])
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    daily = d.set_index("timestamp")["close"].resample("1D").last().dropna()
    rv = min(max(float(np.log(daily / daily.shift(1)).std() * math.sqrt(252)), 0.08), 0.80)
    iv = rv * VRP
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, tp, sl)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av, tsv = h[idx], l[idx], c[idx], A[idx], ts.to_numpy()[idx]
    n = len(idx); K = 5
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
                i += 1; continue
            a = Av[i]; up, dn = cv[i] + tp * a, cv[i] - sl * a
            res, j = None, i + 1
            while j < min(i + HBAR + 1, n):
                if lv[j] <= dn:
                    res = 0; break
                if hv[j] >= up:
                    res = 1; break
                j += 1
            ex = j if res is not None else min(j, n - 1)
            if res == 1:
                Sx, win = cv[i] + tp * a, 1
            elif res == 0:
                Sx, win = cv[i] - sl * a, 0
            else:
                win = 1 if cv[ex] > cv[i] else 0; Sx = cv[ex]
            dt = (pd.Timestamp(tsv[ex]) - pd.Timestamp(tsv[i])).total_seconds() / 86400.0
            out.append({"S0": cv[i], "Sx": Sx, "a": a, "sig": iv, "dt": max(dt, 1 / 24), "win": win})
            i = j + 1
    return out


def main():
    print(f"BUYING deep-ITM weekly calls, implied vol = realized x {VRP}, 30-min engine\n")
    print(f"  {'ratio':>6} {'stock win%':>10} {'subset':>13} {'opt win%':>9} {'opt net/trade':>14}")
    for tp, sl in ((1.5, 1), (2, 1), (4, 1)):
        trades = []
        for tk in TICKERS:
            try:
                trades += trades_of(tk, tp, sl)
            except Exception as e:
                print(f"  [skip {tk}] {e}")
        sw = np.mean([t["win"] for t in trades])
        am = np.array([t["a"] / t["S0"] for t in trades])
        thr10 = np.quantile(am, 0.90)
        for name, sub in (("all signals", trades),
                          ("top-10% move", [t for t, x in zip(trades, am) if x >= thr10])):
            r = np.array([opt_ret(t, "call", 0.95, 7, HS) for t in sub])
            print(f"  {tp:g}:{sl:g}   {sw:>10.1%} {name:>13} {(r>0).mean():>9.1%} {r.mean()*100:>+13.2f}%")
        print()
    print("READ: option net/trade > 0 anywhere? A wider target lifts the winning move but drops")
    print("the win rate -- and option BUYERS pay theta+spread on every loser.")


if __name__ == "__main__":
    main()
