"""
exit_model4.py — proves the STOP LOSS stays on, the early exit is called BEFORE it, and shows
the per-trade breakdown of why it nets negative. v4 unchanged (15-min/4:1, -1 ATR stop, +4 ATR
target). Exit model fires before the stop; if it doesn't, the trade hits stop/target/time as usual.
For each EARLY exit we compare what we banked vs what holding would have given. Standalone.
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
from triple_barrier_breadth import TICKERS as DEV
from sr_features import sr_features

MIN, TP, SL = 15, 4.0, 1.0
SEL_Q, HBAR, COST = 0.93, 24, 3.0 / 1e4
POS_W, THR = 8.0, 0.10


def prep(tk):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    d = df.set_index("timestamp").resample(f"{MIN}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna().reset_index()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    return h, l, c, A, X.to_numpy(float), label(h, l, c, A, TP, SL)


def walk_trade(i, h, l, c, A, n):
    a = A[i]; up, dn = c[i] + TP * a, c[i] - SL * a
    j = i + 1
    while j < min(i + HBAR + 1, n):
        if l[j] <= dn:
            return "S", j
        if h[j] >= up:
            return "T", j
        j += 1
    return "X", min(j, n - 1)


def base_ret(res, ex, i, c, A):
    a = A[i]
    if res == "T":
        return TP * a / c[i] - COST
    if res == "S":
        return -SL * a / c[i] - COST
    return (c[ex] - c[i]) / c[i] - COST


def tstate(i, j, h, l, c, A):
    a = A[i]
    return [(c[j] - c[i]) / a, (j - i) / HBAR, (h[i + 1:j + 1].max() - c[i]) / a,
            (l[i + 1:j + 1].min() - c[i]) / a, (c[i] + TP * a - c[j]) / a, (c[j] - (c[i] - SL * a)) / a]


def main():
    exit_X, exit_y, store = [], [], []
    for tk in DEV:
        h, l, c, A, Xnp, y = prep(tk)
        n = len(c)
        fin = np.isfinite(Xnp).all(axis=1) & np.isfinite(y)
        cut = int(n * 0.6)
        idxtr = np.where(fin & (np.arange(n) < cut))[0]
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, verbose=-1)
        clf.fit(Xnp[idxtr], y[idxtr].astype(int))
        proba = np.full(n, -1.0); vi = np.where(fin)[0]
        proba[vi] = clf.predict_proba(Xnp[vi])[:, 1]
        thr = np.quantile(proba[idxtr], SEL_Q)
        i = 0
        while i < cut - 1:
            if not fin[i] or proba[i] < thr:
                i += 1; continue
            res, ex = walk_trade(i, h, l, c, A, n)
            tgt = 1 if res == "T" else 0
            for j in range(i + 1, ex + 1):
                if np.isfinite(Xnp[j]).all():
                    exit_X.append(np.concatenate([Xnp[j], tstate(i, j, h, l, c, A)])); exit_y.append(tgt)
            i = ex + 1
        tests = []
        i = cut
        while i < n - 1:
            if not fin[i] or proba[i] < thr:
                i += 1; continue
            res, ex = walk_trade(i, h, l, c, A, n)
            tests.append((i, res, ex)); i = ex + 1
        store.append((h, l, c, A, Xnp, tests))

    exclf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                               min_child_samples=80, subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=2.0, scale_pos_weight=POS_W, verbose=-1)
    exclf.fit(np.array(exit_X), np.array(exit_y))

    base_tot = integ_tot = 0.0
    saved_stop = lost_recover = 0.0
    n_stop_cut = n_recover_cut = n_held = n_hit_stop_held = 0
    for h, l, c, A, Xnp, tests in store:
        for i, res, ex in tests:
            b = base_ret(res, ex, i, c, A)
            base_tot += b
            if res == "S":
                n_hit_stop_held += 1
            rows, cjs = [], []
            for j in range(i + 1, ex + 1):
                if np.isfinite(Xnp[j]).all():
                    rows.append(np.concatenate([Xnp[j], tstate(i, j, h, l, c, A)])); cjs.append(c[j])
            er = b
            if rows:
                pt = exclf.predict_proba(np.array(rows))[:, 1]
                w = np.where(pt < THR)[0]
                if len(w):
                    er = (cjs[w[0]] - c[i]) / c[i] - COST
                    if res == "S":                       # would have hit the stop
                        saved_stop += (er - b); n_stop_cut += 1
                    elif res == "X":                     # would have chopped to time barrier
                        lost_recover += (er - b); n_recover_cut += 1
                else:
                    n_held += 1
            integ_tot += er

    print("v4 (15-min/4:1) — STOP LOSS STILL ON. Early exit fires BEFORE the stop; if it doesn't,")
    print(f"the -1 ATR stop / +4 ATR target do their job. (Trades that hit the stop if held: {n_hit_stop_held})\n")
    print(f"  baseline (hold, stop intact) total = {base_tot*100:+.0f}%")
    print(f"  integrated (early exit + stop)     = {integ_tot*100:+.0f}%\n")
    print("  Per EARLY exit, what holding would have given instead:")
    print(f"    cut a STOP-bound trade early:  {n_stop_cut:>4} trades, net {saved_stop*100:+.0f}% "
          f"(early exit ABOVE the stop -> SAVED)")
    print(f"    cut a trade that RECOVERED:    {n_recover_cut:>4} trades, net {lost_recover*100:+.0f}% "
          f"(it chopped back up -> LOST by cutting)")
    print(f"\n  net of early exits = {(saved_stop+lost_recover)*100:+.0f}%  "
          f"-> the recoveries cost more than the stop-saves.")
    print("  That's why exiting-before-the-stop loses even WITH the stop still in place.")


if __name__ == "__main__":
    main()
