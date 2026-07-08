"""
exit_model.py — a learned EARLY-EXIT / trade-invalidation model for v4 (15-min/4:1).

Idea: after entering, keep scoring the trade. A second LightGBM (the "exit model") is trained
on IN-TRADE states (current features + how the trade is doing) to predict whether the trade
will EVENTUALLY hit target or stop. If at any bar P(eventual win) drops below a threshold, the
trade is 'invalidated' -> exit NOW at the current price (bank partial profit / cut early).

Clean split: entry model + exit model trained on the first 60% of each name; both applied to
the last 40% (out-of-sample). Compares HOLD-to-barrier vs EARLY-EXIT. Run with 'fresh' to test
on the holdout names. Standalone; touches no frozen snapshot.
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

FRESH = ["IWM", "GLD", "META", "XOM", "KO"]
MIN, TP, SL = 15, 4.0, 1.0
SEL_Q, HBAR, COST = 0.93, 24, 3.0 / 1e4
THRS = [0.05, 0.08, 0.12, 0.16, 0.20]


def prep(tk):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    d = df.set_index("timestamp").resample(f"{MIN}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna().reset_index()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, TP, SL)
    return h, l, c, A, X.to_numpy(float), y


def walk_trade(i, h, l, c, A, n):
    a = A[i]; up, dn = c[i] + TP * a, c[i] - SL * a
    j = i + 1
    while j < min(i + HBAR + 1, n):
        if l[j] <= dn:
            return 0, j
        if h[j] >= up:
            return 1, j
        j += 1
    ex = min(j, n - 1)
    return (1 if c[ex] > c[i] else 0), ex


def tstate(i, j, h, l, c, A):
    a = A[i]
    return [(c[j] - c[i]) / a, (j - i) / HBAR,
            (h[i + 1:j + 1].max() - c[i]) / a, (l[i + 1:j + 1].min() - c[i]) / a,
            (c[i] + TP * a - c[j]) / a, (c[j] - (c[i] - SL * a)) / a]


def run(names, tag):
    exit_X, exit_y, store = [], [], []
    for tk in names:
        h, l, c, A, Xnp, y = prep(tk)
        n = len(c)
        fin = np.isfinite(Xnp).all(axis=1) & np.isfinite(y)
        cut = int(n * 0.6)
        idxtr = np.where(fin & (np.arange(n) < cut))[0]
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, verbose=-1)
        clf.fit(Xnp[idxtr], y[idxtr].astype(int))
        proba = np.full(n, -1.0)
        vi = np.where(fin)[0]
        proba[vi] = clf.predict_proba(Xnp[vi])[:, 1]
        thr = np.quantile(proba[idxtr], SEL_Q)
        # build exit-training samples from TRAIN-period trades
        i = 0
        while i < cut - 1:
            if not fin[i] or proba[i] < thr:
                i += 1; continue
            out, ex = walk_trade(i, h, l, c, A, n)
            for j in range(i + 1, ex + 1):
                if np.isfinite(Xnp[j]).all():
                    exit_X.append(np.concatenate([Xnp[j], tstate(i, j, h, l, c, A)]))
                    exit_y.append(out)
            i = ex + 1
        # collect TEST-period trades for the backtest
        tests = []
        i = cut
        while i < n - 1:
            if not fin[i] or proba[i] < thr:
                i += 1; continue
            out, ex = walk_trade(i, h, l, c, A, n)
            tests.append((i, out, ex))
            i = ex + 1
        store.append((h, l, c, A, Xnp, tests))

    exclf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                               min_child_samples=80, subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=2.0, verbose=-1)
    exclf.fit(np.array(exit_X), np.array(exit_y))

    # precompute per test-trade: baseline ret + in-trade exit-prob path
    pre = []
    for h, l, c, A, Xnp, tests in store:
        for i, out, ex in tests:
            a = A[i]
            base = (TP * a if out == 1 else -SL * a) / c[i] - COST
            rows, cjs = [], []
            for j in range(i + 1, ex + 1):
                if np.isfinite(Xnp[j]).all():
                    rows.append(np.concatenate([Xnp[j], tstate(i, j, h, l, c, A)])); cjs.append(c[j])
            pw = exclf.predict_proba(np.array(rows))[:, 1] if rows else np.array([])
            pre.append((base, c[i], np.array(cjs), pw, out))

    base = np.array([p[0] for p in pre])
    print(f"=== EXIT MODEL on v4 (15-min/4:1) [{tag}] — {len(pre)} OOS test trades ===")
    print(f"  HOLD to barrier (baseline): win%={(base>0).mean():.1%}  total={base.sum()*100:+.0f}%  "
          f"mean={base.mean()*1e4:+.1f}bps")
    print(f"  {'exit thr':>9} {'%exited':>8} {'win%':>6} {'total%':>8} {'mean bps':>9}")
    for THR in THRS:
        rets, nearly = [], 0
        for b, ci, cjs, pw, out in pre:
            er = b
            below = np.where(pw < THR)[0]
            if len(below):
                er = (cjs[below[0]] - ci) / ci - COST; nearly += 1
            rets.append(er)
        r = np.array(rets)
        print(f"  {'<'+str(THR):>9} {nearly/len(pre):>8.0%} {(r>0).mean():>6.1%} "
              f"{r.sum()*100:>+8.0f} {r.mean()*1e4:>+9.1f}")
    print("  Improvement = a threshold with HIGHER total than baseline (early exits saved more than")
    print("  they gave up). Must then hold on the FRESH names.\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "fresh":
        run(FRESH, "FRESH HOLDOUT")
    else:
        run(DEV, "DEV")


if __name__ == "__main__":
    main()
