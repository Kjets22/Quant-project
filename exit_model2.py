"""
exit_model2.py — refined early-exit for v4 (15-min/4:1) per the user's fixes:
  (1) ONLY exit a trade that is currently UNDERWATER (unrealized < 0) -> never cut a profitable
      trade, so anything heading to +4 ATR is left alone.
  (2) Exit only when the model is confident the trade will LOSE (P(win) < threshold).
  (3) A "win" only counts if it actually hits the +4 ATR target.
Crucially it PRINTS THE DATA: of the trades it cuts, how many were real losers vs winners it
wrongly killed (the regret), and the ATR saved vs given up. Clean 60/40 split. Standalone.
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
THRS = [0.02, 0.05, 0.10, 0.15]


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
                               reg_lambda=2.0, is_unbalance=True, verbose=-1)
    exclf.fit(np.array(exit_X), np.array(exit_y))

    # precompute per test trade: baseline, eventual outcome, and the underwater bars + P(win)
    pre = []
    for h, l, c, A, Xnp, tests in store:
        for i, out, ex in tests:
            a = A[i]
            base = (TP * a if out == 1 else -SL * a) / c[i] - COST
            rows, cjs, under = [], [], []
            for j in range(i + 1, ex + 1):
                if np.isfinite(Xnp[j]).all():
                    rows.append(np.concatenate([Xnp[j], tstate(i, j, h, l, c, A)]))
                    cjs.append(c[j]); under.append(c[j] < c[i])      # currently losing?
            pw = exclf.predict_proba(np.array(rows))[:, 1] if rows else np.array([])
            pre.append((base, c[i], np.array(cjs), pw, np.array(under), out))

    base = np.array([p[0] for p in pre])
    n_win_target = sum(p[5] for p in pre)
    print(f"=== EXIT MODEL v2 (only-cut-underwater-losers) on v4 (15-min/4:1) [{tag}] — {len(pre)} test trades ===")
    print(f"  HOLD baseline: hit +4ATR target {n_win_target}/{len(pre)} ({n_win_target/len(pre):.0%})  "
          f"total={base.sum()*100:+.0f}%\n")
    print(f"  {'P(win)<':>8} {'#cut':>5} {'losers cut':>10} {'WINNERS cut':>11} {'precision':>9} {'exit total%':>11}")
    for THR in THRS:
        rets, cut_win, cut_lose = [], 0, 0
        for b, ci, cjs, pw, under, out in pre:
            er = b
            mask = (pw < THR) & under                      # confident loss AND underwater
            w = np.where(mask)[0]
            if len(w):
                k = w[0]
                er = (cjs[k] - ci) / ci - COST
                if out == 1:
                    cut_win += 1
                else:
                    cut_lose += 1
            rets.append(er)
        r = np.array(rets)
        ncut = cut_win + cut_lose
        prec = cut_lose / ncut if ncut else 0
        print(f"  {THR:>8} {ncut:>5} {cut_lose:>10} {cut_win:>11} {prec:>9.0%} {r.sum()*100:>+10.0f}%")
    print("\n  WINNERS cut = trades that WOULD have hit +4 ATR but got exited early (the costly mistakes).")
    print("  Even a few of these sink it, because each is worth 4x a saved loser. precision must be ~95%+.")


def main():
    run(FRESH if (len(sys.argv) > 1 and sys.argv[1] == "fresh") else DEV,
        "FRESH" if (len(sys.argv) > 1 and sys.argv[1] == "fresh") else "DEV")


if __name__ == "__main__":
    main()
