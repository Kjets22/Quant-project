"""
validate_two_sided.py — holdout-validate the TWO-SIDED stock strategy (long on up-signals,
short on down-signals). Trains a long model AND a short model, 30-min, top-7%, walk-forward,
pooled. Reports combined win%, Sharpe, per-fold consistency, deflated Sharpe on DEV, then
the decisive FRESH-ticker holdout. Includes a short-borrow cost. Standalone.
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

from triple_barrier_ml import atr, features
from triple_barrier_breadth import TICKERS as DEV
from sr_features import sr_features
from trials import deflated_sharpe_ratio

FRESH = ["IWM", "GLD", "META", "XOM", "KO"]
MIN, TP, SL = 30, 1.0, 1.0
SEL_Q, HBAR, COST_BPS = 0.93, 24, 3.0
SHORT_BORROW_BPS = 2.0          # extra cost on short trades (borrow/locate)


def bars(tk):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    g = df.set_index("timestamp").resample(f"{MIN}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.reset_index()


def label_dir(h, l, c, A, side):
    n = len(c); y = np.full(n, np.nan)
    for i in range(n - 1):
        a = A[i]
        up = c[i] + (SL if side == "short" else TP) * a
        dn = c[i] - (SL if side == "long" else TP) * a
        for j in range(i + 1, min(i + HBAR + 1, n)):
            if side == "long":
                if l[j] <= dn:
                    y[i] = 0; break
                if h[j] >= up:
                    y[i] = 1; break
            else:
                if h[j] >= up:
                    y[i] = 0; break
                if l[j] <= dn:
                    y[i] = 1; break
    return y


def gen(tk, side):
    d = bars(tk)
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label_dir(h, l, c, A, side)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
    n = len(idx); K = 5
    bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
    cost = (COST_BPS + (SHORT_BORROW_BPS if side == "short" else 0)) / 1e4
    rets, fold = [], []
    for k in range(K):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, verbose=-1)
        clf.fit(Xv.iloc[:bnds[k]], yv[:bnds[k]])
        thr = np.quantile(clf.predict_proba(Xv.iloc[:bnds[k]])[:, 1], SEL_Q)
        proba = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
        i = bnds[k]
        while i < bnds[k + 1] - 1:
            if proba[i - bnds[k]] < thr:
                i += 1; continue
            a = Av[i]
            if side == "long":
                tgt, stp = cv[i] + TP * a, cv[i] - SL * a
            else:
                tgt, stp = cv[i] - TP * a, cv[i] + SL * a
            res, j = None, i + 1
            while j < min(i + HBAR + 1, n):
                if side == "long":
                    if lv[j] <= stp:
                        res = 0; break
                    if hv[j] >= tgt:
                        res = 1; break
                else:
                    if hv[j] >= stp:
                        res = 0; break
                    if lv[j] <= tgt:
                        res = 1; break
                j += 1
            if res is None:
                res = 1 if ((cv[min(j, n - 1)] > cv[i]) == (side == "long")) else 0
            rets.append((TP * a if res == 1 else -SL * a) / cv[i] - cost)
            fold.append(k)
            i = j + 1
    return rets, fold


def evaluate(names, ntrials, tag):
    allr, allf = [], []
    long_tot = short_tot = 0.0
    for tk in names:
        for side in ("long", "short"):
            r, f = gen(tk, side)
            allr += r; allf += f
            if side == "long":
                long_tot += sum(r) * 100
            else:
                short_tot += sum(r) * 100
    r = np.array(allr); f = np.array(allf); n = len(r)
    yrs = 3.0; per_year = n / yrs
    sh = r.mean() / r.std() * math.sqrt(per_year)
    fsh = [r[f == k].mean() / r[f == k].std() * math.sqrt(per_year)
           for k in sorted(set(f)) if r[f == k].std() > 1e-12]
    dsr = deflated_sharpe_ratio(r.mean() / r.std(), n, ntrials, np.var(fsh) / per_year,
                                skew=float(pd.Series(r).skew()), kurt=float(pd.Series(r).kurt()) + 3)
    print(f"=== TWO-SIDED 30-min/1:1 [{tag}: {names}] ===")
    print(f"  trades={n}  win%={(r>0).mean():.1%}  total={r.sum()*100:.0f}%  "
          f"(long {long_tot:+.0f}% / short {short_tot:+.0f}%)  mean/trade={r.mean()*1e4:+.1f} bps")
    print(f"  annualized Sharpe ~ {sh:.2f}   per-fold {[round(x,2) for x in fsh]} "
          f"({sum(x>0 for x in fsh)}/{len(fsh)} +)")
    print(f"  DEFLATED SHARPE (after {ntrials} trial{'s' if ntrials>1 else ''}) = {dsr:.3f}\n")


def main():
    print("Validating the two-sided strategy (includes 2 bps short-borrow cost)\n")
    evaluate(DEV, 35, "DEV")
    evaluate(FRESH, 1, "FRESH HOLDOUT")
    print("FRESH win% > 50% + positive Sharpe + folds mostly positive = the two-sided edge is real.")


if __name__ == "__main__":
    main()
