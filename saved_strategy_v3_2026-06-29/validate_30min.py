"""
validate_30min.py — the validation gauntlet for the 30-min / 1.5:1 candidate.

Walk-forward, non-overlapping trades, top-7% selectivity, 3 bps, pooled. Reports
per-trade Sharpe (annualized), per-fold consistency, and the DEFLATED SHARPE
(accounting for the ~35 configs explored across the whole project).

  python validate_30min.py          -> dev names (the 8-name basket)
  python validate_30min.py fresh     -> the 5 fresh holdout names (IWM GLD META XOM KO)

The 'fresh' run is the decisive test: those names never influenced any config choice,
so if the edge holds there it isn't multiple-testing overfit. Standalone — touches no
frozen snapshot.
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
from triple_barrier_breadth import TICKERS as DEV
from sr_features import sr_features
from trials import deflated_sharpe_ratio

FRESH = ["IWM", "GLD", "META", "XOM", "KO"]
MIN = 30
TP, SL = 1.5, 1.0
SEL_Q = 0.93
HBAR = 24
COST_BPS = 3.0


def bars(tk):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv",
                     parse_dates=["timestamp"])
    g = df.set_index("timestamp").resample(f"{MIN}min").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna()
    return g.reset_index()


def trades_of(ticker):
    d = bars(ticker)
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, TP, SL)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
    n = len(idx)
    K = 5
    bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
    rets, fold = [], []
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
            out, j = None, i + 1
            while j < min(i + HBAR + 1, n):
                if lv[j] <= dn:
                    out = 0; break
                if hv[j] >= up:
                    out = 1; break
                j += 1
            if out is None:
                out = 1 if cv[min(j, n - 1)] > cv[i] else 0
            rets.append((TP * a if out == 1 else -SL * a) / cv[i] - COST_BPS / 1e4)
            fold.append(k)
            i = j + 1
    return np.array(rets), np.array(fold)


def main():
    mode = "fresh" if (len(sys.argv) > 1 and sys.argv[1] == "fresh") else "dev"
    names = FRESH if mode == "fresh" else DEV
    n_trials = 1 if mode == "fresh" else 35
    allr, allf = [], []
    for tk in names:
        try:
            r, f = trades_of(tk)
        except Exception as e:
            print(f"  [skip {tk}] {e}")
            continue
        allr.append(r); allf.append(f)
    if not allr:
        print("no trades / no data (did the fresh fetch finish?)"); return
    r = np.concatenate(allr)
    f = np.concatenate(allf)
    n = len(r)
    yrs = 3.0
    per_year = n / yrs
    sh_ann = r.mean() / r.std() * math.sqrt(per_year)
    fsh = []
    for k in sorted(set(f)):
        rk = r[f == k]
        if rk.std() > 1e-12:
            fsh.append(rk.mean() / rk.std() * math.sqrt(per_year))
    sr_obs = r.mean() / r.std()
    dsr = deflated_sharpe_ratio(sr_obs, n, n_trials, np.var(fsh) / per_year,
                                skew=float(pd.Series(r).skew()),
                                kurt=float(pd.Series(r).kurt()) + 3)
    print(f"=== VALIDATION: 30-min / 1.5:1 / top-7% / 3 bps  [{mode.upper()} names: {names}] ===")
    print(f"  trades={n}  win%={(r>0).mean():.1%}  total return={r.sum()*100:.0f}%  "
          f"mean/trade={r.mean()*1e4:+.1f} bps")
    print(f"  annualized Sharpe ~ {sh_ann:.2f}")
    print(f"  per-fold Sharpes: {[round(x,2) for x in fsh]}  "
          f"({sum(x>0 for x in fsh)}/{len(fsh)} positive)")
    print(f"  DEFLATED SHARPE (P>0 after {n_trials} trial{'s' if n_trials>1 else ''}) = {dsr:.3f}")
    if mode == "dev":
        print("  >0.95 => survives multiple-testing on the dev names.")
    else:
        print("  FRESH holdout: high win% + positive Sharpe here = the edge is real, not overfit.")


if __name__ == "__main__":
    main()
