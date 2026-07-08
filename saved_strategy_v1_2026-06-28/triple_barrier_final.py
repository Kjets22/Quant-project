"""
triple_barrier_final.py — the validation gauntlet on the S/R + selectivity edge.

Fixed config (no test-set tuning): 1:1, base+S/R features, selectivity threshold
chosen on TRAIN only (top ~7%), 3 bps cost, walk-forward, non-overlapping trades,
pooled across the 8-name basket. Reports per-trade return Sharpe (annualized),
per-fold consistency, and the DEFLATED SHARPE accounting for the ~25 configs we
explored across the whole project. DSR>0.95 => survives multiple-testing; else the
+$142 was our best-looking overfit.
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

from triple_barrier_ml import atr, features, hourly, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features
from trials import deflated_sharpe_ratio

COST_BPS = 3.0
SEL_Q = 0.93          # top ~7% (chosen on TRAIN probas, applied to test)
N_TRIALS = 25         # configs explored across the project (for deflation)
TP = SL = 1.0
HBAR = 24


def trades_of(ticker):
    d = hourly(ticker)
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
    rets, fold_id = [], []
    for k in range(K):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        clf.fit(Xv.iloc[:bnds[k]], yv[:bnds[k]])
        thr = np.quantile(clf.predict_proba(Xv.iloc[:bnds[k]])[:, 1], SEL_Q)  # TRAIN-only
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
            pnl = (TP * a if out == 1 else -SL * a) - COST_BPS / 1e4 * cv[i]
            rets.append(pnl / cv[i])          # return on notional
            fold_id.append(k)
            i = j + 1
    return np.array(rets), np.array(fold_id)


def main():
    allr, allf = [], []
    for tk in TICKERS:
        try:
            r, f = trades_of(tk)
        except Exception:
            continue
        allr.append(r); allf.append(f)
    r = np.concatenate(allr)
    f = np.concatenate(allf)
    n = len(r)
    yrs = 5.0 * 0.6                                  # ~OOS span (last 60% of ~5y)
    per_year = n / yrs
    sh_ann = r.mean() / r.std() * np.sqrt(per_year)
    # per-fold annualized Sharpe for the deflation variance
    fsh = []
    for k in sorted(set(f)):
        rk = r[f == k]
        if rk.std() > 1e-12:
            fsh.append(rk.mean() / rk.std() * np.sqrt(per_year))
    sr_obs = r.mean() / r.std()
    dsr = deflated_sharpe_ratio(sr_obs, n, N_TRIALS, np.var(fsh) / per_year,
                                skew=float(pd.Series(r).skew()),
                                kurt=float(pd.Series(r).kurt()) + 3)
    print("=== FINAL VALIDATION: S/R + top-7% selectivity, 1:1, 3 bps ===")
    print(f"  trades={n}  win%={(r>0).mean():.1%}  total return={r.sum()*100:.1f}%  "
          f"mean/trade={r.mean()*1e4:.1f} bps")
    print(f"  annualized Sharpe ~ {sh_ann:.2f}")
    print(f"  per-fold Sharpes: {[round(x,2) for x in fsh]}  ({sum(x>0 for x in fsh)}/{len(fsh)} positive)")
    print(f"  DEFLATED SHARPE (P>0 after {N_TRIALS} trials) = {dsr:.3f}")
    print("  >0.95 => survives multiple-testing (a real, if tiny, edge);")
    print("  <0.95 => not validated -- the best-looking overfit of the search.")


if __name__ == "__main__":
    main()
