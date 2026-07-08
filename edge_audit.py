"""
edge_audit.py — (1) fix the one real leakage in the backtest (add an EMBARGO so a bar's
forward-looking label can't spill into the test fold), and (2) let the model TELL US what
the edge is, via gain-based feature importance. Runs v3 and v4 on dev AND the fresh holdout.

Embargo: before each walk-forward test fold, purge the last H training bars (their labels
look H bars ahead, into the test region). This removes the only look-ahead in the pipeline.
If the win-rate margin over break-even SURVIVES the embargo on the FRESH names, the edge is
real and the backtest is clean. Standalone; touches no frozen snapshot.
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
SEL_Q, HBAR, COST_BPS = 0.93, 24, 3.0


def bars(tk, mins):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    g = df.set_index("timestamp").resample(f"{mins}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.reset_index()


def run(mins, tp, sl, names, embargo):
    TT = TW = 0
    SR = 0.0
    imp = None
    cols = None
    for tk in names:
        d = bars(tk, mins)
        h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
        A = atr(h, l, c)
        X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                       sr_features(d).reset_index(drop=True)], axis=1)
        cols = list(X.columns)
        y = label(h, l, c, A, tp, sl)
        m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
        idx = np.where(m)[0]
        Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
        hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
        n = len(idx); K = 5
        bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
        if imp is None:
            imp = np.zeros(len(cols))
        for k in range(K):
            tr_end = bnds[k] - (HBAR if embargo else 0)      # EMBARGO: purge last H train bars
            if tr_end < 300:
                continue
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                     min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                     reg_lambda=1.0, verbose=-1)
            clf.fit(Xv.iloc[:tr_end], yv[:tr_end])
            imp += clf.booster_.feature_importance(importance_type="gain")
            thr = np.quantile(clf.predict_proba(Xv.iloc[:tr_end])[:, 1], SEL_Q)
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
                if res is None:
                    res = 1 if cv[min(j, n - 1)] > cv[i] else 0
                TT += 1; TW += res
                SR += (tp * a if res == 1 else -sl * a) / cv[i] - COST_BPS / 1e4
                i = j + 1
    be = sl / (sl + tp)
    return TW / TT, be, SR * 100, SR / TT * 1e4, (imp / imp.sum() if imp.sum() else imp), cols


def main():
    for mins, tp, sl, name in ((30, 1.5, 1.0, "v3 (30-min/1.5:1)"), (15, 4.0, 1.0, "v4 (15-min/4:1)")):
        print(f"================  {name}  ================")
        print(f"  {'universe':>16} {'embargo':>8} {'win%':>6} {'be':>5} {'margin':>8} {'mean bps':>9} {'total%':>8}")
        imp_keep = cols_keep = None
        for label_, names in (("DEV", DEV), ("FRESH holdout", FRESH)):
            for emb in (False, True):
                wr, be, tot, bps, imp, cols = run(mins, tp, sl, names, emb)
                print(f"  {label_:>16} {('ON' if emb else 'off'):>8} {wr:>6.1%} {be:>5.0%} "
                      f"{wr-be:>+8.1%} {bps:>+9.1f} {tot:>+8.0f}", flush=True)
                if label_ == "DEV" and emb:
                    imp_keep, cols_keep = imp, cols
        order = np.argsort(imp_keep)[::-1]
        print(f"  what the model leans on most (gain %, embargoed DEV):")
        print("   ", "  ".join(f"{cols_keep[o]} {imp_keep[o]*100:.0f}%" for o in order[:7]))
        print()


if __name__ == "__main__":
    main()
