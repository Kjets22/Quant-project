"""
sel_options.py — v4 (15-min/4:1) at tighter selectivity (top 6/4/2/1%), STOCK vs OPTIONS total.
Higher selectivity raises the win rate; does that finally make BUYING calls profitable? Shows
options priced realistically (implied vol = realized x1.25, buy ask/sell bid) AND frictionless
(no tax) so you can see the tax's effect. Standalone; touches no frozen snapshot.
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

MIN, TP, SL = 15, 4.0, 1.0
HBAR, COST_BPS = 24, 3.0
SEL = [0.94, 0.96, 0.98, 0.99]            # top 6%, 4%, 2%, 1%
VRP, HS = 1.25, 0.005


def walk(proba, thr, hv, lv, cv, Av, i0, i1, n, iv):
    out, i = [], i0
    while i < i1 - 1:
        if proba[i - i0] < thr:
            i += 1; continue
        a = Av[i]; up, dn = cv[i] + TP * a, cv[i] - SL * a
        res, j = None, i + 1
        while j < min(i + HBAR + 1, n):
            if lv[j] <= dn:
                res = 0; break
            if hv[j] >= up:
                res = 1; break
            j += 1
        if res is None:
            res = 1 if cv[min(j, n - 1)] > cv[i] else 0
        Sx = up if res == 1 else dn
        sret = (TP * a if res == 1 else -SL * a) / cv[i] - COST_BPS / 1e4
        dt = max((j if j < n else n - 1) - i, 1) * MIN / (60 * 24)
        t = {"S0": cv[i], "Sx": Sx, "a": a, "sig": iv, "dt": dt}
        out.append((sret, opt_ret(t, "call", 0.95, 7, HS),
                    opt_ret(t, "call", 0.95, 7, 0.0, frictionless=True), res))
        i = j + 1
    return out


def main():
    agg = {q: [] for q in SEL}
    for tk in TICKERS:
        try:
            df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
        except Exception:
            continue
        d = df.set_index("timestamp").resample(f"{MIN}min").agg(
            high=("high", "max"), low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum")).dropna().reset_index()
        h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
        A = atr(h, l, c)
        daily = d.set_index("timestamp")["close"].resample("1D").last().dropna()
        iv = min(max(float(np.log(daily / daily.shift(1)).std() * math.sqrt(252)), 0.08), 0.80) * VRP
        X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                       sr_features(d).reset_index(drop=True)], axis=1)
        y = label(h, l, c, A, TP, SL)
        m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
        idx = np.where(m)[0]
        Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
        hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
        n = len(idx); K = 5
        bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
        for k in range(K):
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                     min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                     reg_lambda=1.0, verbose=-1)
            clf.fit(Xv.iloc[:bnds[k]], yv[:bnds[k]])
            ptr = clf.predict_proba(Xv.iloc[:bnds[k]])[:, 1]
            proba = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
            for q in SEL:
                agg[q] += walk(proba, np.quantile(ptr, q), hv, lv, cv, Av, bnds[k], bnds[k + 1], n, iv)

    print("v4 (15-min/4:1) by selectivity — STOCK vs OPTIONS total\n")
    print(f"  {'tier':>8} {'trades':>7} {'win%':>6} {'STOCK total':>12} {'OPT total(tax)':>15} {'OPT total(no tax)':>18}")
    for q in SEL:
        a = np.array(agg[q])
        s, o, of, w = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        print(f"  {'top '+str(round((1-q)*100))+'%':>8} {len(a):>7} {w.mean():>6.1%} "
              f"{s.sum()*100:>+11.0f}% {o.sum()*100:>+14.0f}% {of.sum()*100:>+17.0f}%")
    print("\n  Even as win% climbs with selectivity, does OPT total(tax) ever beat STOCK? (No tax shows")
    print("  the gross leverage; the gap between the two columns is the option tax you can't avoid.)")


if __name__ == "__main__":
    main()
