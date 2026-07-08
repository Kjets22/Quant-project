"""
optimize_tpsl.py — does a wider target (1.5:1, 2:1) beat the 1:1 bracket, for
STOCK and for OPTIONS? Pooled across the 8-name basket, walk-forward, top-7%
selectivity, 3 bps stock cost. Options = deep-ITM weekly calls priced by
Black-Scholes, BUYING THE ASK and SELLING THE BID (full spread crossing).

Does NOT touch the frozen saved strategy (separate file). Pure tuning study.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import math
import numpy as np
import pandas as pd
import lightgbm as lgb

from triple_barrier_ml import atr, features, hourly, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features
from options_strategy import opt_ret   # deep-ITM/ATM BS pricer w/ ask/bid crossing

COST_BPS = 3.0
SEL_Q = 0.93
HBAR = 24
OPT_M = 0.95       # deep-ITM call (least option tax, the winning structure)
OPT_DTE = 7        # weekly


def gen(ticker, tp, sl):
    d = hourly(ticker)
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    daily = d.set_index("timestamp")["close"].resample("1D").last().dropna()
    sig = min(max(float(np.log(daily / daily.shift(1)).std() * math.sqrt(252)), 0.08), 0.80)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, tp, sl)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av, tsv = h[idx], l[idx], c[idx], A[idx], ts[idx]
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
            up, dn = cv[i] + tp * a, cv[i] - sl * a
            res, j = None, i + 1
            while j < min(i + HBAR + 1, n):
                if lv[j] <= dn:
                    res = 0; break
                if hv[j] >= up:
                    res = 1; break
                j += 1
            exit_j = j if res is not None else min(j, n - 1)
            if res == 1:
                Sx, win = cv[i] + tp * a, 1
            elif res == 0:
                Sx, win = cv[i] - sl * a, 0
            else:
                win = 1 if cv[exit_j] > cv[i] else 0
                Sx = cv[exit_j]
            stock_ret = (tp * a if win else -sl * a) / cv[i] - COST_BPS / 1e4
            dt = (pd.Timestamp(tsv[exit_j]) - pd.Timestamp(tsv[i])).total_seconds() / 86400.0
            out.append({"S0": cv[i], "Sx": Sx, "a": a, "sig": sig,
                        "dt": max(dt, 1 / 24), "win": win, "sret": stock_ret})
            i = j + 1
    return out


def opt_block(trades, hs):
    am = np.array([t["a"] / t["S0"] for t in trades])
    thr10 = np.quantile(am, 0.90)
    res = {}
    for name, sub in (("all", trades),
                      ("top10%move", [t for t, a_ in zip(trades, am) if a_ >= thr10])):
        r = np.array([opt_ret(t, "call", OPT_M, OPT_DTE, hs) for t in sub])
        res[name] = (len(r), (r > 0).mean(), r.mean() * 100, r.sum() * 100)
    return res


def main():
    print("Tuning target:stop  —  stock vs options (deep-ITM weekly, buy ask / sell bid)\n")
    for tp, sl in ((1.0, 1.0), (1.5, 1.0), (2.0, 1.0)):
        trades = []
        for tk in TICKERS:
            try:
                trades += gen(tk, tp, sl)
            except Exception as e:
                print(f"  [skip {tk}] {e}")
        sret = np.array([t["sret"] for t in trades])
        win = np.mean([t["win"] for t in trades])
        be = sl / (sl + tp)
        yrs = 3.0
        sharpe = sret.mean() / sret.std() * math.sqrt(len(sret) / yrs) if sret.std() > 0 else 0
        print(f"=== target:stop = {tp:g}:{sl:g}   (break-even win% = {be:.0%}) ===")
        print(f"  STOCK : n={len(trades)}  win%={win:.1%}  total={sret.sum()*100:+.0f}%  "
              f"mean/trade={sret.mean()*1e4:+.1f}bps  Sharpe~{sharpe:.2f}")
        for hs, lab in ((0.005, "tight (SPY/QQQ-like)"), (0.010, "blended basket")):
            ob = opt_block(trades, hs)
            print(f"  OPTIONS ask/bid @ {hs*2:.0%} round-trip spread ({lab}):")
            for name in ("all", "top10%move"):
                n, w, mt, tot = ob[name]
                print(f"      {name:>11}: n={n:>4}  win%={w:>5.1%}  net/trade={mt:>+6.2f}%  total={tot:>+7.0f}%")
        print()
    print("READ: 'better' = higher stock total/Sharpe AND options net/trade less negative or positive.")
    print("Wider targets make each move bigger (helps options clear their tax) but lower the win rate.")


if __name__ == "__main__":
    main()
