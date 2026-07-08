"""
two_sided_options.py — DIRECTIONAL options: up-signal -> buy CALL, down-signal -> buy PUT.

Same engine, but trains TWO models: a long model (price hits +1 ATR first) and a short model
(price hits -1 ATR first). On each high-conviction signal it buys a deep-ITM weekly option on
the matching side, priced by Black-Scholes at REALISTIC implied vol (calls = realized x1.25,
puts = realized x1.35 for put skew), buy ask / sell bid. Reports STOCK two-sided vs OPTIONS
two-sided so we can see if the directional edge exists AND whether options keep it. 30-min,
basket, walk-forward. Standalone; touches no frozen snapshot.
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
from triple_barrier_breadth import TICKERS
from sr_features import sr_features

MIN, TP, SL = 30, 1.0, 1.0
SEL_Q, HBAR, COST_BPS = 0.93, 24, 3.0
RFR = 0.04
IV_CALL, IV_PUT, HS = 1.25, 1.35, 0.005      # implied-vol markups + half-spread


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S, K, T, sig):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (RFR + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return S * _ncdf(d1) - K * math.exp(-RFR * T) * _ncdf(d1 - sig * math.sqrt(T))


def bs_put(S, K, T, sig):
    return bs_call(S, K, T, sig) - S + K * math.exp(-RFR * T)


def opt_ret(S0, Sx, sig, dt, kind, m, dte):
    K = m * S0
    T0, T1 = dte / 365.0, max(dte - dt, 1e-4) / 365.0
    f = bs_call if kind == "call" else bs_put
    p0, p1 = f(S0, K, T0, sig), f(Sx, K, T1, sig)
    if p0 <= 0.05:
        return 0.0
    return (p1 * (1 - HS) - p0 * (1 + HS)) / (p0 * (1 + HS))


def label_dir(h, l, c, A, side):
    n = len(c); y = np.full(n, np.nan)
    for i in range(n - 1):
        a = A[i]
        up, dn = c[i] + (SL if side == "short" else TP) * a, c[i] - (SL if side == "long" else TP) * a
        for j in range(i + 1, min(i + HBAR + 1, n)):
            hi, lo = h[j] >= up, l[j] <= dn
            if side == "long":
                if lo:
                    y[i] = 0; break
                if hi:
                    y[i] = 1; break
            else:
                if hi:
                    y[i] = 0; break
                if lo:
                    y[i] = 1; break
    return y


def gen(tk, side):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    d = df.set_index("timestamp").resample(f"{MIN}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna().reset_index()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    daily = d.set_index("timestamp")["close"].resample("1D").last().dropna()
    rv = min(max(float(np.log(daily / daily.shift(1)).std() * math.sqrt(252)), 0.08), 0.80)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label_dir(h, l, c, A, side)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
    n = len(idx); K = 5
    bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
    kind = "call" if side == "long" else "put"
    mny = 0.95 if side == "long" else 1.05
    iv = rv * (IV_CALL if side == "long" else IV_PUT)
    srets, orets = [], []
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
            ex = j if (res is not None and j < n) else min(j, n - 1)
            if res is None:
                res = 1 if ((cv[ex] > cv[i]) == (side == "long")) else 0
            Sx = tgt if res == 1 else stp
            sret = (TP * a if res == 1 else -SL * a) / cv[i] - COST_BPS / 1e4
            dt = max((ex - i) * MIN / (60 * 24), 1 / 48)
            srets.append(sret)
            orets.append(opt_ret(cv[i], Sx, iv, dt, kind, mny, 7))
            i = j + 1
    return np.array(srets), np.array(orets)


def main():
    print("DIRECTIONAL options: up->CALL, down->PUT (deep-ITM weekly, implied vol, buy ask/sell bid)\n")
    res = {}
    for side in ("long", "short"):
        sr, orr = [], []
        for tk in TICKERS:
            try:
                a, b = gen(tk, side)
            except Exception as e:
                print(f"  [skip {tk} {side}] {e}"); continue
            sr.append(a); orr.append(b)
        res[side] = (np.concatenate(sr), np.concatenate(orr))
    print(f"  {'side':>16} {'n':>6} {'STOCK total':>12} {'STOCK bps':>10} {'OPTION total':>13} {'OPTION/trade':>13}")
    cs = co = 0.0
    for side, instr in (("long", "CALL"), ("short", "PUT")):
        s, o = res[side]
        cs += s.sum() * 100; co += o.sum() * 100
        print(f"  {side+' ('+instr+')':>16} {len(s):>6} {s.sum()*100:>+11.0f}% {s.mean()*1e4:>+9.1f} "
              f"{o.sum()*100:>+12.0f}% {o.mean()*100:>+12.2f}%")
    print(f"  {'COMBINED':>16} {'':>6} {cs:>+11.0f}% {'':>10} {co:>+12.0f}%")
    print("\n  If STOCK combined > 0 but OPTION combined < 0: the directional edge is real on BOTH")
    print("  sides, but buying calls+puts still loses to the vol-risk-premium + spread (the tax).")


if __name__ == "__main__":
    main()
