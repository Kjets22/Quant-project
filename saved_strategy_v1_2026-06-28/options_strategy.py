"""
options_strategy.py — express the frozen S/R+selectivity stock edge with OPTIONS.

The stock strategy: long-only, 1-ATR target / 1-ATR stop, hold <=24h, 54% win,
~+9 bps edge/trade. Here we replay EVERY trade and reprice it as an option
(Black-Scholes, per-ticker realized-vol as the IV proxy, realistic bid/ask + theta),
to find which structure best survives option frictions. All trades are calls
(strategy is long-only). S_exit = the actual underlying close at the bracket exit.

We decompose each structure into:  gross directional P&L  -  (theta + spread tax).
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

from triple_barrier_ml import atr, features, hourly, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features

SEL_Q = 0.93
TP = SL = 1.0
HBAR = 24
RFR = 0.04


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S, K, T, sig, r=RFR):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def trades_of(ticker):
    d = hourly(ticker)
    ts = pd.to_datetime(d["timestamp"])
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    # per-ticker IV proxy: annualized daily realized vol (clamped to sane band)
    daily = d.set_index("timestamp")["close"].resample("1D").last().dropna()
    sig = float(np.log(daily / daily.shift(1)).std() * math.sqrt(252))
    sig = min(max(sig, 0.08), 0.80)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, TP, SL)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
    tsv = ts.to_numpy()[idx]
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
            up, dn = cv[i] + TP * a, cv[i] - SL * a
            res, j = None, i + 1
            while j < min(i + HBAR + 1, n):
                if lv[j] <= dn:
                    res = 0; break
                if hv[j] >= up:
                    res = 1; break
                j += 1
            exit_j = j if (res is not None) else min(j, n - 1)
            win = 1 if (res == 1 or (res is None and cv[exit_j] > cv[i])) else 0
            dt_days = (pd.Timestamp(tsv[exit_j]) - pd.Timestamp(tsv[i])).total_seconds() / 86400.0
            out.append({"S0": cv[i], "Sx": cv[exit_j], "a": a, "sig": sig,
                        "dt": max(dt_days, 1 / 24), "win": win})
            i = j + 1
    return out


def opt_ret(tr, kind, m, dte, half_spread, frictionless=False):
    """Return on premium for one trade under a given option structure."""
    S0, Sx, a, sig = tr["S0"], tr["Sx"], tr["a"], tr["sig"]
    T0 = dte / 365.0
    T1 = T0 if frictionless else max(dte - tr["dt"], 1e-4) / 365.0
    hs = 0.0 if frictionless else half_spread
    if kind == "call":
        K = m * S0
        p0 = bs_call(S0, K, T0, sig)
        p1 = bs_call(Sx, K, T1, sig)
        gross0, gross1 = p0, p1
    else:  # debit spread: long ATM, short at the +1 ATR target (we exit there anyway)
        K1, K2 = S0, S0 + a
        c0a, c0b = bs_call(S0, K1, T0, sig), bs_call(S0, K2, T0, sig)
        c1a, c1b = bs_call(Sx, K1, T1, sig), bs_call(Sx, K2, T1, sig)
        p0, p1 = c0a - c0b, c1a - c1b
        gross0, gross1 = c0a + c0b, c1a + c1b
    if p0 <= 1e-6:
        return 0.0
    entry = p0 + hs * gross0
    proceeds = p1 - hs * gross1
    return (proceeds - entry) / entry


def evaluate(trades, kind, m, dte, hs, tag):
    full = np.array([opt_ret(t, kind, m, dte, hs) for t in trades])
    gross = np.array([opt_ret(t, kind, m, dte, hs, frictionless=True) for t in trades])
    win = (full > 0).mean()
    return {
        "tag": tag, "n": len(full), "win": win,
        "mean": full.mean() * 100, "gross": gross.mean() * 100,
        "tax": (gross.mean() - full.mean()) * 100, "total": full.sum() * 100,
        "p10": np.percentile(full, 10) * 100,
    }


def main():
    trades = []
    for tk in TICKERS:
        try:
            trades += trades_of(tk)
        except Exception as e:
            print(f"  [skip] {tk}: {e}")
    print(f"replaying {len(trades)} stock trades as options "
          f"(underlying win% = {np.mean([t['win'] for t in trades]):.1%})\n")

    HS = 0.010   # 1.0% half-spread (2% round-trip) baseline, blended liquid names
    rows = [
        evaluate(trades, "call", 1.00, 7, HS, "ATM call, 1wk"),
        evaluate(trades, "call", 0.98, 7, HS, "ITM call (delta~.6), 1wk"),
        evaluate(trades, "call", 0.95, 7, HS, "deep-ITM call (~stock), 1wk"),
        evaluate(trades, "call", 1.00, 14, HS, "ATM call, 2wk (less theta)"),
        evaluate(trades, "call", 1.00, 2, HS, "ATM call, 2dte (max leverage)"),
        evaluate(trades, "spread", 1.00, 7, HS, "debit spread, short@target, 1wk"),
    ]
    hdr = f"  {'structure':>32} {'n':>5} {'win%':>6} {'gross%':>7} {'tax%':>6} {'net/trade%':>10} {'total%':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['tag']:>32} {r['n']:>5} {r['win']:>6.1%} {r['gross']:>+7.2f} "
              f"{r['tax']:>6.2f} {r['mean']:>+10.2f} {r['total']:>+8.0f}")
    print("\n  gross% = directional P&L if options were frictionless (pure leverage of the move)")
    print("  tax%   = theta + bid/ask drag per trade;  net = gross - tax")
    print("  An option structure 'makes sense' only if net/trade stays clearly > 0.\n")

    best = max(rows, key=lambda r: r["mean"])
    print(f"  BEST: {best['tag']}  ->  net {best['mean']:+.2f}%/trade, "
          f"win {best['win']:.0%}, 10th-pctile trade {best['p10']:+.0f}%")
    # spread sensitivity on the winner
    print("\n  spread sensitivity (best structure):")
    kind = "spread" if "spread" in best["tag"] else "call"
    mny = 1.00 if "ATM" in best["tag"] or "spread" in best["tag"] else (0.98 if "ITM call (" in best["tag"] else 0.95)
    dte = 14 if "2wk" in best["tag"] else (2 if "2dte" in best["tag"] else 7)
    for hs in (0.005, 0.010, 0.020):
        r = evaluate(trades, kind, mny, dte, hs, f"half-spread {hs:.1%}")
        print(f"    {r['tag']:>20}: net {r['mean']:+.2f}%/trade  win {r['win']:.0%}  total {r['total']:+.0f}%")

    # THE key lever: tax is ~fixed/trade, gross scales with move size. Filter to the
    # biggest-expected-move setups and see if gross finally clears the option tax.
    am = np.array([t["a"] / t["S0"] for t in trades])
    qs = np.quantile(am, [0.0, 0.5, 0.75, 0.90, 0.97])
    print("\n  move-size selectivity (deep-ITM 1wk, the least-tax structure):")
    print(f"    {'subset':>14} {'n':>5} {'avg move%':>9} {'gross%':>7} {'tax%':>6} {'net/trade%':>10} {'win%':>6}")
    for lo, lab in [(qs[0], "all"), (qs[1], "top 50% move"), (qs[2], "top 25% move"),
                    (qs[3], "top 10% move"), (qs[4], "top 3% move")]:
        sub = [t for t, a_ in zip(trades, am) if a_ >= lo]
        r = evaluate(sub, "call", 0.95, 7, HS, lab)
        avg = np.mean([t["a"] / t["S0"] for t in sub]) * 100
        print(f"    {lab:>14} {r['n']:>5} {avg:>9.2f} {r['gross']:>+7.2f} {r['tax']:>6.2f} "
              f"{r['mean']:>+10.2f} {r['win']:>6.1%}")


if __name__ == "__main__":
    main()
