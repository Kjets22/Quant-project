"""
market_vs_limit.py — same signals, two execution styles, head to head.

MARKET account: enter at the signal bar close, pay 5 bps effective (3 commission + 2 slip).
LIMIT account:  rest a buy-limit AT the signal close price for W bars, pay 3 bps (no spread
                crossing). Fill only if a later bar trades BELOW the limit (strict low <
                limit). Unfilled after W bars -> trade MISSED (0 P&L). If the fill bar's low
                also breaches the stop, the trade is stopped same-bar (conservative).
Bracket levels are frozen at signal time (same as live). Audited env: embargo, no-bfill
ATR, ATR floor, corrected time-exit accounting. Walk-forward, dev basket.

  python market_vs_limit.py          # v3 and v4 (the high-frequency, slippage-sensitive configs)
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

from triple_barrier_ml import features, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features
from wide_hunter import atr_fixed

SEL_Q, MIN_ATR_PCT = 0.93, 0.0012
COST_MKT = 5.0 / 1e4          # market: commission + spread/slippage
COST_LMT = 3.0 / 1e4          # limit: commission only (maker, no spread crossing)
CONFIGS = [("v3", 30, 24, 1.5, 1.0), ("v4", 15, 24, 4.0, 1.0)]
WINDOWS = [1, 4]              # limit patience, in bars


def sim_market(i, hv, lv, cv, Av, n, tp, sl, hbar):
    a = Av[i]; up, dn = cv[i] + tp * a, cv[i] - sl * a
    res, j = None, i + 1
    while j < min(i + hbar + 1, n):
        if lv[j] <= dn:
            res = 0; break
        if hv[j] >= up:
            res = 1; break
        j += 1
    ex = min(j, n - 1)
    if res == 1:
        r = tp * a / cv[i] - COST_MKT
    elif res == 0:
        r = -sl * a / cv[i] - COST_MKT
    else:
        r = (cv[ex] - cv[i]) / cv[i] - COST_MKT
    return r, (res == 1), ex


def sim_limit(i, hv, lv, cv, Av, n, tp, sl, hbar, W):
    """Rest limit at cv[i] for W bars. Returns (ret|None, win, exit_idx)."""
    a = Av[i]; lim = cv[i]; up, dn = cv[i] + tp * a, cv[i] - sl * a
    fb = None
    for j in range(i + 1, min(i + W + 1, n)):
        if lv[j] < lim:                      # strict: price traded through the limit
            fb = j; break
    if fb is None:
        return None, False, min(i + W, n - 1)          # missed
    res, j = None, fb
    if lv[fb] <= dn:                                    # fill bar also hit the stop
        res = 0
    else:
        j = fb + 1
        while j < min(i + hbar + 1, n):
            if lv[j] <= dn:
                res = 0; break
            if hv[j] >= up:
                res = 1; break
            j += 1
    ex = min(j, n - 1)
    if res == 1:
        r = (up - lim) / lim - COST_LMT
    elif res == 0:
        r = (dn - lim) / lim - COST_LMT
    else:
        r = (cv[ex] - lim) / lim - COST_LMT
    return r, (res == 1), ex


def run(cfgname, mins, hbar, tp, sl):
    styles = ["market"] + [f"limit_W{w}" for w in WINDOWS]
    out = {s: [] for s in styles}
    missed = {f"limit_W{w}": 0 for w in WINDOWS}
    nsig = 0
    for tk in TICKERS:
        df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv",
                         parse_dates=["timestamp"])
        d = df.set_index("timestamp").resample(f"{mins}min").agg(
            high=("high", "max"), low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum")).dropna().reset_index()
        ts = pd.to_datetime(d["timestamp"]).to_numpy()
        h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
        A = atr_fixed(h, l, c)
        X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                       sr_features(d).reset_index(drop=True)], axis=1)
        y = label(h, l, c, A, tp, sl)
        m = (X.notna().all(axis=1) & np.isfinite(y) & np.isfinite(A)).to_numpy()
        idx = np.where(m)[0]
        Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
        hv, lv, cv, Av, tsv = h[idx], l[idx], c[idx], A[idx], ts[idx]
        n = len(idx); K = 5
        bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
        for k in range(K):
            tr_end = max(bnds[k] - hbar, 300)
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                     min_child_samples=40, subsample=0.8,
                                     colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
            clf.fit(Xv.iloc[:tr_end], yv[:tr_end])
            thr = np.quantile(clf.predict_proba(Xv.iloc[:tr_end])[:, 1], SEL_Q)
            proba = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
            # walk each style independently (their exits differ -> different re-entry points)
            for style in styles:
                i = bnds[k]
                while i < bnds[k + 1] - 1:
                    if proba[i - bnds[k]] < thr or Av[i] / cv[i] < MIN_ATR_PCT:
                        i += 1; continue
                    if style == "market":
                        r, w, ex = sim_market(i, hv, lv, cv, Av, n, tp, sl, hbar)
                        out[style].append((tsv[i], r, w))
                    else:
                        W = int(style.split("W")[1])
                        r, w, ex = sim_limit(i, hv, lv, cv, Av, n, tp, sl, hbar, W)
                        if r is None:
                            missed[style] += 1
                        else:
                            out[style].append((tsv[i], r, w))
                    i = ex + 1
                    if style == "market":
                        nsig += 1
    print(f"\n=== {cfgname} ({mins}-min, {tp:g}:{sl:g}) — market vs limit ===")
    print(f"  {'style':>10} {'filled':>7} {'missed':>7} {'fill%':>6} {'win%':>6} "
          f"{'mean bps':>9} {'total%':>8} {'Sharpe':>7}")
    for style in styles:
        rows = out[style]
        r = np.array([x[1] for x in rows])
        w = np.mean([x[2] for x in rows]) if len(rows) else 0
        ms = missed.get(style, 0)
        fillp = len(rows) / (len(rows) + ms) if (len(rows) + ms) else 1.0
        s = pd.DataFrame(rows, columns=["ts", "r", "w"])
        mon = s.set_index(pd.to_datetime(s["ts"]))["r"].resample("ME").sum()
        sharpe = mon.mean() / mon.std() * np.sqrt(12) if mon.std() > 0 else 0.0
        print(f"  {style:>10} {len(rows):>7} {ms:>7} {fillp:>6.0%} {w:>6.1%} "
              f"{r.mean()*1e4:>+9.1f} {r.sum()*100:>+8.0f} {sharpe:>7.2f}")


if __name__ == "__main__":
    for cfg in CONFIGS:
        run(*cfg)
    print("\nREAD: limit wins only if its total AND Sharpe beat market — cheaper fills must")
    print("outweigh the winners it missed (price never came back to the limit).")
