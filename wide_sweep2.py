"""
wide_sweep2.py — CORRECTED wide-target sweep (5:1 .. 10:1) + v3/v4 reference rows.

Fixes the time-exit accounting bug in wide_sweep.py: a trade that hits neither barrier
within HBAR bars now exits at the ACTUAL close (not a full +target credit), and 'win%'
counts ONLY real target hits. Same audited environment as edge_proof: embargoed training,
no-bfill ATR, 0.12% ATR floor, 5 bps effective cost. Walk-forward, top-7%, 8-name basket.

  python wide_sweep2.py 60        # one timeframe per call (keeps runtime bounded)
  python wide_sweep2.py 30
  python wide_sweep2.py 15
"""

from __future__ import annotations

import sys
from pathlib import Path

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

SEL_Q, HBAR = 0.93, 24
EFF_COST = (3.0 + 2 * 1.0) / 1e4
MIN_ATR_PCT = 0.0012
REF = {60: [], 30: [(1.5, 1.0, "v3 ref")], 15: [(4.0, 1.0, "v4 ref")]}
RATIOS = [(5.0, 1.0, ""), (6.0, 1.0, ""), (7.0, 1.0, ""), (8.0, 1.0, ""),
          (9.0, 1.0, ""), (10.0, 1.0, "")]
_FEAT = {}   # (ticker) -> prepared arrays; features don't depend on the ratio


def atr_fixed(h, l, c, n=24):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n).mean().to_numpy()


def prep(tk, mins):
    key = (tk, mins)
    if key in _FEAT:
        return _FEAT[key]
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    d = df.set_index("timestamp").resample(f"{mins}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna().reset_index()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr_fixed(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    _FEAT[key] = (h, l, c, A, X)
    return _FEAT[key]


def run_cell(mins, tp, sl):
    TT = 0
    n_tgt = n_stp = n_time = 0
    SR = 0.0
    for tk in TICKERS:
        h, l, c, A, X = prep(tk, mins)
        y = label(h, l, c, A, tp, sl)
        m = (X.notna().all(axis=1) & np.isfinite(y) & np.isfinite(A)).to_numpy()
        idx = np.where(m)[0]
        Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
        hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
        n = len(idx); K = 5
        bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
        for k in range(K):
            tr_end = max(bnds[k] - HBAR, 300)
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                     min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                     reg_lambda=1.0, verbose=-1)
            clf.fit(Xv.iloc[:tr_end], yv[:tr_end])
            thr = np.quantile(clf.predict_proba(Xv.iloc[:tr_end])[:, 1], SEL_Q)
            proba = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
            i = bnds[k]
            while i < bnds[k + 1] - 1:
                if proba[i - bnds[k]] < thr or Av[i] / cv[i] < MIN_ATR_PCT:
                    i += 1; continue
                a = Av[i]; up, dn = cv[i] + tp * a, cv[i] - sl * a
                res, j = None, i + 1
                while j < min(i + HBAR + 1, n):
                    if lv[j] <= dn:
                        res = 0; break
                    if hv[j] >= up:
                        res = 1; break
                    j += 1
                ex = min(j, n - 1)
                TT += 1
                if res == 1:                       # real target hit
                    n_tgt += 1
                    SR += tp * a / cv[i] - EFF_COST
                elif res == 0:                     # stop hit
                    n_stp += 1
                    SR += -sl * a / cv[i] - EFF_COST
                else:                              # TIME EXIT at the actual close (the fix)
                    n_time += 1
                    SR += (cv[ex] - cv[i]) / cv[i] - EFF_COST
                i = j + 1
    return TT, n_tgt, n_stp, n_time, SR


def main():
    mins = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    if len(sys.argv) > 2:                      # optional ratio subset, e.g. "5,6,7"
        want = {float(x) for x in sys.argv[2].split(",")}
        cells = REF[mins] + [r for r in RATIOS if r[0] in want]
        suffix = f"_{sys.argv[2].replace(',', '-')}"
    else:
        cells = REF[mins] + RATIOS
        suffix = ""
    out = Path(f"runs/wide_sweep2_{mins}{suffix}.txt"); out.parent.mkdir(exist_ok=True)
    hdr = (f"CORRECTED sweep @ {mins}-min (time exits at actual close; win = target hit only)\n"
           f"  {'ratio':>8} {'trades':>7} {'tgt%':>6} {'stop%':>6} {'time%':>6} "
           f"{'mean bps':>9} {'total%':>8}")
    print(hdr, flush=True)
    with out.open("w") as fh:
        fh.write(hdr + "\n")
        for tp, sl, tag in cells:
            TT, ntg, nst, nti, SR = run_cell(mins, tp, sl)
            flag = "  <== POSITIVE" if SR > 0 else ""
            lbl = f"{tp:g}:{sl:g}" + (f" ({tag})" if tag else "")
            line = (f"  {lbl:>12} {TT:>7} {ntg/TT:>6.1%} {nst/TT:>6.1%} {nti/TT:>6.1%} "
                    f"{SR/TT*1e4:>+9.1f} {SR*100:>+8.0f}{flag}")
            print(line, flush=True); fh.write(line + "\n"); fh.flush()
    print("done", flush=True)


if __name__ == "__main__":
    main()
