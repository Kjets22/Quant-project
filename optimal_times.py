"""
optimal_times.py — make the models MORE ACCURATE via a trainable TIME-OF-DAY filter.

For each walk-forward fold: after training the entry model, walk the TRAINING block and
measure the win rate of its signals per entry hour (UTC). Keep only hours where the train
win rate clears break-even + 2pp with >= 25 trades. Then trade the TEST block only in
those hours. Hour selection uses train data only -> the OOS comparison is honest.

Audited env: embargo, no-bfill ATR, 0.12% ATR floor, 5 bps effective cost, corrected
accounting (time exits at actual close; win = real target hit). Standalone; v3/v4 untouched.

  python optimal_times.py 30 1.5     # v3
  python optimal_times.py 15 4      # v4
"""

from __future__ import annotations

import sys
from collections import defaultdict

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
MIN_HOUR_N, HOUR_EDGE = 25, 0.02


def atr_fixed(h, l, c, n=24):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n).mean().to_numpy()


def walk(mask_ok, proba, thr, hv, lv, cv, Av, tsv, i0, i1, n, tp, sl):
    """Trade loop with corrected accounting. mask_ok(i) gates entries. -> (hour, ret, res)"""
    out, i = [], i0
    while i < i1 - 1:
        p = proba[i - i0]
        if p < thr or Av[i] / cv[i] < MIN_ATR_PCT or not mask_ok(i):
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
        if res == 1:
            r = tp * a / cv[i] - EFF_COST
        elif res == 0:
            r = -sl * a / cv[i] - EFF_COST
        else:
            r = (cv[ex] - cv[i]) / cv[i] - EFF_COST
        out.append((pd.Timestamp(tsv[i]).hour, r, 1 if res == 1 else 0))
        i = j + 1
    return out


def main():
    mins = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    tp = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    sl = 1.0
    be = sl / (sl + tp)
    base_trades, filt_trades = [], []
    hour_stats = defaultdict(lambda: [0, 0])          # train-side: hour -> [n, wins]
    for tk in TICKERS:
        df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
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
            tr_end = max(bnds[k] - HBAR, 300)
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                     min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                     reg_lambda=1.0, verbose=-1)
            clf.fit(Xv.iloc[:tr_end], yv[:tr_end])
            ptr = clf.predict_proba(Xv.iloc[:tr_end])[:, 1]
            thr = np.quantile(ptr, SEL_Q)
            # ---- hour selection on the TRAIN block only ----
            tr_walk = walk(lambda i: True, ptr, thr, hv, lv, cv, Av, tsv, 0, tr_end, n, tp, sl)
            hstat = defaultdict(lambda: [0, 0])
            for hr, r, w in tr_walk:
                hstat[hr][0] += 1; hstat[hr][1] += w
                hour_stats[hr][0] += 1; hour_stats[hr][1] += w
            good = {hr for hr, (nn, ww) in hstat.items()
                    if nn >= MIN_HOUR_N and ww / nn >= be + HOUR_EDGE}
            if len(good) < 2:                          # fallback: no reliable hours -> all
                good = set(range(24))
            # ---- OOS test block: baseline vs hour-filtered ----
            pte = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
            base_trades += walk(lambda i: True, pte, thr, hv, lv, cv, Av, tsv,
                                bnds[k], bnds[k + 1], n, tp, sl)
            filt_trades += walk(lambda i: pd.Timestamp(tsv[i]).hour in good, pte, thr,
                                hv, lv, cv, Av, tsv, bnds[k], bnds[k + 1], n, tp, sl)
        print(f"  ...{tk} done", flush=True)

    def rep(tag, tl):
        r = np.array([x[1] for x in tl]); w = np.mean([x[2] for x in tl])
        print(f"  {tag:>22}: trades={len(tl):>5}  tgt-win%={w:.1%}  margin={w-be:+.1%}  "
              f"mean={r.mean()*1e4:+.1f}bps  total={r.sum()*100:+.0f}%")

    print(f"\n=== OPTIMAL-TIMES: {mins}-min / {tp:g}:1 (break-even {be:.0%}) ===")
    rep("baseline (all hours)", base_trades)
    rep("train-picked hours", filt_trades)
    print("\n  train-side hour profile (UTC, pooled across folds):")
    for hr in sorted(hour_stats):
        nn, ww = hour_stats[hr]
        if nn >= 100:
            bar = "#" * int(max(ww / nn - be, 0) * 200)
            print(f"    {hr:02d}:00  n={nn:>5}  win={ww/nn:.1%}  {bar}")


if __name__ == "__main__":
    main()
