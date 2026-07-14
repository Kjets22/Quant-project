"""
qqq_bounce.py — can a model predict a $2 QQQ bounce within 60 minutes? (last year OOS)

Setup: 5-minute QQQ bars. From each bar close, label 1 if the HIGH reaches close+$2
within the next 12 bars (60 min) BEFORE the LOW hits the stop; ties -> stop. Two variants:
  A: target +$2 / stop -$1  (2:1 payoff, break-even 33.3%)
  B: target +$2 / stop -$2  (1:1 payoff, break-even 50%)
Features: the validated base+S/R block. Model: standard LightGBM config. Selectivity:
top-7% (train-only threshold). Walk-forward: 4 folds across the LAST YEAR, each trained
on all history before the fold (embargoed 12 bars). Costs: 5 bps effective per trade.

Also reports the unconditional base rate (how often $2-in-60min happens at all) so the
model's selected precision has context.
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

from triple_barrier_ml import features
from sr_features import sr_features

H = 12                      # 12 x 5-min = 60 minutes
TGT = 2.0                   # the $2 bounce
EFF_COST = 5.0 / 1e4
SEL_Q = 0.93
YEAR_START = "2025-07-14"


def load_qqq():
    base = pd.read_csv("data_cache/QQQ_5minute_2021-06-01_2026-06-01.csv",
                       parse_dates=["timestamp"])
    parts = [base]
    rec = sorted(Path("data_cache").glob("QQQ_recent_2026-06-01_*.csv"))
    if rec:
        parts.append(pd.read_csv(rec[-1], parse_dates=["timestamp"]))
    df = (pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
    return df.reset_index(drop=True)


def label_dollar(h, l, c, tgt, stp):
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        up, dn = c[i] + tgt, c[i] - stp
        for j in range(i + 1, min(i + H + 1, n)):
            if l[j] <= dn:
                y[i] = 0; break
            if h[j] >= up:
                y[i] = 1; break
    return y


def run(stp, tag):
    d = load_qqq()
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label_dollar(h, l, c, TGT, stp)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, tsv = h[idx], l[idx], c[idx], ts[idx]
    n = len(idx)
    yr = np.where(tsv >= np.datetime64(YEAR_START))[0]
    be = stp / (stp + TGT)
    print(f"\n=== variant {tag}: target +${TGT:g} / stop -${stp:g} (break-even {be:.1%}) ===")
    print(f"  bars in test year: {len(yr)}  |  unconditional base rate "
          f"P($2 bounce in 60min, no-stop-first) = {yv[yr].mean():.1%}")
    bnds = np.linspace(yr[0], n, 5).astype(int)          # 4 folds across the year
    rets, wins, taken = [], 0, 0
    for k in range(4):
        tr_end = max(bnds[k] - H, 500)
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, verbose=-1)
        clf.fit(Xv.iloc[:tr_end], yv[:tr_end])
        thr = np.quantile(clf.predict_proba(Xv.iloc[:tr_end])[:, 1], SEL_Q)
        proba = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
        i = bnds[k]
        while i < bnds[k + 1] - 1:
            if proba[i - bnds[k]] < thr:
                i += 1; continue
            up, dn = cv[i] + TGT, cv[i] - stp
            res, j = None, i + 1
            while j < min(i + H + 1, n):
                if lv[j] <= dn:
                    res = 0; break
                if hv[j] >= up:
                    res = 1; break
                j += 1
            ex = min(j, n - 1)
            if res == 1:
                r = TGT / cv[i] - EFF_COST
            elif res == 0:
                r = -stp / cv[i] - EFF_COST
            else:
                r = (cv[ex] - cv[i]) / cv[i] - EFF_COST      # 60-min clock exit at close
            rets.append((tsv[i], r))
            wins += 1 if res == 1 else 0
            taken += 1
            i = j + 1
    r = np.array([x[1] for x in rets])
    s = pd.DataFrame(rets, columns=["ts", "r"])
    mon = s.set_index(pd.to_datetime(s["ts"]))["r"].resample("ME").sum()
    sharpe = mon.mean() / mon.std() * np.sqrt(12) if mon.std() > 0 else 0.0
    print(f"  MODEL top-7%: trades={taken}  win%={wins/taken:.1%}  margin={wins/taken-be:+.1%}"
          f"  mean={r.mean()*1e4:+.1f}bps  total={r.sum()*100:+.1f}%  Sharpe={sharpe:.2f}")
    return taken, wins / taken, be, r


if __name__ == "__main__":
    print("QQQ $2-bounce-in-60min prediction — last year out-of-sample, 5 bps costs")
    run(1.0, "A (2:1)")
    run(2.0, "B (1:1)")
    print("\nREAD: the model 'predicts $2 bounces' if selected win% clears the break-even line")
    print("AND the margin survives costs (mean bps > 0). Compare win% to the base rate for skill.")
