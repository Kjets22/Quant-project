"""
vob_model.py — the ORDER-BOOK trader (vOB): does book state predict short moves?

Joins the NBBO features (vob_data.py backfill) with 5-min price bars, builds
book-centric features (current + short history of imbalance / microprice
pressure / spread / activity), labels with a small triple barrier, trains
LightGBM walk-forward, and reports honest out-of-sample results per month.

  python vob_model.py QQQ
  python vob_model.py QQQ --tgt 0.003 --stop 0.002 --H 24

Research only. The point of iteration 1 is a clean read on whether L1 book
information carries ANY edge at 5-min scale after 5 bps costs.
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

from alpaca_bot2 import full_series

COST = 5.0 / 1e4
VOB = Path("data_cache/vob")


def arg(flag, default):
    if flag in sys.argv:
        return float(sys.argv[sys.argv.index(flag) + 1])
    return default


def load_book(tk):
    fs = sorted(VOB.glob(f"{tk}_*.parquet"))
    if not fs:
        raise SystemExit(f"no vob data for {tk} — run vob_data.py first")
    b = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    b["timestamp"] = pd.to_datetime(b["timestamp"])
    return b.drop_duplicates(subset="timestamp").sort_values("timestamp")


def build(tk):
    book = load_book(tk)
    px = full_series(tk)[["timestamp", "high", "low", "close", "volume"]]
    d = book.merge(px, on="timestamp", how="inner").reset_index(drop=True)
    # book features: level + short memory + interactions
    for col in ("imb", "micro_dev", "spread_bps", "quote_hz"):
        d[f"{col}_m1"] = d[col].shift(1)
        d[f"{col}_m3"] = d[col].rolling(3).mean().shift(1)
        d[f"{col}_m12"] = d[col].rolling(12).mean().shift(1)
    d["imb_run"] = (np.sign(d["imb"]) == np.sign(d["imb"].shift(1))).rolling(6) \
        .sum()
    d["press"] = d["imb"] * d["micro_dev"]
    d["ret1"] = d["close"].pct_change()
    d["ret6"] = d["close"].pct_change(6)
    d["vol6"] = d["ret1"].rolling(6).std()
    d["book_v_flow"] = d["imb"] - d["imb_m12"]
    return d


FEATS = ["imb", "micro_dev", "spread_bps", "quote_hz", "bs", "as_",
         "imb_m1", "imb_m3", "imb_m12", "micro_dev_m1", "micro_dev_m3",
         "micro_dev_m12", "spread_bps_m1", "spread_bps_m3", "spread_bps_m12",
         "quote_hz_m1", "quote_hz_m3", "quote_hz_m12",
         "imb_run", "press", "ret1", "ret6", "vol6", "book_v_flow"]


def label(d, tgt, stop, H):
    c = d["close"].to_numpy()
    h = d["high"].to_numpy()
    l = d["low"].to_numpy()
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        up, dn = c[i] * (1 + tgt), c[i] * (1 - stop)
        for j in range(i + 1, min(i + H + 1, n)):
            hu, hd = h[j] >= up, l[j] <= dn
            if hu and hd:
                break
            if hd:
                y[i] = 0; break
            if hu:
                y[i] = 1; break
    return y


def main():
    tk = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "QQQ"
    tgt, stop, H = arg("--tgt", 0.003), arg("--stop", 0.002), int(arg("--H", 24))
    q = arg("--q", 0.90)
    d = build(tk)
    print(f"vOB {tk}: {len(d)} joined bars "
          f"({d['timestamp'].min()} .. {d['timestamp'].max()})")
    y = label(d, tgt, stop, H)
    X = d[FEATS]
    fin = X.notna().all(axis=1).to_numpy() & np.isfinite(y)
    months = sorted(d.loc[fin, "timestamp"].dt.to_period("M").unique())
    if len(months) < 3:
        raise SystemExit("need >=3 months of book data — extend the backfill")
    print(f"walk-forward over {len(months)} months | bracket +{tgt:.1%}/-{stop:.1%} "
          f"H={H} gate top-{(1 - q) * 100:.0f}%\n")
    allr = []
    per = d["timestamp"].dt.to_period("M")
    for k in range(2, len(months)):
        tr_m = months[:k]
        te_m = months[k]
        tr = fin & per.isin(tr_m).to_numpy()
        te = fin & (per == te_m).to_numpy()
        tr_idx = np.where(tr)[0]
        tr_idx = tr_idx[:-H] if len(tr_idx) > H else tr_idx
        if len(tr_idx) < 800 or y[tr_idx].sum() < 30:
            continue
        clf = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04,
                                 num_leaves=15, min_child_samples=40,
                                 subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, verbose=-1)
        clf.fit(X.iloc[tr_idx], y[tr_idx].astype(int))
        thr = np.quantile(clf.predict_proba(X.iloc[tr_idx])[:, 1], q)
        te_idx = np.where(te)[0]
        p = clf.predict_proba(X.iloc[te_idx])[:, 1]
        c = d["close"].to_numpy()
        h = d["high"].to_numpy()
        l = d["low"].to_numpy()
        rets = []
        last_exit = -1
        for ii, i in enumerate(te_idx):
            if p[ii] < thr or i <= last_exit:
                continue
            up, dn = c[i] * (1 + tgt), c[i] * (1 - stop)
            res, j = None, i + 1
            while j < min(i + H + 1, len(c)):
                if l[j] <= dn:
                    res = 0; break
                if h[j] >= up:
                    res = 1; break
                j += 1
            ex = min(j, len(c) - 1)
            px = up if res == 1 else (dn if res == 0 else c[ex])
            rets.append((px - c[i]) / c[i] - COST)
            last_exit = ex
        r = np.array(rets)
        allr.append(r)
        if len(r):
            print(f"  {te_m}: n={len(r):>3} win={(r > 0).mean():>4.0%} "
                  f"avg={r.mean() * 1e4:>+6.1f}bp total={r.sum() * 100:>+6.2f}%")
        else:
            print(f"  {te_m}: no trades above gate")
        # feature relevance on the last fold
        if k == len(months) - 1:
            imp = sorted(zip(FEATS, clf.feature_importances_),
                         key=lambda z: -z[1])[:8]
            print("\n  top features:", ", ".join(f"{a}({b})" for a, b in imp))
    if allr:
        r = np.concatenate(allr)
        t = r.mean() / r.std() * np.sqrt(len(r)) if len(r) > 1 and r.std() > 0 else 0
        print(f"\nOOS TOTAL: n={len(r)} win={(r > 0).mean():.0%} "
              f"avg={r.mean() * 1e4:+.1f}bp total={r.sum() * 100:+.2f}% t={t:+.2f}")


if __name__ == "__main__":
    main()
