"""
probe_pct.py — user spec: +0.4% target / -0.2% stop on QQQ (percentage barriers — era-
consistent, unlike fixed dollars). 2:1 payoff, real break-even ~41.7% after 5 bps costs.
Variants: {lgbm_s, histgb} x {top-7%, top-3% gate} x {2h, 4h clock}. Three-stage ladder:
arena (worst of 3 half-years) -> gate 2024-25 -> one-shot final vs vP/vQ references.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from evolve3 import data, MODELS3, SUBS, TRAIN_END, GATE0, GATE1, EFF_COST, CACHE

TGT_P, STOP_P = 0.004, 0.002
VARIANTS = [{"model": m, "q": q, "H": H} for m in ("lgbm_s", "histgb")
            for q in (0.93, 0.97) for H in (24, 48)]


def label_pct(H):
    f = CACHE / f"y_pct_{TGT_P}_{STOP_P}_{H}.npy"
    if f.exists():
        return np.load(f)
    ts, feats, h, l, c, hours = data()
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        up, dn = c[i] * (1 + TGT_P), c[i] * (1 - STOP_P)
        for j in range(i + 1, min(i + H + 1, n)):
            hu, hd = h[j] >= up, l[j] <= dn
            if hu and hd:
                break
            if hd:
                y[i] = 0; break
            if hu:
                y[i] = 1; break
    np.save(f, y)
    return y


def proba_pct(g, train_before):
    f = CACHE / f"p_pct_{g['model']}_{g['H']}_{train_before}.npy"
    if f.exists():
        return np.load(f)
    ts, feats, h, l, c, hours = data()
    X = feats["F4"]
    y = label_pct(g["H"])
    fin = X.notna().all(axis=1).to_numpy()
    tr = np.where(fin & np.isfinite(y) & (ts < np.datetime64(train_before)))[0][:-g["H"]]
    clf = MODELS3[g["model"]]()
    clf.fit(X.iloc[tr], y[tr].astype(int))
    proba = np.full(len(X), np.nan)
    ok = np.where(fin)[0]
    proba[ok] = clf.predict_proba(X.iloc[ok])[:, 1]
    np.save(f, proba)
    return proba


def sim_pct(g, proba, lo, hi, train_before):
    ts, feats, h, l, c, hours = data()
    tr_mask = np.isfinite(proba) & (ts < np.datetime64(train_before))
    thr = np.quantile(proba[tr_mask], g["q"])
    idx = np.where(np.isfinite(proba) & (ts >= np.datetime64(lo)) & (ts < np.datetime64(hi)))[0]
    if len(idx) == 0:
        return []
    rets = []
    n = len(c)
    i, last = int(idx[0]), int(idx[-1])
    while i <= last:
        if not np.isfinite(proba[i]) or proba[i] < thr:
            i += 1; continue
        up, dn = c[i] * (1 + TGT_P), c[i] * (1 - STOP_P)
        res, j = None, i + 1
        while j < min(i + g["H"] + 1, n):
            hu, hd = h[j] >= up, l[j] <= dn
            if (hu and hd) or hd:
                res = 0; break
            if hu:
                res = 1; break
            j += 1
        ex = min(j, n - 1)
        if res is None:
            rets.append((c[ex] - c[i]) / c[i] - EFF_COST)
        else:
            rets.append((TGT_P if res == 1 else -STOP_P) - EFF_COST)
        i = j + 1
    return rets


def stats(rets):
    r = np.array(rets)
    if len(r) == 0:
        return 0, 0.0, 0.0, 0.0
    return len(r), float((r > 0).mean()), float(r.mean() * 1e4), float(r.sum() * 100)


def main():
    print(f"PROBE-PCT: QQQ +{TGT_P:.1%} / -{STOP_P:.1%} (2:1), real break-even ~41.7%")
    best, best_fit = None, -999
    for g in VARIANTS:
        proba = proba_pct(g, TRAIN_END)
        subs, ok = [], True
        for lo, hi in SUBS:
            n, w, m, t = stats(sim_pct(g, proba, lo, hi, TRAIN_END))
            if n < 12:
                ok = False; break
            subs.append(t)
        fit = min(subs) if ok else -99.0
        print(f"  {g['model']:>7} q={g['q']:.2f} H={g['H']:>2}: fit={fit:+7.2f}%  "
              f"subs={[round(s, 2) for s in subs] if ok else 'DQ'}", flush=True)
        if fit > best_fit:
            best_fit, best = fit, g
    print(f"\nBest: {best}  arena fit {best_fit:+.2f}%")
    if best_fit <= 0:
        print("VERDICT: not profitable in every arena half-year — rejected at stage 1.")
        return
    print("=== GATE (2024-07-14..2025-07-14) ===")
    p = proba_pct(best, GATE0)
    n, w, m, t = stats(sim_pct(best, p, GATE0, GATE1, GATE0))
    print(f"  n={n} win%={w:.1%} mean={m:+.1f}bps total={t:+.2f}%")
    if t <= 0:
        print("VERDICT: failed the gate.")
        return
    print("=== FINAL one-shot (2025-07-14..now) ===")
    p = proba_pct(best, GATE1)
    n, w, m, t = stats(sim_pct(best, p, GATE1, "2099-01-01", GATE1))
    print(f"  PROBE : n={n} win%={w:.1%} mean={m:+.1f}bps total={t:+.2f}%")
    print("  vP ref: n=279 win%=59.5% mean=+1.5bps total=+4.18%")
    print("  vQ ref: n=106 win%=61.3% mean=+2.6bps total=+2.77%")


if __name__ == "__main__":
    main()
