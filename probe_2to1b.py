"""
probe_2to1b.py — $2/$1, 2h clock, FAIR gate for asymmetric payoffs: top-quantile of
P(win) (like v3/v4), not the symmetric confidence gate (which DQ'd everything because
P(win) centers on the ~25% base rate, not 0.5). Same 3-stage ladder.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from evolve3 import gproba, data, SUBS, TRAIN_END, GATE0, GATE1, EFF_COST, gkey

BASE = {"feat": "F4", "tgt": 2.0, "stop": 1.0, "H": 24, "rth": 0}
VARIANTS = [dict(BASE, model=m, q=q) for m in ("lgbm_s", "histgb")
            for q in (0.90, 0.93, 0.97)]


def sim_q(g, proba, lo, hi, train_before):
    """Trade sim with TOP-QUANTILE-of-P(win) gate."""
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
        up, dn = c[i] + g["tgt"], c[i] - g["stop"]
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
            rets.append((g["tgt"] if res == 1 else -g["stop"]) / c[i] - EFF_COST)
        i = j + 1
    return rets


def stats(rets):
    r = np.array(rets)
    if len(r) == 0:
        return 0, 0.0, 0.0, 0.0
    return len(r), float((r > 0).mean()), float(r.mean() * 1e4), float(r.sum() * 100)


def main():
    print("PROBE-B: QQQ +$2 / -$1, 2h clock, top-quantile gate (break-even 33.3% pre-cost)")
    best, best_fit = None, -999
    for g in VARIANTS:
        proba = gproba(g, TRAIN_END)
        if proba is None:
            print(f"  {g['model']:>7} q={g['q']:.2f}: no model (label too sparse)")
            continue
        subs = []
        ok = True
        for lo, hi in SUBS:
            n, w, m, t = stats(sim_q(g, proba, lo, hi, TRAIN_END))
            if n < 12:
                ok = False
                break
            subs.append(t)
        fit = min(subs) if ok else -99.0
        print(f"  {g['model']:>7} q={g['q']:.2f}: fit={fit:+7.2f}%  "
              f"subs={[round(s, 2) for s in subs] if ok else 'DQ (too few trades)'}")
        if fit > best_fit:
            best_fit, best = fit, g
    print(f"\nBest: {gkey(best)}  arena fit {best_fit:+.2f}%")
    if best_fit <= 0:
        print("VERDICT: $2/$1-in-2h is not profitable in every arena half-year even at the "
              "best variant — the $1 stop still dies to noise. Geometry rejected fairly.")
        return
    print("\n=== GATE (2024-07-14..2025-07-14) ===")
    p = gproba(best, GATE0)
    n, w, m, t = stats(sim_q(best, p, GATE0, GATE1, GATE0))
    print(f"  n={n} win%={w:.1%} mean={m:+.1f}bps total={t:+.2f}%")
    if t <= 0:
        print("VERDICT: failed the gate.")
        return
    print("\n=== FINAL one-shot (2025-07-14..now) ===")
    p = gproba(best, GATE1)
    n, w, m, t = stats(sim_q(best, p, GATE1, "2099-01-01", GATE1))
    print(f"  PROBE : n={n} win%={w:.1%} mean={m:+.1f}bps total={t:+.2f}%")
    print("  vP ref: n=279 win%=59.5% mean=+1.5bps total=+4.18%")
    print("  vQ ref: n=106 win%=61.3% mean=+2.6bps total=+2.77%")


if __name__ == "__main__":
    main()
