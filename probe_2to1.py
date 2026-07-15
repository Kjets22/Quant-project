"""
probe_2to1.py — user-specified geometry: QQQ, +$2 target / -$1 stop, 2-hour clock (H=24).
Tests 6 variants (2 models x 3 confidence gates) on the Evolution-III arena fitness
(worst of three half-year P&Ls), then the best variant goes through the same GATE and
FINAL ladder, reported next to vP and vQ for context. Break-even win rate: 1/(1+2) = 33.3%
before costs (~38-40% after, given the small $1 stop vs 5 bps cost).
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from evolve3 import fitness, gproba, sim, GATE0, GATE1, gkey

BASE = {"feat": "F4", "tgt": 2.0, "stop": 1.0, "H": 24, "rth": 0}
VARIANTS = [dict(BASE, model=m, q=q) for m in ("lgbm_s", "histgb")
            for q in (0.85, 0.90, 0.95)]


def trade_stats(g, proba, lo, hi, train_before):
    rets = sim(g, proba, lo, hi, train_before)
    r = np.array(rets)
    n = len(r)
    if n == 0:
        return 0, 0, 0, 0
    return n, float((r > 0).mean()), float(r.mean() * 1e4), float(r.sum() * 100)


def main():
    print("PROBE: QQQ +$2 / -$1, 2-hour clock — arena fitness (worst of 3 half-years)")
    best, best_fit = None, -999
    for g in VARIANTS:
        fit, subs = fitness(g)
        print(f"  {g['model']:>7} q={g['q']:.2f}: fit={fit:+7.2f}%  "
              f"subs={[round(s, 2) for s in subs] if subs else 'DQ'}")
        if fit > best_fit:
            best_fit, best = fit, g
    print(f"\nBest variant: {gkey(best)}  (arena fit {best_fit:+.2f}%)")
    if best_fit <= 0:
        print("VERDICT: no variant is profitable in every arena half-year — geometry fails "
              "the first bar; gate/final skipped.")
        return
    print("\n=== GATE (2024-07-14..2025-07-14) ===")
    p = gproba(best, GATE0)
    n, w, m, t = trade_stats(best, p, GATE0, GATE1, GATE0)
    print(f"  n={n} win%={w:.1%} mean={m:+.1f}bps total={t:+.2f}%")
    if t <= 0:
        print("VERDICT: failed the gate — arena result did not generalize.")
        return
    print("\n=== FINAL one-shot (2025-07-14..now) ===")
    p = gproba(best, GATE1)
    n, w, m, t = trade_stats(best, p, GATE1, "2099-01-01", GATE1)
    print(f"  PROBE : n={n} win%={w:.1%} mean={m:+.1f}bps total={t:+.2f}%")
    print("  vP ref: n=279 win%=59.5% mean=+1.5bps total=+4.18%")
    print("  vQ ref: n=106 win%=61.3% mean=+2.6bps total=+2.77%")


if __name__ == "__main__":
    main()
