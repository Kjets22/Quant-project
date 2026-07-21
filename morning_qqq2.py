"""
morning_qqq2.py — ROUND 2: diagnose the morning regime, then two-sided FOLLOW and
REVERSAL families (round 1's long-only momentum passed the arena but died in the
2024-25 gate year -> the sign of the morning edge likely flips with regime).

Everything stays 9:30->12:00, hard noon exit, 2bps headline cost, same ladder.
Adds: per-era diagnostics, follow (long+short symmetric), fade (reversal), first-bar
rules, and a realized-vol regime switch (momentum in quiet tape, reversion in wild).
Cumulative configs tried across rounds is reported for multiplicity honesty.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from morning_qqq import (COST, SUBS, GATE, FINAL, load, days, window, run,
                         bracket_to_noon)

ERAS = SUBS + [GATE, ("2025-07-14", "2026-01-14"), ("2026-01-14", "2099-01-01")]


def long_noon(day, i0, stop_pct=None, entry_px=None):
    e = float(day["am"]["close"].iloc[i0]) if entry_px is None else entry_px
    stop_px = e * (1 - stop_pct) if stop_pct else None
    return bracket_to_noon(day["am"], i0, None, stop_px, entry_px=e) - COST


def short_noon(day, i0, stop_pct=None, entry_px=None):
    am = day["am"]
    e = float(am["close"].iloc[i0]) if entry_px is None else entry_px
    stop_px = e * (1 + stop_pct) if stop_pct else None
    h = am["high"].to_numpy(); c = am["close"].to_numpy()
    for j in range(i0 + 1, len(am)):
        if stop_px is not None and h[j] >= stop_px:
            return (e - stop_px) / e - COST
    return (e - c[-1]) / e - COST


# ---------------------------------------------------------------- follow / fade
def od_follow(day, thr, stop_pct=None):
    """Overnight direction, both ways: pm>+thr -> long, pm<-thr -> short, exit noon."""
    if day["pm_ret"] is None:
        return None
    if day["pm_ret"] > thr:
        return long_noon(day, 0, stop_pct)
    if day["pm_ret"] < -thr:
        return short_noon(day, 0, stop_pct)
    return None


def od_fade(day, thr, stop_pct=None):
    """Fade the overnight move: pm<-thr -> long, pm>+thr -> short."""
    if day["pm_ret"] is None:
        return None
    if day["pm_ret"] < -thr:
        return long_noon(day, 0, stop_pct)
    if day["pm_ret"] > thr:
        return short_noon(day, 0, stop_pct)
    return None


def f30_follow(day, thr, stop_pct=None):
    am = day["am"]
    if len(am) < 8:
        return None
    f30 = (float(am["close"].iloc[5]) - day["open"]) / day["open"]
    if f30 > thr:
        return long_noon(day, 6, stop_pct)
    if f30 < -thr:
        return short_noon(day, 6, stop_pct)
    return None


def f30_fade(day, thr, stop_pct=None):
    am = day["am"]
    if len(am) < 8:
        return None
    f30 = (float(am["close"].iloc[5]) - day["open"]) / day["open"]
    if f30 < -thr:
        return long_noon(day, 6, stop_pct)
    if f30 > thr:
        return short_noon(day, 6, stop_pct)
    return None


def f05_follow(day, thr):
    """First 5-min bar direction, enter its close, exit noon."""
    am = day["am"]
    if day["prev_close"] is None or len(am) < 3:
        return None
    f05 = (float(am["close"].iloc[0]) - float(am["open"].iloc[0])) / day["open"]
    if f05 > thr:
        return long_noon(day, 0)
    if f05 < -thr:
        return short_noon(day, 0)
    return None


def vol_switch(day, thr, vol_thr):
    """Realized-vol regime: quiet tape -> follow the first 30 min; wild -> fade it."""
    if day["atr"] is None or day["prev_close"] is None:
        return None
    atr_pct = day["atr"] / day["prev_close"]
    if atr_pct < vol_thr:
        return f30_follow(day, thr)
    return f30_fade(day, thr)


def od_f30_agree(day, thr):
    """Trade only when overnight AND first-30 agree; direction = their sign."""
    am = day["am"]
    if day["pm_ret"] is None or len(am) < 8:
        return None
    f30 = (float(am["close"].iloc[5]) - day["open"]) / day["open"]
    if day["pm_ret"] > thr and f30 > 0:
        return long_noon(day, 6)
    if day["pm_ret"] < -thr and f30 < 0:
        return short_noon(day, 6)
    return None


STRATS = [
    ("ODF 0.10",        lambda d: od_follow(d, 0.0010)),
    ("ODF 0.20",        lambda d: od_follow(d, 0.0020)),
    ("ODF 0.30",        lambda d: od_follow(d, 0.0030)),
    ("ODF 0.20 s0.4",   lambda d: od_follow(d, 0.0020, 0.004)),
    ("ODX 0.20",        lambda d: od_fade(d, 0.0020)),
    ("ODX 0.40",        lambda d: od_fade(d, 0.0040)),
    ("ODX 0.60",        lambda d: od_fade(d, 0.0060)),
    ("F30F 0.10",       lambda d: f30_follow(d, 0.0010)),
    ("F30F 0.20",       lambda d: f30_follow(d, 0.0020)),
    ("F30F 0.30",       lambda d: f30_follow(d, 0.0030)),
    ("F30F 0.20 s0.4",  lambda d: f30_follow(d, 0.0020, 0.004)),
    ("F30X 0.20",       lambda d: f30_fade(d, 0.0020)),
    ("F30X 0.40",       lambda d: f30_fade(d, 0.0040)),
    ("F30X 0.60",       lambda d: f30_fade(d, 0.0060)),
    ("F05F 0.05",       lambda d: f05_follow(d, 0.0005)),
    ("F05F 0.10",       lambda d: f05_follow(d, 0.0010)),
    ("VSW t.2 v1.2",    lambda d: vol_switch(d, 0.002, 0.012)),
    ("VSW t.2 v1.6",    lambda d: vol_switch(d, 0.002, 0.016)),
    ("AGREE 0.10",      lambda d: od_f30_agree(d, 0.0010)),
    ("AGREE 0.20",      lambda d: od_f30_agree(d, 0.0020)),
]


def main():
    df = load()
    dd = days(df)

    print("=== DIAGNOSTIC: raw morning drift by era (no strategy, 0 cost) ===")
    print(f"  {'era':<24} {'open->noon':>11} {'10:00->noon':>12} {'|pm|>0.2 follow':>16}")
    for lo, hi in ERAS:
        w = window(dd, lo, hi)
        on = np.array([(float(d['am']['close'].iloc[-1]) - d['open']) / d['open']
                       for d in w])
        tn = np.array([(float(d['am']['close'].iloc[-1]) - float(d['am']['close'].iloc[6]))
                       / float(d['am']['close'].iloc[6]) for d in w if len(d['am']) > 8])
        fo = np.array([np.sign(d['pm_ret']) * (float(d['am']['close'].iloc[-1])
                       - d['open']) / d['open']
                       for d in w if d['pm_ret'] is not None and abs(d['pm_ret']) > 0.002])
        print(f"  {lo}..{hi[:10]:<12} {on.sum()*100:>+10.2f}% {tn.sum()*100:>+11.2f}% "
              f"{fo.sum()*100:>+15.2f}%")

    print(f"\nROUND 2: {len(STRATS)} configs (cumulative with round 1: "
          f"{41 + len(STRATS)}) | 2bps\n")
    print(f"  {'strategy':<18} {'worst':>7}  {'subs 2022H1..2024H1':<40} n/half")
    arena_pass = []
    for name, fn in STRATS:
        subs, ns = [], []
        for lo, hi in SUBS:
            r = run(window(dd, lo, hi), fn)
            ns.append(len(r))
            subs.append(float(r.sum() * 100) if len(r) >= 8 else -99.0)
        worst = min(subs)
        flag = "  <-- ARENA PASS" if worst > 0 else ""
        print(f"  {name:<18} {worst:>+6.2f}%  {str([round(s,1) for s in subs]):<40} "
              f"{ns}{flag}")
        if worst > 0:
            arena_pass.append((name, fn))
    print(f"\n=== GATE {GATE[0]}..{GATE[1]} ({len(arena_pass)} survivors) ===")
    gate_pass = []
    for name, fn in arena_pass:
        r = run(window(dd, *GATE), fn)
        tot = r.sum() * 100 if len(r) >= 10 else -99
        print(f"  {name:<18} n={len(r):>3} win={(r > 0).mean() if len(r) else 0:.0%} "
              f"tot={tot:+.2f}%{'  <-- GATE PASS' if tot > 0 else ''}")
        if tot > 0:
            gate_pass.append((name, fn))
    print(f"\n=== FINAL one-shot {FINAL[0]}..now ({len(gate_pass)} survivors) ===")
    for name, fn in gate_pass:
        r = run(window(dd, *FINAL), fn)
        if len(r) < 10:
            print(f"  {name:<18} too few trades ({len(r)})")
            continue
        t = r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else 0
        r0 = r + COST
        print(f"  {name:<18} n={len(r):>3} win={(r > 0).mean():.0%} "
              f"avg={r.mean()*1e4:+.1f}bp tot={r.sum()*100:+.2f}% t={t:+.2f} | "
              f"0bps {r0.sum()*100:+.2f}% / 5bps {(r0 - 5e-4).sum()*100:+.2f}%")


if __name__ == "__main__":
    main()
