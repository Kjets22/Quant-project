"""
morning_qqq4.py — ROUND 4: probe the near-miss family — two-sided 15-min ORB with
narrow-range (compression) filter. Round 3's "ORB2 15m R2 NR.3" hit worst=-0.51% with
4/5 halves positive; this round maps its neighborhood (target ratio, NR cutoff, entry
deadline, sides, vol filter) plus TLT-fade with stops. Same ladder, 2bps, noon exit.
Discipline: arena selection only; ONE gate look for survivors; gate-failures are dead.
"""

from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from morning_qqq import COST, SUBS, GATE, FINAL, load, days, window, run, bracket_to_noon
from morning_qqq2 import long_noon, short_noon
from morning_qqq3 import load_aux, attach_aux, add_nr


def orb2s(day, or_bars, rr, nr=None, last_entry=690, sides="both", volx=None):
    am = day["am"]
    if len(am) < or_bars + 3:
        return None
    if nr is not None and (day.get("nr_rank") is None or day["nr_rank"] > nr):
        return None
    if volx is not None and (day.get("pm_vol_x") is None or day["pm_vol_x"] < volx):
        return None
    hi = float(am["high"].iloc[:or_bars].max())
    lo = float(am["low"].iloc[:or_bars].min())
    c = am["close"].to_numpy(); m = am["min"].to_numpy()
    h = am["high"].to_numpy(); l = am["low"].to_numpy()
    for i in range(or_bars, len(am)):
        if m[i] > last_entry:
            return None
        if c[i] > hi and sides in ("both", "long"):
            e = c[i]; risk = e - lo
            if risk <= 0 or risk / e > 0.012:
                return None
            tgt = e + rr * risk if rr else None
            return bracket_to_noon(am, i, tgt, lo, entry_px=e) - COST
        if c[i] < lo and sides in ("both", "short"):
            e = c[i]; risk = hi - e
            if risk <= 0 or risk / e > 0.012:
                return None
            stop_px = hi
            tgt_px = e - rr * risk if rr else None
            for j in range(i + 1, len(am)):
                if h[j] >= stop_px:
                    return (e - stop_px) / e - COST
                if tgt_px is not None and l[j] <= tgt_px:
                    return (e - tgt_px) / e - COST
            return (e - c[-1]) / e - COST
    return None


def tlt_fade_s(day, thr, stop_pct):
    if day.get("tlt_pm") is None:
        return None
    if day["tlt_pm"] > thr:
        return short_noon(day, 0, stop_pct)
    if day["tlt_pm"] < -thr:
        return long_noon(day, 0, stop_pct)
    return None


STRATS = [
    ("ORB2-15 R1.5 NR.3",      lambda d: orb2s(d, 3, 1.5, nr=0.3)),
    ("ORB2-15 R2 NR.3",        lambda d: orb2s(d, 3, 2.0, nr=0.3)),
    ("ORB2-15 R3 NR.3",        lambda d: orb2s(d, 3, 3.0, nr=0.3)),
    ("ORB2-15 noon NR.3",      lambda d: orb2s(d, 3, None, nr=0.3)),
    ("ORB2-15 R2 NR.4",        lambda d: orb2s(d, 3, 2.0, nr=0.4)),
    ("ORB2-15 R2 NR.2",        lambda d: orb2s(d, 3, 2.0, nr=0.2)),
    ("ORB2-15 R2 NR.3 e10:30", lambda d: orb2s(d, 3, 2.0, nr=0.3, last_entry=630)),
    ("ORB2-15 R2 NR.3 LONG",   lambda d: orb2s(d, 3, 2.0, nr=0.3, sides="long")),
    ("ORB2-15 R2 NR.3 SHORT",  lambda d: orb2s(d, 3, 2.0, nr=0.3, sides="short")),
    ("ORB2-15 R2 vol1.2x",     lambda d: orb2s(d, 3, 2.0, volx=1.2)),
    ("ORB2-15 R2 NR.3 v1.2x",  lambda d: orb2s(d, 3, 2.0, nr=0.3, volx=1.2)),
    ("ORB2-10 R2 NR.3",        lambda d: orb2s(d, 2, 2.0, nr=0.3)),
    ("ORB2-25 R2 NR.3",        lambda d: orb2s(d, 5, 2.0, nr=0.3)),
    ("TLTX 0.3 s0.4",          lambda d: tlt_fade_s(d, 0.003, 0.004)),
    ("TLTX 0.5 s0.4",          lambda d: tlt_fade_s(d, 0.005, 0.004)),
    ("TLTX 0.5 s0.6",          lambda d: tlt_fade_s(d, 0.005, 0.006)),
]


def main():
    df = load()
    dd = days(df)
    add_nr(dd)
    attach_aux(dd, load_aux("TLT"), "tlt_pm")

    print(f"ROUND 4: {len(STRATS)} configs (cumulative: {78 + len(STRATS)}) | "
          f"ORB2-15/NR probe + TLT-fade stops | 2bps\n")
    print(f"  {'strategy':<24} {'worst':>7}  {'subs 2022H1..2024H1':<40} n/half")
    arena_pass = []
    for name, fn in STRATS:
        subs, ns = [], []
        for lo, hi in SUBS:
            r = run(window(dd, lo, hi), fn)
            ns.append(len(r))
            subs.append(float(r.sum() * 100) if len(r) >= 8 else -99.0)
        worst = min(subs)
        flag = "  <-- ARENA PASS" if worst > 0 else ""
        print(f"  {name:<24} {worst:>+6.2f}%  {str([round(s,1) for s in subs]):<40} "
              f"{ns}{flag}")
        if worst > 0:
            arena_pass.append((name, fn))
    print(f"\n=== GATE {GATE[0]}..{GATE[1]} ({len(arena_pass)} survivors) ===")
    gate_pass = []
    for name, fn in arena_pass:
        r = run(window(dd, *GATE), fn)
        tot = r.sum() * 100 if len(r) >= 10 else -99
        print(f"  {name:<24} n={len(r):>3} win={(r > 0).mean() if len(r) else 0:.0%} "
              f"tot={tot:+.2f}%{'  <-- GATE PASS' if tot > 0 else ''}")
        if tot > 0:
            gate_pass.append((name, fn))
    print(f"\n=== FINAL one-shot {FINAL[0]}..now ({len(gate_pass)} survivors) ===")
    for name, fn in gate_pass:
        r = run(window(dd, *FINAL), fn)
        if len(r) < 10:
            print(f"  {name:<24} too few trades ({len(r)})")
            continue
        t = r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else 0
        r0 = r + COST
        print(f"  {name:<24} n={len(r):>3} win={(r > 0).mean():.0%} "
              f"avg={r.mean()*1e4:+.1f}bp tot={r.sum()*100:+.2f}% t={t:+.2f} | "
              f"0bps {r0.sum()*100:+.2f}% / 5bps {(r0 - 5e-4).sum()*100:+.2f}%")


if __name__ == "__main__":
    main()
