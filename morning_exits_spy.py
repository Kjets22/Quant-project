"""
morning_exits_spy.py — TRANSFER CHECK of the exit-management sweep onto SPY.

The QQQ exit sweep (morning_exits.py) produced NO overlay passing the improvement
bar (dArena>0 & dGate>0 & dFinal>0 & t>1.5). Per protocol we transfer-check the
3 overlays with the HIGHEST paired-t from that table:
    b1  stop->BE once +1.0x touched (arms next bar)      t=+0.93
    t1  trail = best close -1.0x risk, floor = OR stop   t=-0.03
    t2  trail = best close -1.5x risk, floor = OR stop   t=-0.23

Machinery is imported from morning_exits (entry re-implementation vm_entry,
honest bar-walk simulator, metrics, ladder) — nothing is duplicated. SPY data
comes via morning_validate.load_ticker('SPY'). Both tickers run through the
exact same code path; QQQ is re-run here only to compute the reference deltas
next to SPY's so direction-match is computed, not quoted.

An exit tweak that only helps on QQQ is likely noise; one that helps on both
is structure. Research only. Usage:  python morning_exits_spy.py
"""

from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from morning_qqq import COST, load, days
from morning_qqq3 import add_nr
from morning_qqq4 import orb2s
from morning_validate import load_ticker
import morning_exits as ME

TOP = ["b1", "t1", "t2"]          # highest paired-t on QQQ (see docstring)
OVL = {name: (cfg, desc) for name, cfg, desc in ME.OVERLAYS}


def run_ticker(dd, label):
    """Fixed vM entries + baseline + TOP overlays through morning_exits.simulate.
    Baseline is verified per-trade against morning_qqq4.orb2s on THIS ticker."""
    entries = []
    for day in dd:
        ent = ME.vm_entry(day)
        if ent is not None:
            entries.append((day, ent))
    dates = [day["date"] for day, _ in entries]

    base_r = np.array([ME.simulate(day, ent)[0] for day, ent in entries])

    # per-trade verification vs the frozen champion implementation on this ticker
    ref = {}
    for day in dd:
        v = orb2s(day, ME.OR_BARS, ME.RR, nr=ME.NR,
                  last_entry=ME.LAST_ENTRY, sides="both")
        if v is not None:
            ref[day["date"]] = v
    assert set(ref) == set(dates), (
        f"{label}: entry mismatch engine {len(dates)} vs orb2s {len(ref)}")
    for d, r in zip(dates, base_r):
        assert abs(r - ref[d]) < 1e-12, f"{label} {d}: {r} vs {ref[d]}"

    tot, win, avg, sharpe, maxdd = ME.metrics(dates, base_r)
    arena, gate, final = ME.ladder(dates, base_r)
    print(f"{label} BASELINE vM (verified vs orb2s per-trade): n={len(base_r)} "
          f"{dates[0]}..{dates[-1]}")
    print(f"  tot={tot:+.2f}% win={win:.1%} avg={avg:+.1f}bp Sharpe={sharpe:.2f} "
          f"maxDD={maxdd:.2f}% | arena_worst={arena:+.2f}% gate={gate:+.2f}% "
          f"final={final:+.2f}%")

    out = {"n": len(base_r), "tot": tot, "arena": arena, "gate": gate,
           "final": final, "rows": {}}
    for name in TOP:
        cfg, desc = OVL[name]
        r = np.array([ME.simulate(day, ent, **cfg)[0] for day, ent in entries])
        o_tot, o_win, o_avg, o_sh, o_dd = ME.metrics(dates, r)
        o_ar, o_ga, o_fi = ME.ladder(dates, r)
        dvec = r - base_r
        sd = dvec.std(ddof=1)
        t = float(dvec.mean() / sd * np.sqrt(len(dvec))) if sd > 0 else 0.0
        n_ch = int((np.abs(dvec) > 1e-12).sum())
        out["rows"][name] = dict(
            tot=o_tot, d_tot=o_tot - tot, d_arena=o_ar - arena,
            d_gate=o_ga - gate, d_final=o_fi - final, t=t,
            win=o_win, n_changed=n_ch, desc=desc)
    return out


def main():
    print("=== TRANSFER CHECK: top-3 paired-t exit overlays, QQQ -> SPY ===\n")

    qdd = days(load())
    add_nr(qdd)
    q = run_ticker(qdd, "QQQ")
    print()
    sdd = days(load_ticker("SPY"))
    add_nr(sdd)
    s = run_ticker(sdd, "SPY")

    hdr = (f"\n  {'ov':<3} {'tk':<4} {'dTot%':>7} {'dArena':>7} {'dGate':>7} "
           f"{'dFinal':>7} {'t_pair':>7} {'nChg':>5}  description")
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    for name in TOP:
        for tk, res in (("QQQ", q), ("SPY", s)):
            r = res["rows"][name]
            print(f"  {name:<3} {tk:<4} {r['d_tot']:>+7.2f} {r['d_arena']:>+7.2f} "
                  f"{r['d_gate']:>+7.2f} {r['d_final']:>+7.2f} {r['t']:>+7.2f} "
                  f"{r['n_changed']:>5}  [{r['desc']}]")
        qr, sr = q["rows"][name], s["rows"][name]
        marks = []
        for k, lab in (("d_tot", "dTot"), ("d_gate", "dGate"),
                       ("d_final", "dFinal"), ("t", "t")):
            same = np.sign(qr[k]) == np.sign(sr[k])
            marks.append(f"{lab}:{'MATCH' if same else 'FLIP'}")
        print(f"      -> direction vs QQQ: {'  '.join(marks)}")
    print("\n(An overlay is structural only if it helps on BOTH tickers; "
          "sign flips on SPY mark the QQQ move as noise.)")
    return dict(qqq=q, spy=s)


if __name__ == "__main__":
    main()
