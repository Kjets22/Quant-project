"""
morning_exits.py — EXIT-MANAGEMENT SWEEP on the frozen champion vM entries.

vM = orb2s(or_bars=5, rr=2.0, nr=0.3, last_entry=690, sides='both'):
  NR<=0.3 prior-day compression filter, opening range = first 5 bars (25 min),
  enter on first 5-min close beyond the range (entries until 11:30),
  stop = far side of range, target = 2x risk, hard exit 12:00.

This module re-implements the ENTRY exactly (verified per-trade against
morning_qqq4.orb2s before any sweeping) and then applies exit overlays to the
SAME entries — entries never change, one trade per day.

Honesty rules implemented:
  * within a 5-min bar the STOP always resolves before the TARGET;
  * multi-level exits inside one bar resolve worst-first (stop, then scale
    levels ascending in favourability, then the rest-target);
  * state changes (breakeven arming, trailing-stop updates) only take effect
    on the NEXT bar after the trigger bar completes;
  * indicator exits (inside-OR, VWAP, time stops, consecutive-closes) fire at
    bar close.
Trailing stops are never looser than the original OR stop (stated for t1;
applied to t2 as well so risk never exceeds the defined 1x).

Decisive statistic per overlay: paired per-trade t of (overlay - baseline).
An overlay counts as an improvement only if delta > 0 in arena_worst AND gate
AND final AND paired t > 1.5.

Research only. Usage:  python morning_exits.py
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from morning_qqq import COST, SUBS, GATE, FINAL, load, days, window
from morning_qqq3 import add_nr
from morning_qqq4 import orb2s

OR_BARS, RR, NR, LAST_ENTRY = 5, 2.0, 0.3, 690


# --------------------------------------------------------------------- entry
def vm_entry(day):
    """Exact re-implementation of the orb2s(5, 2.0, nr=0.3) ENTRY.
    Returns dict(i0, e, dir, hi, lo, risk) or None. Never changes per overlay."""
    am = day["am"]
    if len(am) < OR_BARS + 3:
        return None
    if day.get("nr_rank") is None or day["nr_rank"] > NR:
        return None
    hi = float(am["high"].iloc[:OR_BARS].max())
    lo = float(am["low"].iloc[:OR_BARS].min())
    c = am["close"].to_numpy()
    m = am["min"].to_numpy()
    for i in range(OR_BARS, len(am)):
        if m[i] > LAST_ENTRY:
            return None
        if c[i] > hi:
            e = c[i]
            risk = e - lo
            if risk <= 0 or risk / e > 0.012:
                return None
            return {"i0": i, "e": e, "dir": 1, "hi": hi, "lo": lo, "risk": risk}
        if c[i] < lo:
            e = c[i]
            risk = hi - e
            if risk <= 0 or risk / e > 0.012:
                return None
            return {"i0": i, "e": e, "dir": -1, "hi": hi, "lo": lo, "risk": risk}
    return None


# ----------------------------------------------------------------- simulator
def simulate(day, ent, scales=(), rest_target=RR, be_arm=None, trail=None,
             exit_or=False, exit_vwap=False, time_stop=None, consec=None):
    """Walk bars after entry applying the overlay's exit rules.

    scales      : ((fraction, mult), ...) — limit exits at entry + mult*risk
    rest_target : risk-multiple target for the remaining position (None = noon)
    be_arm      : move stop to breakeven once +mult risk touched (arms NEXT bar)
    trail       : trailing stop = extreme close -/+ mult*risk (updates NEXT bar),
                  never looser than the OR stop
    exit_or     : exit at bar close if close comes back inside the opening range
    exit_vwap   : exit at bar close if close crosses morning VWAP against position
    time_stop   : (minutes, mult) — one-shot check at first close >= minutes after
                  entry: exit if P&L < mult*risk
    consec      : exit after `consec` consecutive closes against the position

    Returns (net_return_after_cost, legs) with legs = [(weight, price, reason)].
    """
    am = day["am"]
    i0, e, d, risk = ent["i0"], ent["e"], ent["dir"], ent["risk"]
    hi, lo = ent["hi"], ent["lo"]
    h = am["high"].to_numpy(); l = am["low"].to_numpy()
    c = am["close"].to_numpy(); m = am["min"].to_numpy()
    vw = None
    if exit_vwap:
        tp = (am["high"] + am["low"] + am["close"]) / 3
        vw = ((tp * am["volume"]).cumsum() / am["volume"].cumsum()).to_numpy()

    def lvl(mult):                       # price at +mult*risk in favour
        return e + d * mult * risk

    base_stop = lo if d == 1 else hi
    stop_px, stop_kind = base_stop, "stop"   # effective for the CURRENT bar
    be_armed = False
    ext_close = e                        # best close since entry (incl. entry bar)
    pending = sorted([list(s) for s in scales], key=lambda s: s[1])
    legs, w = [], 1.0
    ts_done, n_adv, prev_close = False, 0, e

    for j in range(i0 + 1, len(am)):
        # 1) intrabar stop — always resolves before any target/scale (worst-first)
        if (l[j] <= stop_px) if d == 1 else (h[j] >= stop_px):
            legs.append((w, stop_px, stop_kind))
            w = 0.0
            break
        # 2) scale levels ascending, then rest-target (worst-first ordering)
        while pending:
            frac, mult = pending[0]
            px = lvl(mult)
            if not ((h[j] >= px) if d == 1 else (l[j] <= px)):
                break
            legs.append((frac, px, f"scale{mult:g}x"))
            w -= frac
            pending.pop(0)
        if w > 1e-12 and rest_target is not None:
            px = lvl(rest_target)
            if (h[j] >= px) if d == 1 else (l[j] <= px):
                legs.append((w, px, "target"))
                w = 0.0
                break
        # 3) close-based early-outs (fire at bar close)
        out = None
        if exit_or and ((c[j] < hi) if d == 1 else (c[j] > lo)):
            out = "eOR"
        if out is None and exit_vwap and \
                ((c[j] < vw[j]) if d == 1 else (c[j] > vw[j])):
            out = "eVWAP"
        if out is None and time_stop is not None and not ts_done \
                and m[j] - m[i0] >= time_stop[0]:
            ts_done = True
            if d * (c[j] - e) < time_stop[1] * risk:
                out = "eTIME"
        if consec is not None:
            n_adv = n_adv + 1 if d * (c[j] - prev_close) < 0 else 0
            if out is None and n_adv >= consec:
                out = "eCONSEC"
        if out is not None:
            legs.append((w, c[j], out))
            w = 0.0
            break
        # 4) state updates — arm on the NEXT bar only
        if be_arm is not None and not be_armed:
            px = lvl(be_arm)
            if (h[j] >= px) if d == 1 else (l[j] <= px):
                be_armed = True
        ext_close = max(ext_close, c[j]) if d == 1 else min(ext_close, c[j])
        new_stop, kind = base_stop, "stop"
        if be_armed and ((e > new_stop) if d == 1 else (e < new_stop)):
            new_stop, kind = e, "be"
        if trail is not None:
            tlv = ext_close - d * trail * risk
            if (tlv > new_stop) if d == 1 else (tlv < new_stop):
                new_stop, kind = tlv, "trail"
        stop_px, stop_kind = new_stop, kind
        prev_close = c[j]

    if w > 1e-12:
        legs.append((w, c[-1], "noon"))
    ret = sum(wk * d * (px - e) / e for wk, px, _ in legs) - COST
    return ret, legs


# ------------------------------------------------------------------ overlays
OVERLAYS = [
    # name , cfg dict , description
    ("s1", dict(scales=((0.5, 1.0),), rest_target=None),
     "50% off at +1.0x, rest to noon, stop unchanged"),
    ("s2", dict(scales=((0.5, 1.0),), rest_target=None, be_arm=1.0),
     "50% off at +1.0x, stop->BE for rest (arms next bar)"),
    ("s3", dict(scales=((0.5, 1.0), (0.25, 2.0)), rest_target=None),
     "50% at +1x, 25% at +2x, rest to noon, stop unchanged"),
    ("s4", dict(scales=((0.5, 0.5),), rest_target=None),
     "50% off at +0.5x, rest to noon"),
    ("b1", dict(be_arm=1.0), "stop->BE once +1.0x touched (arms next bar)"),
    ("b2", dict(be_arm=0.5), "stop->BE once +0.5x touched (arms next bar)"),
    ("t1", dict(trail=1.0), "trail = best close -1.0x risk, floor = OR stop"),
    ("t2", dict(trail=1.5), "trail = best close -1.5x risk, floor = OR stop"),
    ("e1", dict(exit_or=True), "exit on close back inside opening range"),
    ("e2", dict(exit_vwap=True), "exit on close across morning VWAP"),
    ("e3", dict(time_stop=(60, 0.0)), "time stop: out 60min after entry if P&L<0"),
    ("e4", dict(time_stop=(45, 0.25)), "time stop: out 45min if P&L<+0.25x"),
    ("e5", dict(consec=3), "exit after 3 consecutive adverse closes"),
    ("c1", dict(scales=((0.5, 1.0),), rest_target=None, exit_or=True),
     "s1 + e1"),
    ("c2", dict(scales=((0.5, 1.0),), rest_target=None, be_arm=1.0),
     "s1 + b1 (== s2 by construction)"),
    ("c3", dict(exit_or=True, time_stop=(60, 0.0)), "e1 + e3"),
    ("c4", dict(be_arm=1.0, exit_or=True), "b1 + e1 (extra combo)"),
    ("c5", dict(be_arm=1.0, time_stop=(60, 0.0)), "b1 + e3 (extra combo)"),
]


# ------------------------------------------------------------------- metrics
def metrics(dates, r):
    r = np.asarray(r, dtype=float)
    tot = float(r.sum() * 100)
    win = float((r > 0).mean())
    avg = float(r.mean() * 1e4)
    eq = np.cumsum(r) * 100
    peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
    maxdd = float(np.max(peak - eq))
    s = pd.Series(r, index=pd.to_datetime([str(d) for d in dates]))
    mo = s.resample("ME").sum()
    sharpe = float(mo.mean() / mo.std() * np.sqrt(12)) if mo.std() > 0 else 0.0
    return tot, win, avg, sharpe, maxdd


def wsum(dates, r, lo, hi):
    lo, hi = pd.Timestamp(lo).date(), pd.Timestamp(hi).date()
    return float(sum(x for d, x in zip(dates, r) if lo <= d < hi) * 100)


def ladder(dates, r):
    arena = min(wsum(dates, r, lo, hi) for lo, hi in SUBS)
    gate = wsum(dates, r, *GATE)
    final = wsum(dates, r, *FINAL)
    return arena, gate, final


# ---------------------------------------------------------------------- main
def main():
    df = load()
    dd = days(df)
    add_nr(dd)

    # -------- entries (fixed for every overlay) --------
    entries = []
    for day in dd:
        ent = vm_entry(day)
        if ent is not None:
            entries.append((day, ent))
    dates = [day["date"] for day, _ in entries]

    # -------- baseline through the overlay engine + verification --------
    base_r, base_legs = [], []
    for day, ent in entries:
        ret, legs = simulate(day, ent)          # stop=OR, target=2x, noon
        base_r.append(ret)
        base_legs.append(legs)
    base_r = np.array(base_r)

    # per-trade equality vs the frozen champion implementation
    ref = {}
    for day in dd:
        v = orb2s(day, OR_BARS, RR, nr=NR, last_entry=LAST_ENTRY, sides="both")
        if v is not None:
            ref[day["date"]] = v
    assert set(ref) == set(dates), (
        f"entry mismatch: engine {len(dates)} vs orb2s {len(ref)} trades")
    for d, r in zip(dates, base_r):
        assert abs(r - ref[d]) < 1e-12, f"return mismatch on {d}: {r} vs {ref[d]}"

    n = len(base_r)
    tot, win, avg, sharpe, maxdd = metrics(dates, base_r)
    assert 300 <= n <= 330, f"baseline n={n} not ~315"
    assert 19.5 <= tot <= 24.0, f"baseline total={tot:.2f}% not ~+21.7%"
    assert 0.54 <= win <= 0.62, f"baseline win={win:.1%} not ~58%"
    arena, gate, final = ladder(dates, base_r)

    print(f"vM EXIT-MANAGEMENT SWEEP | {n} fixed entries "
          f"{dates[0]}..{dates[-1]} | cost {COST*1e4:.0f}bps")
    print(f"BASELINE verified vs orb2s per-trade (max |diff| < 1e-12)")
    print(f"BASELINE n={n} tot={tot:+.2f}% win={win:.1%} avg={avg:+.1f}bp "
          f"Sharpe={sharpe:.2f} maxDD={maxdd:.2f}% | "
          f"arena_worst={arena:+.2f}% gate={gate:+.2f}% final={final:+.2f}%\n")

    hdr = (f"  {'ov':<3} {'n':>3} {'tot%':>8} {'win':>6} {'avg_bp':>7} "
           f"{'Sharpe':>7} {'maxDD':>6} {'arena':>7} {'gate':>7} {'final':>7} "
           f"{'t_pair':>7} {'dTot':>7}  verdict")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows, breakdowns = [], {}
    for name, cfg, desc in OVERLAYS:
        r, all_legs = [], []
        for day, ent in entries:
            ret, legs = simulate(day, ent, **cfg)
            r.append(ret)
            all_legs.append(legs)
        r = np.array(r)
        o_tot, o_win, o_avg, o_sh, o_dd = metrics(dates, r)
        o_ar, o_ga, o_fi = ladder(dates, r)
        dvec = r - base_r
        sd = dvec.std(ddof=1)
        t = float(dvec.mean() / sd * np.sqrt(len(dvec))) if sd > 0 else 0.0
        improve = (o_ar > arena) and (o_ga > gate) and (o_fi > final) and t > 1.5
        verdict = "IMPROVE" if improve else ""
        print(f"  {name:<3} {len(r):>3} {o_tot:>+8.2f} {o_win:>6.1%} "
              f"{o_avg:>+7.1f} {o_sh:>7.2f} {o_dd:>6.2f} {o_ar:>+7.2f} "
              f"{o_ga:>+7.2f} {o_fi:>+7.2f} {t:>+7.2f} "
              f"{o_tot - tot:>+7.2f}  {verdict}  [{desc}]")
        # exit breakdown: final-leg reason per trade + scale-fill counts
        fin_reason = Counter(legs[-1][2] for legs in all_legs)
        scale_fills = Counter(rn for legs in all_legs for _, _, rn in legs
                              if rn.startswith("scale"))
        breakdowns[name] = {"final_leg": dict(fin_reason),
                            "scale_fills": dict(scale_fills)}
        rows.append(dict(name=name, desc=desc, n=len(r), total=round(o_tot, 2),
                         win=round(o_win, 3), avg_bp=round(o_avg, 1),
                         sharpe=round(o_sh, 2), maxDD=round(o_dd, 2),
                         arena_worst=round(o_ar, 2), gate=round(o_ga, 2),
                         final=round(o_fi, 2), paired_t=round(t, 2),
                         delta_total=round(o_tot - tot, 2),
                         d_arena=round(o_ar - arena, 2),
                         d_gate=round(o_ga - gate, 2),
                         d_final=round(o_fi - final, 2),
                         improve=improve))

    print("\n=== EXIT BREAKDOWNS (final-leg reason counts; scale fills separate) ===")
    for name, cfg, desc in OVERLAYS:
        b = breakdowns[name]
        sf = f" | scale fills {b['scale_fills']}" if b["scale_fills"] else ""
        print(f"  {name}: {b['final_leg']}{sf}")
    bl_fin = Counter(legs[-1][2] for legs in base_legs)
    print(f"  baseline: {dict(bl_fin)}")

    imp = [x for x in rows if x["improve"]]
    imp.sort(key=lambda x: -x["paired_t"])
    print(f"\n=== IMPROVEMENTS (dArena>0 & dGate>0 & dFinal>0 & t>1.5): "
          f"{[x['name'] for x in imp] or 'NONE'} ===")
    return dict(baseline=dict(n=n, total=round(tot, 2), win=round(win, 3),
                              avg_bp=round(avg, 1), sharpe=round(sharpe, 2),
                              maxDD=round(maxdd, 2), arena_worst=round(arena, 2),
                              gate=round(gate, 2), final=round(final, 2)),
                rows=rows, improvements=imp, breakdowns=breakdowns,
                baseline_breakdown=dict(bl_fin))


if __name__ == "__main__":
    main()
