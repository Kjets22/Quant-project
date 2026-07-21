"""
morning_hf.py — HIGH-FREQUENCY MORNING SWEEP: QQQ 9:30->12:00, hard noon exit,
targeting a trade-day fraction >= 0.48 overall AND in the gate year AND in the
final year (a trade-day = a day with >= 1 fill).

Champion vM (orb2s or5 R2 NR.3) only fires ~26% of days; this round widens the
entry set / loosens the filter WITHOUT changing the honesty ladder:
  ARENA = worst of FIVE half-years 2022-01-14..2024-07-14 (must be > 0)
  GATE  = 2024-07-14..2025-07-14, ONE look for arena survivors only
  FINAL = 2025-07-14..now, one-shot for gate survivors. Never tuned on gate/final.
Headline cost 2 bps ROUND TRIP PER FILL (multi-leg days pay cost per leg);
0/5 bps sensitivity on finalists. Research only.

Families (~242 configs, every arena line printed):
  F1  orb2s with loose/no NR compression filter (nr None/.5/.7/.85 — the rank is
      quantized in 1/6 steps over a 7d window, so .5/.6 and .7/.8 collapse; the
      distinct effective buckets are used and fire rates are printed first)
  F2  orb2s gated by a CONTINUOUS vol filter: prior-day range / 60d median <= k
  F3  STOP-AND-REVERSE ORB: on breakout stop-out, flip at the stop price into the
      opposite direction (stop = other OR extreme, target rr x OR-range), max 2
      fills/day, noon exit
  F4  ENSEMBLE books: validated core orb2s(5, 2.0, nr=0.3) on compression days +
      a secondary rule on the OTHER days (P&L = sum of legs, cost per leg)
  F5  ORB-else-revert: plain two-sided ORB; if NO fill by 11:00, fade the
      deviation from morning VWAP into noon (monetizes chop days)

Usage:  python morning_hf.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from morning_qqq import (COST, SUBS, GATE, FINAL, load, days, window,
                         bracket_to_noon, gap_rule, vwap_rev)
from morning_qqq2 import f30_follow, od_follow
from morning_qqq3 import add_nr
from morning_qqq4 import orb2s

OVERALL = ("2022-01-14", "2099-01-01")     # frequency measured over the eval span
FREQ_MIN = 0.48                            # trade-day fraction floor (all 3 spans)


# --------------------------------------------------------------- extra day context
def add_volratio(dd, look=60, min_p=20):
    """vol_ratio = prior-day range / median of trailing `look` prior-day ranges.
    Everything is known at the open (uses prev_high/prev_low only)."""
    rngs = []
    for d in dd:
        rng = (d["prev_high"] - d["prev_low"]) if d["prev_high"] is not None else None
        rngs.append(rng)
        wnd = [x for x in rngs[-look:] if x is not None]
        if rng is None or len(wnd) < min_p:
            d["vol_ratio"] = None
        else:
            d["vol_ratio"] = rng / float(np.median(wnd))


def volr_ok(day, k):
    vr = day.get("vol_ratio")
    return vr is not None and vr <= k


# ----------------------------------------------------------------- new strategies
# All STRATS callables return (gross_return, n_fills) or None.  gross is BEFORE
# costs; the harness charges cost * n_fills so 0/2/5bps sensitivity is exact even
# on 2-fill (stop-and-reverse / ensemble) days.
def g1(net):
    """Wrap a net-of-COST single-fill return (existing module convention)."""
    return None if net is None else (net + COST, 1)


def orb2s_cap(day, or_bars, rr, risk_cap, last_entry=690):
    """orb2s clone with a configurable risk cap (skip wide-OR days). Net of COST."""
    am = day["am"]
    if len(am) < or_bars + 3:
        return None
    hi = float(am["high"].iloc[:or_bars].max())
    lo = float(am["low"].iloc[:or_bars].min())
    c = am["close"].to_numpy(); m = am["min"].to_numpy()
    h = am["high"].to_numpy(); l = am["low"].to_numpy()
    for i in range(or_bars, len(am)):
        if m[i] > last_entry:
            return None
        if c[i] > hi:
            e = c[i]; risk = e - lo
            if risk <= 0 or risk / e > risk_cap:
                return None
            tgt = e + rr * risk if rr else None
            return bracket_to_noon(am, i, tgt, lo, entry_px=e) - COST
        if c[i] < lo:
            e = c[i]; risk = hi - e
            if risk <= 0 or risk / e > risk_cap:
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


def _leg2(am, j0, e, stop_px, tgt_px, short):
    """Walk the reversal leg from its (intrabar) entry bar j0.  Conservative: on
    the entry bar only the ADVERSE stop can fire (bar order unknown); from j0+1
    stop is checked before target, noon close fallback.  Gross return."""
    h = am["high"].to_numpy(); l = am["low"].to_numpy(); c = am["close"].to_numpy()
    if short:
        if h[j0] >= stop_px:
            return (e - stop_px) / e
        for j in range(j0 + 1, len(am)):
            if h[j] >= stop_px:
                return (e - stop_px) / e
            if tgt_px is not None and l[j] <= tgt_px:
                return (e - tgt_px) / e
        return (e - c[-1]) / e
    if l[j0] <= stop_px:
        return (stop_px - e) / e
    for j in range(j0 + 1, len(am)):
        if l[j] <= stop_px:
            return (stop_px - e) / e
        if tgt_px is not None and h[j] >= tgt_px:
            return (tgt_px - e) / e
    return (c[-1] - e) / e


def orb_sar(day, or_bars, rr, last_entry=690, risk_cap=0.012):
    """Two-sided ORB with STOP-AND-REVERSE: first breakout as in orb2s (stop =
    far OR extreme, target rr x risk, noon exit); on stop-out FLIP at the stop
    price into the opposite direction (stop = other extreme, target rr x
    OR-range, noon exit).  Max 2 fills/day.  Returns (gross, n_fills)."""
    am = day["am"]
    if len(am) < or_bars + 3:
        return None
    hi = float(am["high"].iloc[:or_bars].max())
    lo = float(am["low"].iloc[:or_bars].min())
    rng = hi - lo
    c = am["close"].to_numpy(); m = am["min"].to_numpy()
    h = am["high"].to_numpy(); l = am["low"].to_numpy()
    for i in range(or_bars, len(am)):
        if m[i] > last_entry:
            return None
        if c[i] > hi:
            e = c[i]; risk = e - lo
            if risk <= 0 or risk / e > risk_cap:
                return None
            tgt = e + rr * risk if rr else None
            for j in range(i + 1, len(am)):
                if l[j] <= lo:                       # stopped -> flip short at lo
                    r1 = (lo - e) / e
                    t2 = lo - rr * rng if rr else None
                    return (r1 + _leg2(am, j, lo, hi, t2, short=True), 2)
                if tgt is not None and h[j] >= tgt:
                    return ((tgt - e) / e, 1)
            return ((c[-1] - e) / e, 1)
        if c[i] < lo:
            e = c[i]; risk = hi - e
            if risk <= 0 or risk / e > risk_cap:
                return None
            tgt = e - rr * risk if rr else None
            for j in range(i + 1, len(am)):
                if h[j] >= hi:                       # stopped -> flip long at hi
                    r1 = (e - hi) / e
                    t2 = hi + rr * rng if rr else None
                    return (r1 + _leg2(am, j, hi, lo, t2, short=False), 2)
                if tgt is not None and l[j] <= tgt:
                    return ((e - tgt) / e, 1)
            return ((e - c[-1]) / e, 1)
    return None


def orb_else_rev(day, or_bars, rr, rev_start=660, band=0.001, stop_pct=0.004):
    """Plain two-sided ORB; if it produced NO fill, the morning is range-bound:
    at the first bar >= 11:00 fade the deviation from morning VWAP (>= band),
    exit at VWAP touch, stop, or noon.  Returns (gross, n_fills)."""
    # ORB entries must stop BEFORE the fade decision time — a real-time trader
    # deciding at rev_start cannot know about a later ORB fill (audit 2026-07-21:
    # the default last_entry=690 here was lookahead, inflating OER by +0.2..+1.7%).
    r = orb2s(day, or_bars, rr, last_entry=rev_start)
    if r is not None:
        return (r + COST, 1)
    am = day["am"]
    if len(am) < 10:
        return None
    tp = (am["high"] + am["low"] + am["close"]) / 3
    vw = ((tp * am["volume"]).cumsum() / am["volume"].cumsum()).to_numpy()
    c = am["close"].to_numpy(); m = am["min"].to_numpy()
    h = am["high"].to_numpy(); l = am["low"].to_numpy()
    for i in range(len(am)):
        if m[i] < rev_start:
            continue
        e = c[i]; dev = (e - vw[i]) / vw[i]
        if dev <= -band:                             # long back to VWAP
            stop_px = e * (1 - stop_pct)
            for j in range(i + 1, len(am)):
                if l[j] <= stop_px:
                    return ((stop_px - e) / e, 1)
                if h[j] >= vw[j]:
                    return ((vw[j] - e) / e, 1)
            return ((c[-1] - e) / e, 1)
        if dev >= band:                              # short back to VWAP
            stop_px = e * (1 + stop_pct)
            for j in range(i + 1, len(am)):
                if h[j] >= stop_px:
                    return ((e - stop_px) / e, 1)
                if l[j] <= vw[j]:
                    return ((e - vw[j]) / e, 1)
            return ((e - c[-1]) / e, 1)
        return None                                  # inside the band: stand aside
    return None


def ensemble(day, secondary):
    """Core orb2s(5,2,NR<=.3) on compression days; `secondary` (gross-form
    callable) on all OTHER days.  Day P&L = the leg that owns the day."""
    nr = day.get("nr_rank")
    if nr is None:
        return None
    if nr <= 0.3:
        r = orb2s(day, 5, 2.0, nr=0.3)
        return None if r is None else (r + COST, 1)
    return secondary(day)


# ------------------------------------------------------------------- config grid
def build_strats():
    S = []
    rr_grid = [(1.5, "1.5"), (2.0, "2"), (3.0, "3"), (None, "n")]
    # F1: loose/no NR filter (distinct quantization buckets of the 7d rank)
    for nr, ntag in [(None, "-"), (0.5, ".5"), (0.7, ".7"), (0.85, ".85")]:
        for ob in (3, 4, 5, 6):
            for rr, rtag in rr_grid:
                for le in (660, 690):
                    S.append((f"O{ob} R{rtag} nr{ntag} e{le}",
                              lambda d, ob=ob, rr=rr, nr=nr, le=le:
                              g1(orb2s(d, ob, rr, nr=nr, last_entry=le))))
    # F2: continuous vol filter (prior-day range / 60d median <= k)
    for k in (0.8, 1.0, 1.2):
        for ob in (3, 4, 5, 6):
            for rr, rtag in rr_grid:
                S.append((f"O{ob} R{rtag} vr{k}",
                          lambda d, ob=ob, rr=rr, k=k:
                          g1(orb2s(d, ob, rr)) if volr_ok(d, k) else None))
    # F3: stop-and-reverse ORB (unfiltered + vol-filtered)
    for k in (None, 1.0, 1.2):
        ktag = "" if k is None else f" vr{k}"
        for ob in (3, 4, 5, 6):
            for rr, rtag in rr_grid:
                S.append((f"SAR{ob} R{rtag}{ktag}",
                          lambda d, ob=ob, rr=rr, k=k:
                          orb_sar(d, ob, rr) if (k is None or volr_ok(d, k))
                          else None))
    # F4: ensembles — core orb2s(5,2,NR.3) + secondary on non-compression days
    secondaries = [
        ("cap.6R2",  lambda d: g1(orb2s_cap(d, 5, 2.0, 0.006))),
        ("cap.8R2",  lambda d: g1(orb2s_cap(d, 5, 2.0, 0.008))),
        ("cap.6R3",  lambda d: g1(orb2s_cap(d, 5, 3.0, 0.006))),
        ("cap.8R3",  lambda d: g1(orb2s_cap(d, 5, 3.0, 0.008))),
        ("orbR2",    lambda d: g1(orb2s(d, 5, 2.0))),
        ("sar5R2",   lambda d: orb_sar(d, 5, 2.0)),
        ("sar5Rn",   lambda d: orb_sar(d, 5, None)),
        ("sar3R2",   lambda d: orb_sar(d, 3, 2.0)),
        ("vwap.2",   lambda d: g1(vwap_rev(d, 0.002, 0.004))),
        ("vwap.3",   lambda d: g1(vwap_rev(d, 0.003, 0.004))),
        ("f30.1",    lambda d: g1(f30_follow(d, 0.001, 0.004))),
        ("f30.2",    lambda d: g1(f30_follow(d, 0.002, 0.004))),
        ("gapU",     lambda d: g1(gap_rule(d, 0.001, 0.010, None, 0.003))),
        ("odf.2",    lambda d: g1(od_follow(d, 0.002, 0.004))),
    ]
    for tag, sec in secondaries:
        S.append((f"ENS+{tag}", lambda d, sec=sec: ensemble(d, sec)))
    # F5: ORB else VWAP-reversion at 11:00
    for ob in (3, 5):
        for rr, rtag in [(2.0, "2"), (None, "n")]:
            S.append((f"OER{ob} R{rtag}",
                      lambda d, ob=ob, rr=rr: orb_else_rev(d, ob, rr)))
    return S


# ---------------------------------------------------------------------- harness
def win_idx(dd, lo, hi):
    lo, hi = pd.Timestamp(lo).date(), pd.Timestamp(hi).date()
    return [i for i, d in enumerate(dd) if lo <= d["date"] < hi]


def stats(res, idx, cost=COST):
    """(net array at `cost` per fill, trade-day fraction, total fills)."""
    fills = [(res[i][0], res[i][1]) for i in idx if res[i] is not None]
    net = np.array([g - n * cost for g, n in fills])
    legs = int(sum(n for _, n in fills))
    freq = len(fills) / len(idx) if idx else 0.0
    return net, freq, legs


def main():
    df = load()
    dd = days(df)
    add_nr(dd)
    add_volratio(dd)
    strats = build_strats()

    ov_idx = win_idx(dd, *OVERALL)
    g_idx = win_idx(dd, *GATE)
    f_idx = win_idx(dd, *FINAL)
    sub_idx = [win_idx(dd, lo, hi) for lo, hi in SUBS]

    print(f"MORNING-HF: {len(strats)} configs | 9:30->12:00, hard noon exit | "
          f"cost {COST*1e4:.0f}bps PER FILL | freq floor {FREQ_MIN:.2f} "
          f"(overall AND gate-yr AND final-yr) | {len(dd)} days "
          f"{dd[0]['date']}..{dd[-1]['date']}\n")

    # --- fire-rate diagnostics (day-filter pass rates BEFORE any ORB trigger) ---
    span = [dd[i] for i in ov_idx]
    ranks = sorted({round(d["nr_rank"], 3) for d in span if d["nr_rank"] is not None})
    print(f"nr_rank quantization over {OVERALL[0]}..now: {ranks}")
    print("  nr<=thr day-fraction: " + "  ".join(
        f"{t}:{np.mean([d['nr_rank'] is not None and d['nr_rank'] <= t for d in span]):.2f}"
        for t in (0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85)))
    print("  vol_ratio<=k day-fraction: " + "  ".join(
        f"{k}:{np.mean([volr_ok(d, k) for d in span]):.2f}"
        for k in (0.8, 1.0, 1.2)) + "\n")

    # ----------------------------------------------------------------- ARENA ---
    print(f"  {'strategy':<20} {'worst':>7}  {'subs 2022H1..2024H1':<42} "
          f"freq ov/gt/fin   flags")
    rows = []
    for name, fn in strats:
        res = [fn(d) for d in dd]
        subs, ns = [], []
        for idx in sub_idx:
            r, _, _ = stats(res, idx)
            ns.append(len(r))
            subs.append(float(r.sum() * 100) if len(r) >= 8 else -99.0)
        worst = min(subs)
        _, fo, _ = stats(res, ov_idx)
        _, fg, _ = stats(res, g_idx)
        _, ff, _ = stats(res, f_idx)
        freq_ok = fo >= FREQ_MIN and fg >= FREQ_MIN and ff >= FREQ_MIN
        flags = ("A" if worst > 0 else ".") + ("F" if freq_ok else ".")
        print(f"  {name:<20} {worst:>+6.2f}%  {str([round(s,1) for s in subs]):<42} "
              f"{fo:.2f}/{fg:.2f}/{ff:.2f}   {flags}"
              f"{'  <-- ARENA PASS' if worst > 0 else ''}")
        rows.append(dict(name=name, fn=fn, res=res, subs=subs, worst=worst,
                         freq=(fo, fg, ff), freq_ok=freq_ok))

    arena_pass = [r for r in rows if r["worst"] > 0]
    n_freq = sum(r["freq_ok"] for r in rows)
    print(f"\narena survivors: {len(arena_pass)} | freq-floor compliant: {n_freq} | "
          f"both: {sum(r['freq_ok'] for r in arena_pass)}")

    # ------------------------------------------------------------------ GATE ---
    print(f"\n=== GATE {GATE[0]}..{GATE[1]} ({len(arena_pass)} arena survivors, "
          f"one look) ===")
    gate_pass = []
    for r in arena_pass:
        g, _, _ = stats(r["res"], g_idx)
        tot = g.sum() * 100 if len(g) >= 10 else -99
        r["gate"] = tot
        print(f"  {r['name']:<20} n={len(g):>3} win={(g > 0).mean() if len(g) else 0:.0%} "
              f"tot={tot:+.2f}% [{'freq-ok' if r['freq_ok'] else 'LOW-FREQ'}]"
              f"{'  <-- GATE PASS' if tot > 0 else ''}")
        if tot > 0:
            gate_pass.append(r)

    # ----------------------------------------------------------------- FINAL ---
    print(f"\n=== FINAL one-shot {FINAL[0]}..now ({len(gate_pass)} gate survivors) ===")
    finalists = []
    for r in gate_pass:
        f2, _, legs = stats(r["res"], f_idx)
        if len(f2) < 10:
            print(f"  {r['name']:<20} too few trades ({len(f2)})")
            continue
        f0, _, _ = stats(r["res"], f_idx, cost=0.0)
        f5, _, _ = stats(r["res"], f_idx, cost=5e-4)
        t = f2.mean() / f2.std() * np.sqrt(len(f2)) if f2.std() > 0 else 0.0
        r.update(final=f2.sum() * 100, final_n=len(f2), final_t=t,
                 final0=f0.sum() * 100, final5=f5.sum() * 100,
                 final_win=(f2 > 0).mean(), final_legs=legs)
        finalists.append(r)
        print(f"  {r['name']:<20} n={len(f2):>3} fills={legs:>3} "
              f"win={(f2 > 0).mean():.0%} avg={f2.mean()*1e4:+.1f}bp "
              f"tot={f2.sum()*100:+.2f}% t={t:+.2f} | 0bps {f0.sum()*100:+.2f}% / "
              f"5bps {f5.sum()*100:+.2f}% [{'freq-ok' if r['freq_ok'] else 'LOW-FREQ'}]")

    # ------------------------------------------------------------- SELECTION ---
    champs = sorted([r for r in finalists if r["freq_ok"]],
                    key=lambda r: (r["final"], r["final_t"], r["final5"]),
                    reverse=True)
    print("\n=== SELECTION (full ladder AND freq floor) ===")
    if champs:
        print(f"{len(champs)} configs pass full ladder + freq floor "
              f"(of {len(strats)} tested)")
        for r in champs[:8]:
            print(f"  {r['name']:<20} final {r['final']:+.2f}% (n={r['final_n']}, "
                  f"t={r['final_t']:+.2f}, 5bps {r['final5']:+.2f}%) "
                  f"freq {r['freq'][0]:.2f}/{r['freq'][1]:.2f}/{r['freq'][2]:.2f} "
                  f"gate {r['gate']:+.2f}% arena_worst {r['worst']:+.2f}%")
        champion_report(dd, champs[0])
    else:
        print("NO config passes BOTH the full ladder and the frequency floor.")
        fc = sorted([r for r in rows if r["freq_ok"]],
                    key=lambda r: r["worst"], reverse=True)
        if fc:
            r = fc[0]
            print(f"best freq-compliant near-miss (arena only, no gate peek): "
                  f"{r['name']} arena_worst {r['worst']:+.2f}% subs "
                  f"{[round(s,1) for s in r['subs']]} freq "
                  f"{r['freq'][0]:.2f}/{r['freq'][1]:.2f}/{r['freq'][2]:.2f}")
        lp = sorted(finalists, key=lambda r: r["final"], reverse=True)
        if lp:
            r = lp[0]
            print(f"best ladder-passer regardless of freq: {r['name']} final "
                  f"{r['final']:+.2f}% (n={r['final_n']}, t={r['final_t']:+.2f}) "
                  f"freq {r['freq'][0]:.2f}/{r['freq'][1]:.2f}/{r['freq'][2]:.2f}")


def champion_report(dd, r):
    print(f"\n=== CHAMPION DETAIL: {r['name']} ===")
    res = r["res"]
    for yr in (2022, 2023, 2024, 2025, 2026):
        idx = win_idx(dd, f"{yr}-01-01", f"{yr + 1}-01-01")
        if not idx:
            continue
        net, fq, legs = stats(res, idx)
        if not len(net):
            continue
        print(f"  {yr}: n={len(net):>3} fills={legs:>3} freq={fq:.2f} "
              f"win={(net > 0).mean():.0%} avg={net.mean()*1e4:+.1f}bp "
              f"tot={net.sum()*100:+.2f}%")
    ov = win_idx(dd, *OVERALL)
    net, fq, legs = stats(res, ov)
    yrs = len(ov) / 252
    print(f"  overall {OVERALL[0]}..now: n={len(net)} fills={legs} "
          f"({len(net)/yrs:.0f} trade-days/yr, {legs/yrs:.0f} fills/yr) "
          f"freq={fq:.2f} win={(net > 0).mean():.0%} "
          f"avg={net.mean()*1e4:+.1f}bp/trade-day tot={net.sum()*100:+.2f}%")
    print(f"  final: {r['final0']:+.2f}% @0bps / {r['final']:+.2f}% @2bps / "
          f"{r['final5']:+.2f}% @5bps  (n={r['final_n']}, fills={r['final_legs']}, "
          f"win={r['final_win']:.0%}, t={r['final_t']:+.2f})")


if __name__ == "__main__":
    main()
