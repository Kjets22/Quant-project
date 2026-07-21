"""
morning_qqq3.py — ROUND 3: CROSS-ASSET morning signals + two-sided ORB + range filters.

Round 1-2 (61 configs): QQQ-only price rules all fail — morning drift flips sign by era.
New information here (never tested in this project for the morning window):
  * SPY-QQQ premarket RELATIVE strength (tech vs market lead-lag)
  * NVDA premarket move (QQQ's dominant driver) -> QQQ morning
  * TLT overnight move (rates shock) -> QQQ morning, incl. regime-dependent sign
  * Two-sided opening-range breakout (long break of OR-high, SHORT break of OR-low),
    noon exit — quant_rth only ever tested long-only ORB held to the close
  * NR4/NR7 narrow-range-day filter (compression -> expansion) on top of ORB
Same ladder, 2bps, hard noon exit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from morning_qqq import (COST, SUBS, GATE, FINAL, load, days, window, run,
                         bracket_to_noon, OPEN)
from morning_qqq2 import long_noon, short_noon


def load_aux(tk):
    base = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv",
                       parse_dates=["timestamp"])
    parts = [base]
    for p in sorted(Path("data_cache").glob(f"{tk}_recent_*.csv")):
        parts.append(pd.read_csv(p, parse_dates=["timestamp"]))
    df = (pd.concat(parts, ignore_index=True)
          .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
          .reset_index(drop=True))
    from zoneinfo import ZoneInfo
    ts = pd.DatetimeIndex(df["timestamp"]).tz_localize("UTC").tz_convert(
        ZoneInfo("America/New_York"))
    df["date"] = ts.date
    df["min"] = ts.hour * 60 + ts.minute
    out, prev_close = {}, None
    for d, g in df.groupby("date", sort=True):
        rth = g[(g["min"] >= OPEN) & (g["min"] < 960)]
        pm = g[g["min"] < OPEN]
        if prev_close and len(pm):
            out[d] = (float(pm["close"].iloc[-1]) - prev_close) / prev_close
        if len(rth):
            prev_close = float(rth["close"].iloc[-1])
    return out          # date -> premarket return vs prev RTH close


def attach_aux(dd, aux, key):
    for d in dd:
        d[key] = aux.get(d["date"])


# ------------------------------------------------------------ cross-asset rules
def rel_strength(day, thr, fade=False):
    """QQQ pm vs SPY pm: rs>thr -> tech leading -> long (or fade -> short)."""
    if day["pm_ret"] is None or day.get("spy_pm") is None:
        return None
    rs = day["pm_ret"] - day["spy_pm"]
    if rs > thr:
        return short_noon(day, 0) if fade else long_noon(day, 0)
    if rs < -thr:
        return long_noon(day, 0) if fade else short_noon(day, 0)
    return None


def nvda_lead(day, thr, fade=False):
    if day.get("nvda_pm") is None:
        return None
    if day["nvda_pm"] > thr:
        return short_noon(day, 0) if fade else long_noon(day, 0)
    if day["nvda_pm"] < -thr:
        return long_noon(day, 0) if fade else short_noon(day, 0)
    return None


def tlt_shock(day, thr, fade=False):
    """TLT overnight move -> QQQ morning (rates lead growth stocks)."""
    if day.get("tlt_pm") is None:
        return None
    if day["tlt_pm"] > thr:
        return short_noon(day, 0) if fade else long_noon(day, 0)
    if day["tlt_pm"] < -thr:
        return long_noon(day, 0) if fade else short_noon(day, 0)
    return None


# ----------------------------------------------------------- two-sided ORB / NR
def orb2(day, or_bars, rr, last_entry=690, nr=None):
    """Two-sided opening-range breakout, bracket rr x risk, noon exit fallback."""
    am = day["am"]
    if len(am) < or_bars + 3:
        return None
    if nr is not None:                       # narrow-range prior-day filter
        if day.get("nr_rank") is None or day["nr_rank"] > nr:
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
            if risk <= 0 or risk / e > 0.012:
                return None
            return bracket_to_noon(am, i, e + rr * risk, lo, entry_px=e) - COST
        if c[i] < lo:
            e = c[i]; risk = hi - e
            if risk <= 0 or risk / e > 0.012:
                return None
            stop_px, tgt_px = hi, e - rr * risk
            for j in range(i + 1, len(am)):
                if h[j] >= stop_px:
                    return (e - stop_px) / e - COST
                if l[j] <= tgt_px:
                    return (e - tgt_px) / e - COST
            return (e - c[-1]) / e - COST
    return None


def add_nr(dd, lookback=7):
    """nr_rank = today's-prior-day range rank within last `lookback` days (0=narrowest)."""
    ranges = []
    for d in dd:
        rng = (d["prev_high"] - d["prev_low"]) if d["prev_high"] else None
        ranges.append(rng)
        if rng is None or len(ranges) < lookback:
            d["nr_rank"] = None
            continue
        wnd = [x for x in ranges[-lookback:] if x is not None]
        d["nr_rank"] = sorted(wnd).index(rng) / max(len(wnd) - 1, 1)


STRATS = [
    ("RS 0.10 follow",    lambda d: rel_strength(d, 0.0010)),
    ("RS 0.20 follow",    lambda d: rel_strength(d, 0.0020)),
    ("RS 0.10 fade",      lambda d: rel_strength(d, 0.0010, fade=True)),
    ("RS 0.20 fade",      lambda d: rel_strength(d, 0.0020, fade=True)),
    ("NVDA 0.5 follow",   lambda d: nvda_lead(d, 0.005)),
    ("NVDA 1.0 follow",   lambda d: nvda_lead(d, 0.010)),
    ("NVDA 1.0 fade",     lambda d: nvda_lead(d, 0.010, fade=True)),
    ("TLT 0.3 follow",    lambda d: tlt_shock(d, 0.003)),
    ("TLT 0.3 fade",      lambda d: tlt_shock(d, 0.003, fade=True)),
    ("TLT 0.5 follow",    lambda d: tlt_shock(d, 0.005)),
    ("ORB2 15m R2",       lambda d: orb2(d, 3, 2.0)),
    ("ORB2 30m R1.5",     lambda d: orb2(d, 6, 1.5)),
    ("ORB2 30m R2",       lambda d: orb2(d, 6, 2.0)),
    ("ORB2 30m R3",       lambda d: orb2(d, 6, 3.0)),
    ("ORB2 30m R2 NR.3",  lambda d: orb2(d, 6, 2.0, nr=0.3)),
    ("ORB2 15m R2 NR.3",  lambda d: orb2(d, 3, 2.0, nr=0.3)),
    ("ORB2 30m R2 NR.5",  lambda d: orb2(d, 6, 2.0, nr=0.5)),
]


def main():
    df = load()
    dd = days(df)
    add_nr(dd)
    for tk, key in (("SPY", "spy_pm"), ("NVDA", "nvda_pm"), ("TLT", "tlt_pm")):
        attach_aux(dd, load_aux(tk), key)

    print(f"ROUND 3: {len(STRATS)} configs (cumulative: {61 + len(STRATS)}) | "
          f"cross-asset + ORB2 + NR | 2bps\n")
    print(f"  {'strategy':<20} {'worst':>7}  {'subs 2022H1..2024H1':<40} n/half")
    arena_pass = []
    for name, fn in STRATS:
        subs, ns = [], []
        for lo, hi in SUBS:
            r = run(window(dd, lo, hi), fn)
            ns.append(len(r))
            subs.append(float(r.sum() * 100) if len(r) >= 8 else -99.0)
        worst = min(subs)
        flag = "  <-- ARENA PASS" if worst > 0 else ""
        print(f"  {name:<20} {worst:>+6.2f}%  {str([round(s,1) for s in subs]):<40} "
              f"{ns}{flag}")
        if worst > 0:
            arena_pass.append((name, fn))
    print(f"\n=== GATE {GATE[0]}..{GATE[1]} ({len(arena_pass)} survivors) ===")
    gate_pass = []
    for name, fn in arena_pass:
        r = run(window(dd, *GATE), fn)
        tot = r.sum() * 100 if len(r) >= 10 else -99
        print(f"  {name:<20} n={len(r):>3} win={(r > 0).mean() if len(r) else 0:.0%} "
              f"tot={tot:+.2f}%{'  <-- GATE PASS' if tot > 0 else ''}")
        if tot > 0:
            gate_pass.append((name, fn))
    print(f"\n=== FINAL one-shot {FINAL[0]}..now ({len(gate_pass)} survivors) ===")
    for name, fn in gate_pass:
        r = run(window(dd, *FINAL), fn)
        if len(r) < 10:
            print(f"  {name:<20} too few trades ({len(r)})")
            continue
        t = r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else 0
        r0 = r + COST
        print(f"  {name:<20} n={len(r):>3} win={(r > 0).mean():.0%} "
              f"avg={r.mean()*1e4:+.1f}bp tot={r.sum()*100:+.2f}% t={t:+.2f} | "
              f"0bps {r0.sum()*100:+.2f}% / 5bps {(r0 - 5e-4).sum()*100:+.2f}%")


if __name__ == "__main__":
    main()
