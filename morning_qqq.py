"""
morning_qqq.py — QQQ MORNING-ONLY strategies: all exposure 9:30 -> 12:00 ET, hard noon exit.

Motivation (new geometry, not a rerun of the failed RTH work):
  * Evo V/VI + quant_rth all held positions into the afternoon or close, and QQQ's
    afternoon/late-day drift is NEGATIVE in this era — that decay was part of every failure.
  * The "am" window in Evo V/VI only limited ENTRIES (<=12:30); exits ran 2-4h clocks.
    Nothing has ever been tested flat-by-noon.
  * Signals use ONLY information available at entry time: overnight gap, premarket
    return/range/volume, prior-day levels, and intra-morning price action.

Rules are hypothesis-first (documented anomaly families), param-light, and pushed through
the project's standard honesty ladder:
  ARENA = worst of FIVE half-years 2022-01..2024-07 (must be > 0)
  GATE  = 2024-07..2025-07 (arena survivors only)
  FINAL = 2025-07..now, one-shot (gate survivors only), with 0/2/5bps cost sensitivity
Headline cost = 2 bps round trip (QQQ-realistic: ~1c spread on ~$500-700 + fees);
5 bps shown for the pessimistic case. Research only — nothing here trades live.

Usage:  python morning_qqq.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

COST = 2.0 / 1e4                     # headline round-trip cost
SUBS = [("2022-01-14", "2022-07-14"), ("2022-07-14", "2023-01-14"),
        ("2023-01-14", "2023-07-14"), ("2023-07-14", "2024-01-14"),
        ("2024-01-14", "2024-07-14")]
GATE = ("2024-07-14", "2025-07-14")
FINAL = ("2025-07-14", "2099-01-01")
_ET = ZoneInfo("America/New_York")
NOON = 720                           # minutes: 12:00 ET
OPEN = 570                           # 9:30 ET


def load():
    base = pd.read_csv("data_cache/QQQ_5minute_2021-06-01_2026-06-01.csv",
                       parse_dates=["timestamp"])
    parts = [base]
    for p in sorted(Path("data_cache").glob("QQQ_recent_*.csv")):
        parts.append(pd.read_csv(p, parse_dates=["timestamp"]))
    df = (pd.concat(parts, ignore_index=True)
          .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
          .reset_index(drop=True))
    ts = pd.DatetimeIndex(df["timestamp"]).tz_localize("UTC").tz_convert(_ET)
    df["date"] = ts.date
    df["min"] = ts.hour * 60 + ts.minute
    return df


def days(df):
    """Per-day record with premarket + morning (9:30-12:00) frames and context."""
    out, prev_close, prev_high, prev_low = [], None, None, None
    pm_vols = []
    for d, g in df.groupby("date", sort=True):
        rth = g[(g["min"] >= OPEN) & (g["min"] < 960)]
        am = g[(g["min"] >= OPEN) & (g["min"] < NOON)].reset_index(drop=True)
        pm = g[g["min"] < OPEN]
        if len(am) < 20:
            if len(rth):
                prev_close = float(rth["close"].iloc[-1])
                prev_high = float(rth["high"].max())
                prev_low = float(rth["low"].min())
            continue
        o = float(am["close"].iloc[0])          # first 5-min bar close = 9:35 fill
        pm_vol = float(pm["volume"].sum()) if len(pm) else 0.0
        pm_vols.append(pm_vol)
        avg_pm_vol = float(np.mean(pm_vols[-21:-1])) if len(pm_vols) > 5 else None
        rec = {
            "date": d, "am": am, "open": o,
            "prev_close": prev_close, "prev_high": prev_high, "prev_low": prev_low,
            "gap": (o - prev_close) / prev_close if prev_close else None,
            "pm_ret": ((float(pm["close"].iloc[-1]) - prev_close) / prev_close
                       if prev_close and len(pm) else None),
            "pm_hi": float(pm["high"].max()) if len(pm) else None,
            "pm_lo": float(pm["low"].min()) if len(pm) else None,
            "pm_vol_x": (pm_vol / avg_pm_vol if avg_pm_vol and avg_pm_vol > 0 else None),
        }
        prev_close = float(rth["close"].iloc[-1])
        prev_high = float(rth["high"].max())
        prev_low = float(rth["low"].min())
        out.append(rec)
    closes = pd.Series([float(d["am"]["close"].iloc[-1]) for d in out], dtype=float)
    daily_ranges = pd.Series(
        [float(d["prev_high"] - d["prev_low"]) if d["prev_high"] else np.nan
         for d in out], dtype=float)
    atr = daily_ranges.rolling(14).mean()        # prior-day ranges -> known at open
    for ma in (50, 200):
        m = closes.rolling(ma).mean().shift(1)
        for i, d in enumerate(out):
            d[f"ma{ma}"] = float(m.iloc[i]) if np.isfinite(m.iloc[i]) else None
    for i, d in enumerate(out):
        d["atr"] = float(atr.iloc[i]) if np.isfinite(atr.iloc[i]) else None
    return out


# ------------------------------------------------------------------ exit engine
def bracket_to_noon(am, i0, tgt_px, stop_px, entry_px=None):
    """Enter at close of bar i0; walk bars to noon; stop checked before target.
    Returns per-trade return BEFORE costs."""
    e = float(am["close"].iloc[i0]) if entry_px is None else entry_px
    h = am["high"].to_numpy(); l = am["low"].to_numpy(); c = am["close"].to_numpy()
    for j in range(i0 + 1, len(am)):
        if stop_px is not None and l[j] <= stop_px:
            return (stop_px - e) / e
        if tgt_px is not None and h[j] >= tgt_px:
            return (tgt_px - e) / e
    return (c[-1] - e) / e


# ------------------------------------------------------------- strategy families
def open_drive(day, thr, stop_pct=None, short=False):
    """Enter 9:35 on premarket signal, exit noon (optional stop)."""
    if day["pm_ret"] is None:
        return None
    sig = day["pm_ret"] if not short else -day["pm_ret"]
    if sig <= thr:
        return None
    e = day["open"]
    stop_px = e * (1 - stop_pct) if (stop_pct and not short) else \
              e * (1 + stop_pct) if (stop_pct and short) else None
    if short:
        r = bracket_to_noon(day["am"], 0, None, None)
        # short with optional stop: mirror by hand
        h = day["am"]["high"].to_numpy(); c = day["am"]["close"].to_numpy()
        if stop_px is not None:
            for j in range(1, len(day["am"])):
                if h[j] >= stop_px:
                    return (e - stop_px) / e - COST
        return (e - c[-1]) / e - COST
    return bracket_to_noon(day["am"], 0, None, stop_px) - COST


def gap_rule(day, lo_g, hi_g, tgt=None, stp=None, short=False):
    """Gap-band entry at 9:35, bracket or noon exit."""
    if day["gap"] is None or not (lo_g <= day["gap"] <= hi_g):
        return None
    e = day["open"]
    if short:
        tgt_px = e * (1 - tgt) if tgt else None
        stop_px = e * (1 + stp) if stp else None
        h = day["am"]["high"].to_numpy(); l = day["am"]["low"].to_numpy()
        c = day["am"]["close"].to_numpy()
        for j in range(1, len(day["am"])):
            if stop_px is not None and h[j] >= stop_px:
                return (e - stop_px) / e - COST
            if tgt_px is not None and l[j] <= tgt_px:
                return (e - tgt_px) / e - COST
        return (e - c[-1]) / e - COST
    tgt_px = e * (1 + tgt) if tgt else None
    stop_px = e * (1 - stp) if stp else None
    return bracket_to_noon(day["am"], 0, tgt_px, stop_px) - COST


def f30_mom(day, thr, need_gap=False, stop_pct=None):
    """At 10:00, if first-30-min return > thr (and gap>0 if required): buy, exit noon."""
    am = day["am"]
    if len(am) < 8:
        return None
    f30 = (float(am["close"].iloc[5]) - day["open"]) / day["open"]
    if f30 <= thr:
        return None
    if need_gap and (day["gap"] is None or day["gap"] <= 0):
        return None
    e = float(am["close"].iloc[6])
    stop_px = e * (1 - stop_pct) if stop_pct else None
    return bracket_to_noon(am, 6, None, stop_px, entry_px=e) - COST


def pmh_breakout(day, rr=None, stop_mode="orlo", last_entry=690):
    """Buy the break of the premarket high (entries 9:35-11:30).
    stop = running morning low ('orlo') or fixed pct; target = rr x risk or noon."""
    if day["pm_hi"] is None or day["prev_close"] is None:
        return None
    am = day["am"]
    c = am["close"].to_numpy(); l = am["low"].to_numpy(); m = am["min"].to_numpy()
    run_lo = np.inf
    for i in range(len(am)):
        run_lo = min(run_lo, l[i])
        if m[i] > last_entry:
            return None
        if c[i] > day["pm_hi"] and i > 0:
            e = c[i]
            stop_px = run_lo if stop_mode == "orlo" else e * (1 - float(stop_mode))
            risk = e - stop_px
            if risk <= 0 or risk / e > 0.012:
                return None
            tgt_px = e + rr * risk if rr else None
            return bracket_to_noon(am, i, tgt_px, stop_px, entry_px=e) - COST
    return None


def dip_buy(day, dip, stop_pct, need_pm=0.0):
    """Strong premarket tape (pm_ret > need_pm), price dips below open in the first
    hour -> buy the dip, exit noon (stop under)."""
    if day["pm_ret"] is None or day["pm_ret"] <= need_pm:
        return None
    am = day["am"]
    c = am["close"].to_numpy(); m = am["min"].to_numpy()
    for i in range(1, len(am)):
        if m[i] > 630:                     # entries in the first hour only
            return None
        if (c[i] - day["open"]) / day["open"] <= -dip:
            e = c[i]
            return bracket_to_noon(am, i, None, e * (1 - stop_pct), entry_px=e) - COST
    return None


def vwap_rev(day, k, stop_pct):
    """Buy a k% discount to morning VWAP (10:00-11:30), exit at VWAP touch or noon."""
    am = day["am"]
    tp = (am["high"] + am["low"] + am["close"]) / 3
    vw = (tp * am["volume"]).cumsum() / am["volume"].cumsum()
    c = am["close"].to_numpy(); m = am["min"].to_numpy(); vwn = vw.to_numpy()
    h = am["high"].to_numpy(); l = am["low"].to_numpy()
    for i in range(6, len(am)):
        if not (600 <= m[i] <= 690):
            continue
        if (c[i] - vwn[i]) / vwn[i] <= -k:
            e = c[i]
            stop_px = e * (1 - stop_pct)
            for j in range(i + 1, len(am)):
                if l[j] <= stop_px:
                    return (stop_px - e) / e - COST
                if h[j] >= vwn[j]:
                    return (vwn[j] - e) / e - COST
            return (c[-1] - e) / e - COST
    return None


def base(day, start_min=OPEN):
    am = day["am"]
    m = am["min"].to_numpy()
    i0 = int(np.searchsorted(m, start_min + 5))
    if i0 >= len(am) - 1:
        return None
    e = float(am["close"].iloc[i0])
    return (float(am["close"].iloc[-1]) - e) / e - COST


def with_trend(fn, ma):
    def wrapped(day):
        if day.get(f"ma{ma}") is None or day["prev_close"] is None:
            return None
        if day["prev_close"] <= day[f"ma{ma}"]:
            return None
        return fn(day)
    return wrapped


def with_vol(fn, x):
    """Only when premarket volume is x times its 20-day average (conviction filter)."""
    def wrapped(day):
        if day["pm_vol_x"] is None or day["pm_vol_x"] < x:
            return None
        return fn(day)
    return wrapped


STRATS = [
    ("BASE open->noon",        lambda d: base(d)),
    ("BASE 10:00->noon",       lambda d: base(d, 600)),
    ("OD pm>0",                lambda d: open_drive(d, 0.0)),
    ("OD pm>0.15",             lambda d: open_drive(d, 0.0015)),
    ("OD pm>0.30",             lambda d: open_drive(d, 0.0030)),
    ("OD pm>0.15 s0.4",        lambda d: open_drive(d, 0.0015, stop_pct=0.004)),
    ("OD SHORT pm<-0.30",      lambda d: open_drive(d, 0.0030, short=True)),
    ("GAPU cont noon",         lambda d: gap_rule(d, 0.001, 0.010)),
    ("GAPU cont t0.4/s0.3",    lambda d: gap_rule(d, 0.001, 0.010, 0.004, 0.003)),
    ("GAPU cont s0.3",         lambda d: gap_rule(d, 0.001, 0.010, None, 0.003)),
    ("GAPD rev noon",          lambda d: gap_rule(d, -0.012, -0.003)),
    ("GAPD rev t0.4/s0.3",     lambda d: gap_rule(d, -0.012, -0.003, 0.004, 0.003)),
    ("GAPX SHORT big-gap-up",  lambda d: gap_rule(d, 0.010, 0.030, 0.004, 0.004,
                                                  short=True)),
    ("GAPX SHORT big-gap-dn",  lambda d: gap_rule(d, -0.030, -0.012, 0.006, 0.004,
                                                  short=True)),
    ("F30 mom>0",              lambda d: f30_mom(d, 0.0)),
    ("F30 mom>0.2",            lambda d: f30_mom(d, 0.002)),
    ("F30 mom>0 +gap",         lambda d: f30_mom(d, 0.0, need_gap=True)),
    ("F30 mom>0 s0.4",         lambda d: f30_mom(d, 0.0, stop_pct=0.004)),
    ("PMHB noon (orlo)",       lambda d: pmh_breakout(d)),
    ("PMHB R1.5 (orlo)",       lambda d: pmh_breakout(d, rr=1.5)),
    ("PMHB R2 (orlo)",         lambda d: pmh_breakout(d, rr=2.0)),
    ("PMHB R2 s0.3",           lambda d: pmh_breakout(d, rr=2.0, stop_mode=0.003)),
    ("PMHB R3 s0.3",           lambda d: pmh_breakout(d, rr=3.0, stop_mode=0.003)),
    ("DIP 0.2 s0.4 pm>0",      lambda d: dip_buy(d, 0.002, 0.004)),
    ("DIP 0.3 s0.4 pm>0",      lambda d: dip_buy(d, 0.003, 0.004)),
    ("DIP 0.3 s0.4 pm>.15",    lambda d: dip_buy(d, 0.003, 0.004, need_pm=0.0015)),
    ("VWAP k0.2 s0.4",         lambda d: vwap_rev(d, 0.002, 0.004)),
    ("VWAP k0.3 s0.4",         lambda d: vwap_rev(d, 0.003, 0.004)),
]
for ma in (50, 200):
    STRATS += [
        (f"OD pm>0.15 >MA{ma}",   with_trend(lambda d: open_drive(d, 0.0015), ma)),
        (f"GAPU noon >MA{ma}",    with_trend(lambda d: gap_rule(d, 0.001, 0.010), ma)),
        (f"PMHB R2 >MA{ma}",      with_trend(lambda d: pmh_breakout(d, rr=2.0), ma)),
        (f"F30>0 >MA{ma}",        with_trend(lambda d: f30_mom(d, 0.0), ma)),
        (f"DIP 0.3 >MA{ma}",      with_trend(lambda d: dip_buy(d, 0.003, 0.004), ma)),
    ]
STRATS += [
    ("OD pm>0.15 vol1.5x",     with_vol(lambda d: open_drive(d, 0.0015), 1.5)),
    ("PMHB R2 vol1.5x",        with_vol(lambda d: pmh_breakout(d, rr=2.0), 1.5)),
    ("GAPU noon vol1.5x",      with_vol(lambda d: gap_rule(d, 0.001, 0.010), 1.5)),
]


def window(dd, lo, hi):
    lo, hi = pd.Timestamp(lo).date(), pd.Timestamp(hi).date()
    return [d for d in dd if lo <= d["date"] < hi]


def run(dd, fn):
    return np.array([x for x in (fn(d) for d in dd) if x is not None])


def main():
    df = load()
    dd = days(df)
    print(f"MORNING-QQQ: {len(STRATS)} configs | 9:30->12:00 only, hard noon exit | "
          f"cost {COST*1e4:.0f}bps | {len(dd)} days "
          f"{dd[0]['date']}..{dd[-1]['date']}\n")
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
    print(f"\n=== GATE {GATE[0]}..{GATE[1]} ({len(arena_pass)} arena survivors) ===")
    gate_pass = []
    for name, fn in arena_pass:
        r = run(window(dd, *GATE), fn)
        tot = r.sum() * 100 if len(r) >= 10 else -99
        print(f"  {name:<24} n={len(r):>3} win={(r > 0).mean() if len(r) else 0:.0%} "
              f"tot={tot:+.2f}%{'  <-- GATE PASS' if tot > 0 else ''}")
        if tot > 0:
            gate_pass.append((name, fn))
    print(f"\n=== FINAL one-shot {FINAL[0]}..now ({len(gate_pass)} gate survivors) ===")
    for name, fn in gate_pass:
        r = run(window(dd, *FINAL), fn)
        if len(r) < 10:
            print(f"  {name:<24} too few trades ({len(r)})")
            continue
        t = r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else 0
        r0 = r + COST                          # remove headline cost
        print(f"  {name:<24} n={len(r):>3} win={(r > 0).mean():.0%} "
              f"avg={r.mean()*1e4:+.1f}bp tot={r.sum()*100:+.2f}% t={t:+.2f} | "
              f"0bps {r0.sum()*100:+.2f}% / 5bps {(r0 - 5e-4).sum()*100:+.2f}%")


if __name__ == "__main__":
    main()
