"""
quant_rth.py — classic QUANT intraday strategies on QQQ, pushed through the ladder.

The complement to Evolution VI: instead of evolving ML genomes, test the documented
intraday anomalies (hypothesis-first => much lower overfitting risk). All long-only,
RTH-only, flat by the close, <= 4h holds — the user's day-trading rules.

Families:
  MOM   market intraday momentum (Gao-Han-Li-Zhou): overnight gap and/or first-30min
        return positive -> buy 15:00/15:30, exit at close
  ORB   opening-range breakout: buy confirmed break of the first 15/30-min high,
        stop = range low, target = R x risk, flat by close
  GAPU  gap-up continuation / GAPD gap-down reversal: enter on the first bar,
        bracket target/stop, flat by close
  VWAP  reversion: buy a k% discount to intraday VWAP (10:00-14:30), exit at VWAP
        touch or close, fixed stop
  BASE  15:30 -> close every day (seasonality control)

Ladder (stricter than the ML arena — rules need no training data):
  ARENA = worst of FIVE half-years 2022-01..2024-07 (all must be positive-ish)
  GATE  = 2024-07..2025-07 (arena-positive only)
  FINAL = 2025-07..now, one-shot (gate-positive only)
Costs: 5 bps round trip. Research only.
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

COST = 5.0 / 1e4
SUBS = [("2022-01-14", "2022-07-14"), ("2022-07-14", "2023-01-14"),
        ("2023-01-14", "2023-07-14"), ("2023-07-14", "2024-01-14"),
        ("2024-01-14", "2024-07-14")]
GATE = ("2024-07-14", "2025-07-14")
FINAL = ("2025-07-14", "2099-01-01")
_ET = ZoneInfo("America/New_York")


def load():
    base = pd.read_csv("data_cache/QQQ_5minute_2021-06-01_2026-06-01.csv",
                       parse_dates=["timestamp"])
    parts = [base]
    for p in sorted(Path("data_cache").glob("QQQ_recent_2026-06-01_*.csv")):
        parts.append(pd.read_csv(p, parse_dates=["timestamp"]))
    df = (pd.concat(parts, ignore_index=True)
          .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
          .reset_index(drop=True))
    ts = pd.DatetimeIndex(df["timestamp"]).tz_localize("UTC").tz_convert(_ET)
    df["date"] = ts.date
    df["min"] = ts.hour * 60 + ts.minute
    df["rth"] = (df["min"] >= 570) & (df["min"] < 960)
    return df


def days(df):
    """Per-day dict: RTH sub-frame + prev close + gap + premarket ret."""
    out, prev_close = [], None
    for d, g in df.groupby("date", sort=True):
        r = g[g["rth"]]
        if len(r) < 30:
            if len(r):
                prev_close = float(r["close"].iloc[-1])
            continue
        pm = g[g["min"] < 570]
        o = float(r["close"].iloc[0])
        rec = {"date": d, "r": r.reset_index(drop=True), "open": o,
               "prev_close": prev_close,
               "gap": (o - prev_close) / prev_close if prev_close else None,
               "pm_ret": ((float(pm["close"].iloc[-1]) - prev_close) / prev_close
                          if prev_close and len(pm) else 0.0)}
        prev_close = float(r["close"].iloc[-1])
        out.append(rec)
    closes = pd.Series([d["r"]["close"].iloc[-1] for d in out], dtype=float)
    for ma in (50, 200):
        m = closes.rolling(ma).mean().shift(1)     # known before today's open
        for i, d in enumerate(out):
            d[f"ma{ma}"] = float(m.iloc[i]) if np.isfinite(m.iloc[i]) else None
    return out


def bracket_exit(r, i0, tgt_px, stop_px):
    """Walk bars i0+1.. within the day; return exit price (target/stop/close)."""
    h, l, c = r["high"].to_numpy(), r["low"].to_numpy(), r["close"].to_numpy()
    for j in range(i0 + 1, len(r)):
        if l[j] <= stop_px:
            return stop_px
        if h[j] >= tgt_px:
            return tgt_px
    return c[-1]


# ---------------- strategy families (each returns per-day return or None) --------
def mom(day, use_gap, use_f30, entry_min):
    if day["gap"] is None:
        return None
    r = day["r"]
    f30 = (float(r["close"].iloc[5]) - day["open"]) / day["open"]
    sig_parts = ([day["gap"]] if use_gap else []) + ([f30] if use_f30 else [])
    if not sig_parts or min(sig_parts) <= 0:
        return None
    m = r["min"].to_numpy()
    i0 = int(np.searchsorted(m, entry_min))
    if i0 >= len(r) - 1:
        return None
    e = float(r["close"].iloc[i0])
    return (float(r["close"].iloc[-1]) - e) / e - COST


def orb(day, or_bars, rr):
    r = day["r"]
    if len(r) < or_bars + 3:
        return None
    hi = float(r["high"].iloc[:or_bars].max())
    lo = float(r["low"].iloc[:or_bars].min())
    m = r["min"].to_numpy()
    c = r["close"].to_numpy()
    for i in range(or_bars, len(r)):
        if m[i] >= 750:                       # entries before 12:30 only
            return None
        if c[i] > hi:
            e = c[i]
            risk = e - lo
            if risk <= 0 or risk / e > 0.012:
                return None
            x = bracket_exit(r, i, e + rr * risk, lo)
            return (x - e) / e - COST
    return None


def gap_rule(day, lo_g, hi_g, tgt, stp):
    if day["gap"] is None or not (lo_g <= day["gap"] <= hi_g):
        return None
    r = day["r"]
    e = float(r["close"].iloc[0])
    x = bracket_exit(r, 0, e * (1 + tgt), e * (1 - stp))
    return (x - e) / e - COST


def vwap_rev(day, k, stp):
    r = day["r"]
    tp = (r["high"] + r["low"] + r["close"]) / 3
    vw = (tp * r["volume"]).cumsum() / r["volume"].cumsum()
    c = r["close"].to_numpy()
    m = r["min"].to_numpy()
    vwn = vw.to_numpy()
    for i in range(6, len(r)):
        if not (600 <= m[i] <= 870):
            continue
        if (c[i] - vwn[i]) / vwn[i] <= -k:
            e = c[i]
            stop_px = e * (1 - stp)
            h, l = r["high"].to_numpy(), r["low"].to_numpy()
            for j in range(i + 1, len(r)):
                if l[j] <= stop_px:
                    return (stop_px - e) / e - COST
                if h[j] >= vwn[j]:
                    return (vwn[j] - e) / e - COST
            return (c[-1] - e) / e - COST
    return None


def base(day, entry_min):
    r = day["r"]
    m = r["min"].to_numpy()
    i0 = int(np.searchsorted(m, entry_min))
    if i0 >= len(r) - 1:
        return None
    e = float(r["close"].iloc[i0])
    return (float(r["close"].iloc[-1]) - e) / e - COST


STRATS = []
for ug, uf in ((1, 0), (0, 1), (1, 1)):
    for em in (900, 930):
        STRATS.append((f"MOM g{ug}f{uf}@{em//60}:{em%60:02d}",
                       lambda d, ug=ug, uf=uf, em=em: mom(d, ug, uf, em)))
for ob in (3, 6):
    for rr in (1.5, 2.0, 3.0):
        STRATS.append((f"ORB {ob*5}min R{rr}", lambda d, ob=ob, rr=rr: orb(d, ob, rr)))
for tgt in (0.004, 0.006):
    STRATS.append((f"GAPU cont t{tgt*100:.1f}",
                   lambda d, tgt=tgt: gap_rule(d, 0.001, 0.010, tgt, 0.003)))
    STRATS.append((f"GAPD rev t{tgt*100:.1f}",
                   lambda d, tgt=tgt: gap_rule(d, -0.012, -0.003, tgt, 0.003)))
for k in (0.003, 0.005):
    STRATS.append((f"VWAP k{k*100:.1f}", lambda d, k=k: vwap_rev(d, k, 0.004)))
STRATS.append(("BASE 15:30->close", lambda d: base(d, 930)))


def with_trend(fn, ma):
    """Regime filter (literature-motivated, one layer only): trade only when the
    prior day's close is above its own trailing MA — breakouts need a trend."""
    def wrapped(day):
        if day.get(f"ma{ma}") is None or day["prev_close"] is None:
            return None
        if day["prev_close"] <= day[f"ma{ma}"]:
            return None
        return fn(day)
    return wrapped


for ma in (50, 200):
    STRATS.append((f"ORB 30min R2 >MA{ma}",
                   with_trend(lambda d: orb(d, 6, 2.0), ma)))
    STRATS.append((f"ORB 30min R3 >MA{ma}",
                   with_trend(lambda d: orb(d, 6, 3.0), ma)))
    STRATS.append((f"GAPU t0.6 >MA{ma}",
                   with_trend(lambda d: gap_rule(d, 0.001, 0.010, 0.006, 0.003), ma)))
    STRATS.append((f"MOM g1f1@15:00 >MA{ma}",
                   with_trend(lambda d: mom(d, 1, 1, 900), ma)))


def window(dd, lo, hi):
    lo, hi = pd.Timestamp(lo).date(), pd.Timestamp(hi).date()
    return [d for d in dd if lo <= d["date"] < hi]


def run(dd, fn):
    rets = [x for x in (fn(d) for d in dd) if x is not None]
    return np.array(rets)


def main():
    df = load()
    dd = days(df)
    print(f"QUANT-RTH: {len(STRATS)} rule strategies | ARENA = worst of 5 half-years "
          f"2022-01..2024-07 | costs {COST*1e4:.0f}bps\n")
    print(f"  {'strategy':<22} {'worst':>7} {'subs (5 half-years)':>38}")
    arena_pass = []
    for name, fn in STRATS:
        subs = []
        for lo, hi in SUBS:
            r = run(window(dd, lo, hi), fn)
            subs.append(float(r.sum() * 100) if len(r) >= 8 else -99.0)
        worst = min(subs)
        flag = " <-- arena PASS" if worst > 0 else ""
        print(f"  {name:<22} {worst:>+6.2f}% {str([round(s,1) for s in subs]):>38}{flag}")
        if worst > 0:
            arena_pass.append((name, fn))
    print(f"\n=== GATE {GATE[0]}..{GATE[1]} ({len(arena_pass)} arena survivors) ===")
    gate_pass = []
    for name, fn in arena_pass:
        r = run(window(dd, *GATE), fn)
        tot = r.sum() * 100 if len(r) >= 10 else -99
        print(f"  {name:<22} n={len(r):>3} win={(r > 0).mean() if len(r) else 0:.0%} "
              f"tot={tot:+.2f}%")
        if tot > 0:
            gate_pass.append((name, fn))
    print(f"\n=== FINAL one-shot {FINAL[0]}..now ({len(gate_pass)} gate survivors) ===")
    for name, fn in gate_pass:
        r = run(window(dd, *FINAL), fn)
        if len(r) < 10:
            print(f"  {name:<22} too few trades"); continue
        t = r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else 0
        print(f"  {name:<22} n={len(r):>3} win={(r > 0).mean():.0%} "
              f"avg={r.mean()*1e4:+.1f}bp tot={r.sum()*100:+.2f}% t={t:+.2f}")


if __name__ == "__main__":
    main()
