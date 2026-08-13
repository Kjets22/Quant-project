"""
vpt_patterns.py — the PATTERN trader (vPT): classic technical patterns, coded
honestly, pushed through the strict multi-window arena that killed quant_rth.

Eight pattern families, each long-only (the stable's mandate), all on the same
5-min extended-hours data the validated strategies use, hourly decision bars:

  SRB   support/resistance breakout: close above the N-day high with volume
  DBL   double bottom: two lows within tol%, entry on neckline break
  MAX   moving-average cross: fast above slow, fresh cross only
  RSI2  Connors RSI(2) dip-buy in an uptrend (above 200-bar MA)
  BBS   Bollinger squeeze expansion: bandwidth percentile low -> upside break
  ENG   bullish engulfing at a local low
  VWR   fade to rolling VWAP in an uptrend
  GAPC  overnight gap-continuation (the one pattern family with a known
        extended-hours edge in this project)

Arena = worst of FIVE half-years 2022-01..2024-07 (rules need no training);
GATE 2024-07..2025-07; FINAL 2025-07..now, one-shot, gate-positive only.
Costs 5 bps. Tickers: the 8-name basket. Bracket exits: 2x risk target,
structure stop, 48-bar clock.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from alpaca_bot2 import full_series
from triple_barrier_breadth import TICKERS

COST = 5.0 / 1e4
SUBS = [("2022-01-14", "2022-07-14"), ("2022-07-14", "2023-01-14"),
        ("2023-01-14", "2023-07-14"), ("2023-07-14", "2024-01-14"),
        ("2024-01-14", "2024-07-14")]
GATE = ("2024-07-14", "2025-07-14")
FINAL = ("2025-07-14", "2099-01-01")
H = 48                                  # hourly bars -> 2 days max


def hourly(tk):
    d = full_series(tk).set_index("timestamp").resample("60min").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna().reset_index()
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    o, h, l, c, v = (d[x].to_numpy(float) for x in
                     ("open", "high", "low", "close", "volume"))
    return ts, o, h, l, c, v


def _sma(x, n):
    return pd.Series(x).rolling(n).mean().to_numpy()


def _rsi(c, n=2):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0)).rolling(n).mean().to_numpy()
    dn = pd.Series(np.where(d < 0, -d, 0)).rolling(n).mean().to_numpy()
    rs = up / np.maximum(dn, 1e-9)
    return 100 - 100 / (1 + rs)


def signals(name, ts, o, h, l, c, v):
    """Boolean entry array + stop distance array (structure-based)."""
    n = len(c)
    sig = np.zeros(n, bool)
    ma50, ma200 = _sma(c, 50), _sma(c, 200)
    hi20 = pd.Series(h).rolling(320).max().shift(1).to_numpy()   # ~20 days of hours
    lo10 = pd.Series(l).rolling(160).min().shift(1).to_numpy()
    vol20 = _sma(v, 20)
    if name == "SRB":
        sig = (c > hi20) & (v > 1.5 * vol20) & (c > ma200)
    elif name == "DBL":
        lo40 = pd.Series(l).rolling(40).min().to_numpy()
        prior = pd.Series(lo40).shift(40).to_numpy()
        neck = pd.Series(h).rolling(40).max().shift(1).to_numpy()
        near = np.abs(lo40 - prior) / np.maximum(prior, 1e-9) < 0.004
        sig = near & (c > neck)
    elif name == "MAX":
        cross = (ma50 > ma200) & (np.roll(ma50, 1) <= np.roll(ma200, 1))
        sig = cross
    elif name == "RSI2":
        r = _rsi(c, 2)
        sig = (r < 10) & (c > ma200)
    elif name == "BBS":
        m = _sma(c, 20)
        sd = pd.Series(c).rolling(20).std().to_numpy()
        bw = 4 * sd / np.maximum(m, 1e-9)
        bw_lo = pd.Series(bw).rolling(320).quantile(0.15).to_numpy()
        sig = (np.roll(bw, 1) < np.roll(bw_lo, 1)) & (c > m + 2 * sd * 0.5) \
            & (c > ma200)
    elif name == "ENG":
        body_dn = np.roll(c, 1) < np.roll(o, 1)
        engulf = (c > o) & (o < np.roll(c, 1)) & (c > np.roll(o, 1))
        at_low = l <= pd.Series(l).rolling(30).min().shift(1).to_numpy() * 1.002
        sig = body_dn & engulf & at_low
    elif name == "VWR":
        tp = (h + l + c) / 3
        vwap = (pd.Series(tp * v).rolling(80).sum()
                / pd.Series(v).rolling(80).sum()).to_numpy()
        sig = (c < vwap * 0.997) & (c > ma200) & (ma50 > ma200)
    elif name == "GAPC":
        et = pd.DatetimeIndex(ts).tz_localize("UTC").tz_convert("America/New_York")
        mins = np.asarray(et.hour * 60 + et.minute)
        dates = np.asarray(et.date)
        newd = np.concatenate([[True], dates[1:] != dates[:-1]])
        prev_c = np.where(newd, np.roll(c, 1), np.nan)
        gap = (o - prev_c) / np.maximum(prev_c, 1e-9)
        sig = newd & (gap > 0.002) & (gap < 0.012)
    stop_d = np.maximum(c - lo10, c * 0.004)             # structure stop, floor 0.4%
    return sig, stop_d


def regime_mask(ts, c):
    """Daily-close > daily 200-MA, known as of the PRIOR day (causal). The
    canonical long-only regime filter — iteration 2's single added layer."""
    idx = pd.DatetimeIndex(ts).tz_localize("UTC").tz_convert("America/New_York")
    days = pd.Series(c, index=idx).groupby(idx.date).last()
    ma = days.rolling(200).mean()
    ok_by_day = (days > ma).astype(bool).shift(1, fill_value=False)
    return np.array([bool(ok_by_day.get(d, False)) for d in idx.date])


def run(name, tk, lo, hi, regime=False):
    ts, o, h, l, c, v = hourly(tk)
    sig, stop_d = signals(name, ts, o, h, l, c, v)
    if regime:
        sig = sig & regime_mask(ts, c)
    idx = np.where(sig & (ts >= np.datetime64(lo)) & (ts < np.datetime64(hi))
                   & np.isfinite(stop_d))[0]
    rets, n = [], len(c)
    last_exit = -1
    for i in idx:
        if i <= last_exit or i >= n - 1:
            continue
        e = c[i]
        stop = e - stop_d[i]
        tgt = e + 2 * stop_d[i]
        res, j = None, i + 1
        while j < min(i + H + 1, n):
            if l[j] <= stop:
                res = 0; break
            if h[j] >= tgt:
                res = 1; break
            j += 1
        ex = min(j, n - 1)
        px = tgt if res == 1 else (stop if res == 0 else c[ex])
        rets.append((px - e) / e - COST)
        last_exit = ex
    return np.array(rets)


PATTERNS = ["SRB", "DBL", "MAX", "RSI2", "BBS", "ENG", "VWR", "GAPC"]


def main():
    regime = "--regime" in sys.argv
    mode = "REGIME-FILTERED (daily close > daily MA200)" if regime else "raw"
    print(f"vPT — pattern trader arena, {mode} (8 tickers, 5bps)")
    print("scoring: worst ACTIVE half (>=8 trades); need >=3 active halves\n")
    print(f"{'pattern':>8} {'worst-act':>10} {'active':>7} {'subs':>44}  verdict")
    surv = []
    for name in PATTERNS:
        subs = []
        for lo, hi in SUBS:
            r = np.concatenate([run(name, tk, lo, hi, regime) for tk in TICKERS])
            subs.append(float(r.sum() * 100) if len(r) >= 8 else None)
        act = [s for s in subs if s is not None]
        worst = min(act) if act else -99.0
        ok = len(act) >= 3 and worst > 0
        tag = " <-- arena PASS" if ok else ""
        disp = [("--" if s is None else round(s, 1)) for s in subs]
        print(f"{name:>8} {worst:>+9.2f}% {len(act):>7} {str(disp):>44}{tag}")
        if ok:
            surv.append(name)
    print(f"\n=== GATE {GATE[0]}..{GATE[1]} ({len(surv)} arena survivors) ===")
    gsurv = []
    for name in surv:
        r = np.concatenate([run(name, tk, *GATE, regime) for tk in TICKERS])
        tot = r.sum() * 100 if len(r) >= 10 else -99
        print(f"  {name}: n={len(r)} win={(r > 0).mean() if len(r) else 0:.0%} "
              f"tot={tot:+.2f}%")
        if tot > 0:
            gsurv.append(name)
    print(f"\n=== FINAL one-shot ({len(gsurv)} gate survivors) ===")
    for name in gsurv:
        r = np.concatenate([run(name, tk, *FINAL, regime) for tk in TICKERS])
        if len(r) < 10:
            print(f"  {name}: too few trades")
            continue
        t = r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else 0
        print(f"  {name}: n={len(r)} win={(r > 0).mean():.0%} "
              f"avg={r.mean() * 1e4:+.1f}bp tot={r.sum() * 100:+.2f}% t={t:+.2f}")


if __name__ == "__main__":
    main()
