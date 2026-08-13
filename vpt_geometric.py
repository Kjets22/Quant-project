"""
vpt_geometric.py — GEOMETRIC chart-pattern detection (the user's spec: head &
shoulders, flags, triangles, double tops/bottoms) on any bar interval 1-60 min.

Method: ZigZag pivots (reversal threshold = k x ATR), then pattern grammars over
the pivot sequence. Every pattern defines: entry trigger (confirmation close),
side (bearish patterns SHORT), stop (pattern invalidation), target (measured
move or R-multiple). Detection is strictly causal: a pattern exists only once
its trigger bar closes; all pivots used must be confirmed (a pivot is confirmed
only after price has reversed k x ATR away, which is when the NEXT pivot leg is
underway — we track confirmation bar indexes and never use a pivot before it).

Exposed:
  detect(tf_df, params) -> list of dict(i_trigger, side, entry, stop, tgt, name)
  simulate(df, trades, H, cost) -> np.array of per-trade returns
  bars(tk, minutes) -> resampled OHLCV frame from the 5-min cache
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

COST = 5.0 / 1e4


def bars(tk, minutes):
    d = full_series(tk).set_index("timestamp").resample(f"{minutes}min").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna().reset_index()
    return d


def atr(h, l, c, n=20):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n).mean().to_numpy()


def zigzag(h, l, c, A, k=2.0):
    """Confirmed pivots: list of (bar_index, price, +1 high/-1 low, confirm_bar).
    A pivot is only usable from its confirm_bar onward (causality)."""
    piv = []
    n = len(c)
    direction = 0
    ext_i, ext_p = 0, c[0]
    for i in range(1, n):
        if not np.isfinite(A[i]):
            continue
        band = k * A[i]
        if direction >= 0:                    # tracking a high
            if h[i] > ext_p:
                ext_i, ext_p = i, h[i]
            if ext_p - l[i] > band:           # reversal down confirms the high
                piv.append((ext_i, ext_p, +1, i))
                direction, ext_i, ext_p = -1, i, l[i]
            elif direction == 0 and h[i] - ext_p > band:
                direction = 1
        if direction <= 0:                    # tracking a low
            if l[i] < ext_p or direction == 0:
                if direction != 0 and l[i] < ext_p:
                    ext_i, ext_p = i, l[i]
            if direction < 0 and h[i] - ext_p > band:   # reversal up confirms low
                piv.append((ext_i, ext_p, -1, i))
                direction, ext_i, ext_p = +1, i, h[i]
    return piv


def _near(a, b, tol):
    return abs(a - b) / max(abs(b), 1e-9) <= tol


def detect(df, k=2.0, tol=0.003, max_span=120, sides="both"):
    """Scan pivots for the pattern grammar; return confirmed trade setups."""
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    A = atr(h, l, c)
    piv = zigzag(h, l, c, A, k)
    out = []
    n = len(c)

    def emit(i_conf, side, entry_trigger, stop, tgt, name):
        """Find the first bar >= i_conf whose close crosses the trigger."""
        for j in range(i_conf, min(i_conf + 30, n)):
            if side == "long" and c[j] > entry_trigger:
                out.append(dict(i=j, side="long", stop=stop, tgt=tgt, name=name))
                return
            if side == "short" and c[j] < entry_trigger:
                out.append(dict(i=j, side="short", stop=stop, tgt=tgt, name=name))
                return

    for a in range(len(piv) - 4):
        p = piv[a:a + 5]
        idx = [x[0] for x in p]
        px = [x[1] for x in p]
        typ = [x[2] for x in p]
        conf = p[-1][3]
        if idx[-1] - idx[0] > max_span:
            continue
        # HEAD & SHOULDERS (bearish): H-L-H-L-H with middle head highest
        if typ == [1, -1, 1, -1, 1] and sides in ("both", "short"):
            ls, l1, hd, l2, rs = px
            if hd > ls and hd > rs and _near(ls, rs, tol * 2) \
                    and min(ls, rs) > max(l1, l2):
                neck = (l1 + l2) / 2
                emit(conf, "short", neck, max(rs, hd * 0.999),
                     neck - (hd - neck), "HS")
        # INVERSE H&S (bullish)
        if typ == [-1, 1, -1, 1, -1] and sides in ("both", "long"):
            ls, h1, hd, h2, rs = px
            if hd < ls and hd < rs and _near(ls, rs, tol * 2) \
                    and max(ls, rs) < min(h1, h2):
                neck = (h1 + h2) / 2
                emit(conf, "long", neck, min(rs, hd * 1.001),
                     neck + (neck - hd), "iHS")
    for a in range(len(piv) - 2):
        p = piv[a:a + 3]
        px = [x[1] for x in p]
        typ = [x[2] for x in p]
        conf = p[-1][3]
        # DOUBLE TOP: H-L-H, tops near-equal -> short below valley
        if typ == [1, -1, 1] and sides in ("both", "short") \
                and _near(px[0], px[2], tol):
            emit(conf, "short", px[1], max(px[0], px[2]),
                 px[1] - (px[0] - px[1]), "DT")
        # DOUBLE BOTTOM: L-H-L -> long above peak
        if typ == [-1, 1, -1] and sides in ("both", "long") \
                and _near(px[0], px[2], tol):
            emit(conf, "long", px[1], min(px[0], px[2]),
                 px[1] + (px[1] - px[0]), "DB")
    # FLAGS: pole = fast move; flag = counter-drift channel; break in pole dir
    look = 12
    for i in range(look * 2, n):
        if not np.isfinite(A[i]) or A[i] <= 0:
            continue
        pole = c[i - look] - c[i - 2 * look]
        drift = c[i] - c[i - look]
        if abs(pole) < 4 * A[i]:
            continue
        if pole > 0 and sides in ("both", "long") \
                and -2 * A[i] < drift < 0.5 * A[i]:
            hi_flag = h[i - look:i + 1].max()
            lo_flag = l[i - look:i + 1].min()
            if c[i] > hi_flag * 0.999 and (hi_flag - lo_flag) < 2.5 * A[i]:
                out.append(dict(i=i, side="long", stop=lo_flag,
                                tgt=c[i] + abs(pole), name="BullFlag"))
        if pole < 0 and sides in ("both", "short") \
                and -0.5 * A[i] < drift < 2 * A[i]:
            hi_flag = h[i - look:i + 1].max()
            lo_flag = l[i - look:i + 1].min()
            if c[i] < lo_flag * 1.001 and (hi_flag - lo_flag) < 2.5 * A[i]:
                out.append(dict(i=i, side="short", stop=hi_flag,
                                tgt=c[i] - abs(pole), name="BearFlag"))
    # TRIANGLES from pivot trendlines: flat highs + rising lows (asc) etc.
    for a in range(len(piv) - 3):
        p = piv[a:a + 4]
        px = [x[1] for x in p]
        typ = [x[2] for x in p]
        conf = p[-1][3]
        if typ == [1, -1, 1, -1]:
            hi1, lo1, hi2, lo2 = px
            if _near(hi1, hi2, tol) and lo2 > lo1 * (1 + tol / 2) \
                    and sides in ("both", "long"):        # ascending
                emit(conf, "long", max(hi1, hi2), lo2,
                     max(hi1, hi2) + (hi1 - lo1), "AscTri")
            if _near(lo1, lo2, tol) and hi2 < hi1 * (1 - tol / 2) \
                    and sides in ("both", "short"):       # descending
                emit(conf, "short", min(lo1, lo2), hi2,
                     min(lo1, lo2) - (hi1 - lo1), "DescTri")
    out.sort(key=lambda t: t["i"])
    return out


def simulate(df, setups, H=48, cost=COST):
    """Bracket-walk each setup; non-overlapping; returns per-trade net returns."""
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    ts = pd.to_datetime(df["timestamp"]).to_numpy()
    n = len(c)
    rets, times, names = [], [], []
    last_exit = -1
    for t in setups:
        i = t["i"]
        if i <= last_exit or i >= n - 1:
            continue
        e = c[i]
        stop, tgt = t["stop"], t["tgt"]
        if t["side"] == "long" and not (stop < e < tgt):
            continue
        if t["side"] == "short" and not (tgt < e < stop):
            continue
        res, j = None, i + 1
        while j < min(i + H + 1, n):
            if t["side"] == "long":
                if l[j] <= stop:
                    res = -abs(e - stop) / e; break
                if h[j] >= tgt:
                    res = abs(tgt - e) / e; break
            else:
                if h[j] >= stop:
                    res = -abs(stop - e) / e; break
                if l[j] <= tgt:
                    res = abs(e - tgt) / e; break
            j += 1
        ex = min(j, n - 1)
        if res is None:
            res = (c[ex] - e) / e * (1 if t["side"] == "long" else -1)
        rets.append(res - cost)
        times.append(ts[i])
        names.append(t["name"])
        last_exit = ex
    return np.array(rets), np.array(times), names
