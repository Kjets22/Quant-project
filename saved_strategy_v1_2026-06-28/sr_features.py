"""
sr_features.py — causal support/resistance features (from the imported S/R design).

The S/R README's idea, distilled to features our triple-barrier ML can use: where
is price relative to support/resistance, prior-day levels, round numbers, and the
volume center? All ATR-normalized and strictly causal (only bars <= t). The bet:
these tell the model whether a long entry sits near a bounce zone (target likely)
or under a rejection zone (stop likely) -- the directional signal brackets lacked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from triple_barrier_ml import atr


def sr_features(df: pd.DataFrame) -> pd.DataFrame:
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    ts = pd.to_datetime(df["timestamp"])
    A = atr(h, l, c)
    A = np.maximum(A, 1e-9)
    H, L, C = pd.Series(h), pd.Series(l), pd.Series(c)

    def res(n):   # distance up to the recent N-bar high (resistance above)
        return (H.rolling(n).max().shift(1).to_numpy() - c) / A

    def sup(n):   # distance down to the recent N-bar low (support below)
        return (c - L.rolling(n).min().shift(1).to_numpy()) / A

    # prior-day levels (causal): resample to day, shift one session, map back
    day = pd.DataFrame({"h": h, "l": l, "c": c}, index=ts)
    daily = day.resample("1D").agg(h=("h", "max"), l=("l", "min"), c=("c", "last")).dropna()
    pdh = daily["h"].shift(1).reindex(ts.dt.normalize().values, method="ffill")
    pdl = daily["l"].shift(1).reindex(ts.dt.normalize().values, method="ffill")
    pdc = daily["c"].shift(1).reindex(ts.dt.normalize().values, method="ffill")

    # trailing volume-weighted price (volume center / POC proxy)
    pv = pd.Series(c * v)
    vwap = (pv.rolling(120).sum() / pd.Series(v).rolling(120).sum()).to_numpy()

    # round-number proximity (psychological levels): nearest $5 multiple
    nearest5 = np.round(c / 5.0) * 5.0

    rng_hi = H.rolling(60).max().to_numpy()
    rng_lo = L.rolling(60).min().to_numpy()

    f = pd.DataFrame({
        "sr_res20": res(20), "sr_sup20": sup(20),
        "sr_res60": res(60), "sr_sup60": sup(60),
        "sr_pdh": (pdh.to_numpy() - c) / A,
        "sr_pdl": (c - pdl.to_numpy()) / A,
        "sr_pdc": (c - pdc.to_numpy()) / A,
        "sr_round5": (c - nearest5) / A,                  # signed dist to round level
        "sr_vwap": (c - vwap) / A,                        # +above / -below volume center
        "sr_rangepos": (c - rng_lo) / (rng_hi - rng_lo + 1e-9),
    })
    return f.replace([np.inf, -np.inf], 0.0)
