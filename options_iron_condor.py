"""
options_iron_condor.py — the last clean options test: DEFINED-RISK short vol.

Each entry: sell ~16-delta call + put, BUY ~6-delta wings (bounded max loss, so a
gap can't produce the unbounded short-straddle tail). Priced from the REAL SPY
chain (per-strike IV), 4-leg spread cost charged, held to expiry. Then:
  * month-by-month P&L (calm vs spike regimes)
  * walk-forward blocks + DEFLATED SHARPE (multiple-testing correction)
If even bounded-risk short vol can't clear a deflated Sharpe after 4-leg costs,
options are the same honest "no validated edge" as everything else.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from options_ml_pipeline import clean_quotes
from trials import deflated_sharpe_ratio

SHORT_DELTA = 0.16
LONG_DELTA = 0.06
DTE_LO, DTE_HI, DTE_TGT = 20, 45, 30
SPREAD_LEG = 0.03      # 3% of each leg's premium (4 legs)
ENTRY_EVERY = 5        # enter weekly (limit overlap)
TRIALS = 12


def pick(group, ctype, td):
    side = group[group["type"] == ctype]
    if len(side) < 1:
        return None
    side = side.assign(dd=(side["delta"].abs() - td).abs())
    r = side.loc[side["dd"].idxmin()]
    return r if r["volume"] >= 1 else None


def main():
    pq = next(Path("data_cache/options").glob("chain_*2026-06-20.parquet"))
    print(f"[load] {pq.name}")
    ch = pd.read_parquet(pq)
    ch = clean_quotes(ch)
    ch["mid"] = (ch["bid"] + ch["ask"]) / 2
    spot = ch.groupby("date")["underlying_price"].first()
    dates = sorted(ch["date"].unique())

    trades = []
    for di in range(0, len(dates), ENTRY_EVERY):
        d = dates[di]
        day = ch[ch["date"] == d]
        day = day[(day["dte"] >= DTE_LO) & (day["dte"] <= DTE_HI)]
        if day.empty:
            continue
        exp = day.iloc[(day["dte"] - DTE_TGT).abs().argsort()].iloc[0]["expiry"]
        g = day[day["expiry"] == exp]
        sc = pick(g, "C", SHORT_DELTA); lp = pick(g, "P", LONG_DELTA)
        sp = pick(g, "P", SHORT_DELTA); lc = pick(g, "C", LONG_DELTA)
        if any(x is None for x in (sc, sp, lc, lp)):
            continue
        credit = sp["mid"] + sc["mid"] - lp["mid"] - lc["mid"]
        if credit <= 0:
            continue
        legs = sp["mid"] + sc["mid"] + lp["mid"] + lc["mid"]
        cost = legs * SPREAD_LEG
        # settle at expiry spot
        future = spot[spot.index >= exp]
        if future.empty:
            continue
        ST = future.iloc[0]
        payoff = (max(0, sp["strike"] - ST) - max(0, lp["strike"] - ST)
                  + max(0, ST - sc["strike"]) - max(0, ST - lc["strike"]))
        pnl = credit - payoff - cost
        max_loss = max(sp["strike"] - lp["strike"], lc["strike"] - sc["strike"]) - credit
        trades.append({"date": pd.Timestamp(d), "pnl": pnl, "credit": credit,
                       "max_loss": max_loss})
    t = pd.DataFrame(trades).set_index("date").sort_index()
    print(f"\niron condors: {len(t)} trades  avg credit ${t.credit.mean():.2f}  "
          f"avg max-loss ${t.max_loss.mean():.2f} (BOUNDED tail)")

    print("\n  month-by-month P&L:")
    for m, s in t.groupby(t.index.to_period("M")):
        print(f"    {str(m)}: n={len(s):2d}  P&L ${s.pnl.sum():7.2f}  win%={(s.pnl>0).mean():.0%}")

    pnl = t["pnl"].to_numpy()
    eq = np.cumsum(pnl)
    dd = (eq - np.maximum.accumulate(eq)).min()
    per_year = ENTRY_EVERY and 252 / ENTRY_EVERY
    sh = pnl.mean() / pnl.std() * np.sqrt(per_year) if pnl.std() > 1e-9 else 0
    print(f"\n  OVERALL: total ${pnl.sum():.0f}  mean ${pnl.mean():.2f}  win% {(pnl>0).mean():.0%}  "
          f"Sharpe(ann)~ {sh:.2f}  worst ${pnl.min():.2f}  maxDD ${dd:.0f}")

    # walk-forward + deflated Sharpe
    K = 5
    bnds = np.linspace(0, len(pnl), K + 1).astype(int)
    blk = [pnl[bnds[i]:bnds[i+1]] for i in range(K) if bnds[i+1] > bnds[i]]
    bsh = [b.mean() / b.std() * np.sqrt(per_year) for b in blk if b.std() > 1e-9]
    sr_obs = pnl.mean() / pnl.std() if pnl.std() > 1e-9 else 0
    dsr = deflated_sharpe_ratio(sr_obs, len(pnl), TRIALS,
                                np.var(bsh) / per_year if len(bsh) > 1 else 0.01,
                                skew=float(pd.Series(pnl).skew()),
                                kurt=float(pd.Series(pnl).kurt()) + 3)
    print(f"  walk-forward block Sharpes: {[round(x,2) for x in bsh]}")
    print(f"  blocks positive: {sum(x>0 for x in bsh)}/{len(bsh)}   "
          f"DEFLATED SHARPE (P>0 after {TRIALS} trials) = {dsr:.3f}")
    print("  (DSR > 0.95 => real after multiple-testing; < 0.95 => not validated.)")


if __name__ == "__main__":
    main()
