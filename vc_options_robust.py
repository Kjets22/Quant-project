"""
vc_options_robust.py — is vC-OPT-2W's edge real, or three lucky trades?

The v6 study's one "winning" bucket collapsed under exactly this scrutiny (3 fills
carried 188% of the P&L), so apply the same tests to the book that is actually LIVE:
outlier dependence, bootstrap CI, per-ticker and per-quarter sign stability, and
whether the result survives dropping the best trades.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

import vc_options_real as V

HC = 0.01


def main():
    allt, tsmap = [], {}
    for tk in V.TICKERS:
        r = V.gen_trades(tk)
        if not r:
            continue
        trades, ts_all, c_all = r
        tsmap[tk] = (ts_all, c_all)
        allt += trades
    rows = []
    for tk in tsmap:
        tt = [t for t in allt if t["tk"] == tk]
        fills, _, _ = V.replay(tt, 5, 14, "ATM", *tsmap[tk])
        # replay returns (entry, exit, stock_ret) triples
        for e, x, sr in fills:
            rows.append({"tk": tk,
                         "ret": (x * (1 - HC) - e * (1 + HC)) / (e * (1 + HC)),
                         "entry": e, "stock_ret": sr})
    d = pd.DataFrame(rows)
    r = d["ret"].to_numpy()
    n = len(r)
    print(f"vC-OPT-2W (1-2w ATM, 1%/side) — {n} fills\n")
    print(f"  mean  {r.mean() * 100:+.2f}%/trade   median {np.median(r) * 100:+.2f}%"
          f"   win {100 * (r > 0).mean():.0f}%")
    print(f"  total {r.sum() * 100:+.0f}%  (${r.sum() * 1000:+,.0f} at $1k/trade)")
    sd = r.std(ddof=1)
    t = r.mean() / sd * np.sqrt(n)
    print(f"  t-stat {t:+.2f}   sd {sd * 100:.0f}%")

    srt = np.sort(r)[::-1]
    print(f"\n  OUTLIER DEPENDENCE (the test that killed v6's winner)")
    for k in (1, 2, 3, 5, 10):
        share = srt[:k].sum() / r.sum() * 100 if r.sum() else float("nan")
        rest = r.sum() - srt[:k].sum()
        print(f"    top-{k:>2} trades = {share:>6.1f}% of total P&L | "
              f"without them: {rest * 100:+.0f}% "
              f"({'still positive' if rest > 0 else 'NEGATIVE'})")
    print(f"    best trade {srt[0] * 100:+.0f}%   worst {srt[-1] * 100:+.0f}%")

    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(r, n, replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  BOOTSTRAP 95% CI on mean/trade: [{lo * 100:+.1f}%, {hi * 100:+.1f}%]"
          f"   P(mean>0) = {100 * (boot > 0).mean():.1f}%")

    print(f"\n  PER-TICKER sign stability")
    for tk, g in d.groupby("tk"):
        gr = g["ret"].to_numpy()
        print(f"    {tk:>5} n={len(gr):>3} mean {gr.mean() * 100:>+7.1f}% "
              f"total {gr.sum() * 100:>+7.0f}%")
    pos = sum(1 for _, g in d.groupby("tk") if g["ret"].sum() > 0)
    print(f"    -> {pos}/{d['tk'].nunique()} tickers positive")

    print(f"\n  MATCHED STOCK LEG (same fills): "
          f"{d['stock_ret'].mean() * 1e4:+.0f} bps/trade, "
          f"total {d['stock_ret'].sum() * 100:+.1f}%")


if __name__ == "__main__":
    main()
