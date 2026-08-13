"""
vob_verdict.py — the decisive vOB evaluation on the FULL book backfill
(Feb-Aug 2026, ~25k bars). Grid over configs; the question that matters:
does any config validate positive, and do BOOK features beat the PRICE-ONLY
control? Time-ordered 60/20/20 OPT/VAL/TEST; TEST untouched here.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import vx_lab as L

GRID = []
for tf in (5, 15):
    for tgt in (0.002, 0.003):
        for stop in (0.0015, 0.002):
            for H in (12, 24):
                for q in (0.85, 0.90):
                    for feats in ("book", "bookprice", "price"):
                        GRID.append(dict(tf=tf, tgt=tgt, stop=stop, H=H, q=q,
                                         feats=feats))


def main():
    book = L.load_vob()
    print(f"vOB VERDICT — {len(book)} bars, {len(GRID)} configs\n")
    best = {}
    for i, cfg in enumerate(GRID):
        try:
            o = L.eval_vob(cfg, book, "opt")
            if not o or (o["sharpe"] or -9) <= 0:
                continue
            v = L.eval_vob(cfg, book, "val")
            if not v:
                continue
            vs = v["sharpe"] or -9
            f = cfg["feats"]
            if f not in best or vs > best[f][0]:
                best[f] = (vs, cfg, o, v)
            if vs > 0:
                print(f"  VAL-POSITIVE: {cfg} OPT {o['sharpe']:.2f} "
                      f"VAL {vs:.2f} (n={v['n']})", flush=True)
        except Exception as e:
            print(f"  [err {cfg}: {e}]", flush=True)
    print("\nbest per feature set (by VAL Sharpe):")
    for f, (vs, cfg, o, v) in sorted(best.items()):
        print(f"  {f:>10}: VAL {vs:+6.2f} (n={v['n']}, tot {v['total']:+.1f}%) "
              f"| OPT {o['sharpe']:.2f}  {cfg}")
    if not best:
        print("  none reached a positive OPT Sharpe at all")


if __name__ == "__main__":
    main()
