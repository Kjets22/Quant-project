#!/usr/bin/env python3
"""End-to-end demo of the support/resistance system on synthetic data.

Run from the project root:
    python scripts/run_pipeline.py --days 90
    python scripts/run_pipeline.py --days 120 --intraday --intraday-days 8

Stages: generate universe -> build timeframes -> assemble premarket dataset ->
walk-forward split -> train LightGBM heatmap -> evaluate vs classic baselines ->
plots + sample premarket map. With --intraday it repeats for the intraday task
and demonstrates the live streaming server.

On real data, swap `make_synthetic_universe` for the Polygon fetch (see
scripts/fetch_polygon.py) -- everything downstream is identical because both
paths emit the same bar schema.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CONFIG
from src.data.synthetic import make_synthetic_universe
from src.data.timeframes import build_timeframes
from src.dataset.assembler import build_premarket_dataset, build_intraday_dataset
from src.models.lgbm_heatmap import HeatmapLGBM
from src.eval.harness import evaluate, summarise
from src.eval.plots import plot_example, plot_metric_summary
from src.serving.premarket import premarket_map, IntradayServer

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs")


def banner(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70, flush=True)


def train_and_eval(ds, meta, tag, cfg):
    masks = ds.split_masks(cfg)
    tr, va, te = masks["train"], masks["val"], masks["test"]
    print(f"[{tag}] rows: train={tr.sum()} val={va.sum()} test={te.sum()} "
          f"| features={len(ds.feature_cols)}")
    model = HeatmapLGBM(mc=cfg.model, feature_cols=ds.feature_cols)
    t0 = time.time()
    model.fit(ds.X.loc[tr], ds.y_support[tr], ds.y_resistance[tr],
              ds.X.loc[va], ds.y_support[va], ds.y_resistance[va])
    print(f"[{tag}] trained in {time.time() - t0:.1f}s "
          f"(best iters: {model.best_iters})")

    res = evaluate(model, ds, meta, te, cfg)
    print("\n" + summarise(res["rows"], res["perbin"]))

    # persist metrics
    pd.DataFrame(res["rows"]).to_csv(os.path.join(OUT, f"metrics_{tag}.csv"),
                                     index=False)
    with open(os.path.join(OUT, f"perbin_{tag}.json"), "w") as f:
        json.dump(res["perbin"], f, indent=2)

    # metric-summary bar charts (recall & precision at the middle K)
    kmid = cfg.eval.top_k_zones[len(cfg.eval.top_k_zones) // 2]
    for channel in ("support", "resistance"):
        plot_metric_summary(res["rows"], "f1", kmid, channel,
                            os.path.join(OUT, f"summary_f1_{channel}_{tag}.png"),
                            f"{tag}: {channel} F1@{kmid} -- model vs baselines")

    # a few qualitative example plots from the test split
    _plot_examples(res, cfg, tag, n=3)
    return model, res


def _plot_examples(res, cfg, tag, n=3):
    pred_sup, pred_res = res["_pred"]
    y_sup, y_res = res["_true"]
    meta = res["_meta"]
    ex_ids = list(res["_true_levels"].keys())
    # pick examples with the most true levels (most interesting)
    ex_ids.sort(key=lambda e: -len(res["_true_levels"][e]))
    for rank, ex in enumerate(ex_ids[:n]):
        m = (meta["example_id"] == ex).to_numpy()
        bp = meta.loc[m, "bin_price"].to_numpy()
        ref = float(meta.loc[m, "ref"].iloc[0])
        atr = float(meta.loc[m, "atr"].iloc[0])
        tk = str(meta.loc[m, "ticker"].iloc[0])
        bl = {name: lv.get(ex, np.array([]))
              for name, lv in res["_baselines"].items()
              if name in ("pivots", "prev_day_hl", "volume_poc")}
        plot_example(pred_sup[m], pred_res[m], y_sup[m], y_res[m], bp, ref, atr,
                     cfg, f"{tag} example: {tk}  ref={ref:.2f} ATR={atr:.3f}",
                     os.path.join(OUT, f"example_{tag}_{rank}_{tk}.png"),
                     baseline_levels=bl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--intraday", action="store_true")
    ap.add_argument("--intraday-days", type=int, default=6,
                    help="max sessions per ticker for the intraday dataset")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = CONFIG

    banner(f"1) Generating synthetic universe ({args.days} days)")
    t0 = time.time()
    uni_1m = make_synthetic_universe(n_days=args.days, seed=args.seed)
    tfs = {tk: build_timeframes(df) for tk, df in uni_1m.items()}
    for tk, d in tfs.items():
        print(f"  {tk}: 1m={len(d['1m'])} 5m={len(d['5m'])} "
              f"1d={len(d['1d'])}")
    print(f"  generated in {time.time() - t0:.1f}s")

    banner("2) PREMARKET task")
    ds = build_premarket_dataset(tfs, cfg, verbose=False)
    model_pm, res_pm = train_and_eval(ds, ds.meta, "premarket", cfg)

    banner("3) Sample premarket maps (next-session S/R)")
    for tk in list(tfs.keys())[:3]:
        m = premarket_map(tfs[tk], model_pm, cfg, ticker=tk, top_k=5)
        print(m.pretty() + "\n")

    if args.intraday:
        banner("4) INTRADAY task")
        ds_id = build_intraday_dataset(tfs, cfg, verbose=False,
                                       max_days_per_ticker=args.intraday_days)
        model_id, res_id = train_and_eval(ds_id, ds_id.meta, "intraday", cfg)

        banner("5) Live intraday server demo (streaming map updates)")
        tk = list(tfs.keys())[0]
        one = tfs[tk]["1m"]
        days = sorted(set(one.index.normalize()))
        last_day = days[-1]
        hist_before = one[one.index.normalize() < last_day]
        sess = one[one.index.normalize() == last_day]
        srv = IntradayServer(hist_before, model_id, cfg, ticker=tk)
        # feed the session in chunks, print the map at a few checkpoints
        checkpoints = [30, 90, 180, 300]
        fed = 0
        for cp in checkpoints:
            chunk = sess.iloc[fed:cp]
            if len(chunk) == 0:
                continue
            m = srv.update(chunk, top_k=4)
            fed = cp
            print(f"-- after {cp} min --")
            print(m.pretty() + "\n")

    banner("Done")
    print(f"Artifacts written to: {OUT}")
    for f in sorted(os.listdir(OUT)):
        print("  " + f)


if __name__ == "__main__":
    main()
