"""Evaluation harness: score a trained model against classic baselines on the
same examples, using both per-bin AP/AUC and zone-level precision@K/recall@K.

Design choice for fairness: classic baselines are polarity-agnostic level
generators (a pivot is just a price), so the *zone-location* test scores every
method -- model included -- against the merged set of true reaction levels
(support peaks UNION resistance peaks). The model's extra ability to label a
level support-vs-resistance is reported separately via per-channel AP/AUC, which
baselines structurally cannot produce.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import Config
from .metrics import (extract_peaks, split_peaks_to_levels, zone_hits,
                      per_bin_ap_auc)
from .baselines import compute_baselines


def _levels_by_example(intensity_flat: np.ndarray, bin_price_flat: np.ndarray,
                       ex_flat: np.ndarray, atr_by_ex: Dict[int, float],
                       cfg: Config, top_k: int | None) -> Dict[int, np.ndarray]:
    ec = cfg.eval
    out = {}
    for ex in np.unique(ex_flat):
        m = ex_flat == ex
        peaks = extract_peaks(intensity_flat[m], bin_price_flat[m],
                              atr_by_ex[int(ex)], ec.peak_min_dist_atr,
                              ec.peak_min_intensity, top_k=top_k)
        out[int(ex)] = split_peaks_to_levels(peaks)
    return out


def _split_by_side(levels: np.ndarray, ref: float, atr: float,
                   tol_atr: float = 0.25):
    """Split a flat baseline level list into (support_side, resistance_side)
    relative to ref. Levels within tol of ref count for both sides (a level at
    current price can act as either)."""
    sup, res = [], []
    tol = tol_atr * max(atr, 1e-9)
    for p in levels:
        if p <= ref + tol:
            sup.append(p)
        if p >= ref - tol:
            res.append(p)
    return np.array(sup), np.array(res)


def evaluate(model, ds, meta: pd.DataFrame, split_mask: np.ndarray,
             cfg: Config) -> Dict:
    """Return tidy evaluation rows + per-bin summary for the rows in
    `split_mask` (boolean over ds.X / meta)."""
    ec = cfg.eval
    X = ds.X.loc[split_mask]
    msub = meta.loc[split_mask].reset_index(drop=True)
    Xr = X.reset_index(drop=True)
    sm = split_mask.to_numpy() if hasattr(split_mask, "to_numpy") else split_mask
    y_sup = ds.y_support[sm]
    y_res = ds.y_resistance[sm]

    pred_sup, pred_res = model.predict(Xr)

    ex_flat = msub["example_id"].to_numpy()
    bp_flat = msub["bin_price"].to_numpy()
    atr_by_ex = {int(e): float(msub.loc[ex_flat == e, "atr"].iloc[0])
                 for e in np.unique(ex_flat)}
    ref_by_ex = {int(e): float(msub.loc[ex_flat == e, "ref"].iloc[0])
                 for e in np.unique(ex_flat)}

    kmax = max(ec.top_k_zones)
    # combined (polarity-agnostic) intensities
    true_comb = np.maximum(y_sup, y_res)
    pred_comb = np.maximum(pred_sup, pred_res)
    true_levels = _levels_by_example(true_comb, bp_flat, ex_flat, atr_by_ex,
                                     cfg, top_k=None)
    model_levels = _levels_by_example(pred_comb, bp_flat, ex_flat, atr_by_ex,
                                      cfg, top_k=kmax)

    # per-channel true + model levels
    true_sup_lv = _levels_by_example(y_sup, bp_flat, ex_flat, atr_by_ex, cfg, None)
    true_res_lv = _levels_by_example(y_res, bp_flat, ex_flat, atr_by_ex, cfg, None)
    model_sup_lv = _levels_by_example(pred_sup, bp_flat, ex_flat, atr_by_ex, cfg, kmax)
    model_res_lv = _levels_by_example(pred_res, bp_flat, ex_flat, atr_by_ex, cfg, kmax)

    baselines = compute_baselines(msub, Xr, cfg, np.unique(ex_flat))

    # assemble per-method, per-channel level maps
    # channel "combined" + "support" + "resistance"
    method_channel: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {
        "model": {"combined": model_levels, "support": model_sup_lv,
                  "resistance": model_res_lv},
    }
    for name, lv_by_ex in baselines.items():
        sup_map, res_map = {}, {}
        for ex, lv in lv_by_ex.items():
            s, r = _split_by_side(lv, ref_by_ex[ex], atr_by_ex[ex], ec.hit_tol_atr)
            sup_map[ex], res_map[ex] = s, r
        method_channel[name] = {"combined": lv_by_ex, "support": sup_map,
                                "resistance": res_map}

    truths = {"combined": true_levels, "support": true_sup_lv,
              "resistance": true_res_lv}

    rows: List[dict] = []
    for name, ch_maps in method_channel.items():
        for channel, truth in truths.items():
            levels_by_ex = ch_maps[channel]
            for k in ec.top_k_zones:
                ps, rs, es = [], [], []
                for ex, tl in truth.items():
                    atr = atr_by_ex[ex]
                    pl = levels_by_ex.get(ex, np.array([]))[:k]
                    h = zone_hits(pl, tl, atr, ec.hit_tol_atr)
                    ps.append(h["precision"])
                    rs.append(h["recall"])
                    if np.isfinite(h["loc_err_atr"]):
                        es.append(h["loc_err_atr"])
                p = float(np.mean(ps)) if ps else 0.0
                r = float(np.mean(rs)) if rs else 0.0
                rows.append({
                    "method": name, "channel": channel, "k": k,
                    "precision": p, "recall": r,
                    "f1": 0.0 if p + r == 0 else 2 * p * r / (p + r),
                    "loc_err_atr": float(np.mean(es)) if es else np.nan,
                })

    # per-bin AP/AUC for the model (where support/resistance labelling matters)
    perbin = {
        "support": per_bin_ap_auc(y_sup, pred_sup),
        "resistance": per_bin_ap_auc(y_res, pred_res),
    }
    return {"rows": rows, "perbin": perbin,
            "n_examples": int(len(np.unique(ex_flat))),
            "_pred": (pred_sup, pred_res), "_true": (y_sup, y_res),
            "_meta": msub, "_X": Xr,
            "_true_levels": true_levels, "_model_levels": model_levels,
            "_baselines": baselines, "_atr_by_ex": atr_by_ex}


def summarise(rows: List[dict], perbin: Dict) -> str:
    df = pd.DataFrame(rows)
    out = ["Per-bin (model):"]
    for ch, d in perbin.items():
        out.append(f"  {ch:11s} AP={d['ap']:.3f}  AUC={d['auc']:.3f}  "
                   f"base={d['base_rate']:.3f}  lift={d['lift']:.1f}x")
    for channel in ("support", "resistance", "combined"):
        out.append(f"\nZone location -- {channel} channel "
                   f"(precision / recall @ K):")
        cdf = df[df["channel"] == channel]
        for k in sorted(cdf["k"].unique()):
            out.append(f"  --- top-{k} ---")
            sub = cdf[cdf["k"] == k].sort_values("f1", ascending=False)
            for _, r in sub.iterrows():
                star = "  <== MODEL" if r["method"] == "model" else ""
                le = r["loc_err_atr"]
                le_s = f"{le:.2f}" if np.isfinite(le) else "  na"
                out.append(f"    {r['method']:14s}  P={r['precision']:.2f}  "
                           f"R={r['recall']:.2f}  F1={r['f1']:.2f}  "
                           f"locErr={le_s} ATR{star}")
    return "\n".join(out)
