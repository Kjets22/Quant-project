"""
Stacked error-predictor (the user's idea, done honestly), QQQ + SPY.

1. Train the BASE trading agent (all-weather + cross + regime + TIME features).
2. Roll it out on TRAIN data; label each in-position bar 'wrong' if it lost money
   (position * next_return < 0).
3. Train a META model (gradient boosting) to predict, from the SAME causal features
   the agent saw, P(the base is wrong).
4. THE HONESTY CHECK: out-of-sample AUC of predicting the base's errors. If AUC ~=
   0.5 the errors are unpredictable (the idea can't work); if > ~0.55 it can.
5. Overlay: on TEST, force FLAT whenever P(wrong) is high. Compare base vs gated vs B&H.

  python run_stacked.py            # launcher + report
  python run_stacked.py --ticker QQQ --fold 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "runs" / "phase3" / "stacked"
OUT.mkdir(parents=True, exist_ok=True)
TICKERS = ["QQQ", "SPY"]
K = 6
TIMESTEPS = int(os.environ.get("STK_TIMESTEPS", "150000"))
TXN = 0.0002


def cfg_for(ticker):
    from basket import ticker_cfg
    cfg = ticker_cfg(ticker)
    cfg.train.device = "cpu"
    cfg.train.total_timesteps = TIMESTEPS
    cfg.train.ent_coef = 0.01
    cfg.train.net_arch = "128,128"
    cfg.reward.reward_mode = "money"
    cfg.reward.flat_bonus = 0.01
    cfg.reward.turnover_penalty_w = 0.4
    cfg.reward.allow_short = True
    cfg.env.window = 96
    cfg.env.use_regime_features = True
    cfg.env.use_cross_features = True
    cfg.env.use_time_features = True       # time-since-open feeds base + meta
    cfg.env.short_only_in_down = True
    cfg.env.force_long_in_up = True
    return cfg


def rollout(model, df, cfg):
    """Return (X=obs seen, pos held, next_$ move, price) per bar."""
    from environment import CaptureTradingEnv
    env = CaptureTradingEnv(df, cfg)
    obs, _ = env.reset(seed=0)
    X, pos, nret, price = [], [], [], []
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        cur = obs.copy()
        tb = env.t
        obs, _r, term, trunc, info = env.step(int(a))
        X.append(cur)
        pos.append(info["position"])
        nret.append(env.close_px[tb + 1] - env.close_px[tb])
        price.append(env.close_px[tb])
        done = term or trunc
    bh = float(env.close_px[-1] - env.close_px[env.window + 1])
    return (np.asarray(X), np.asarray(pos, float), np.asarray(nret, float),
            np.asarray(price, float), bh)


def pnl_of(pos, nret, price):
    gross = float((pos * nret).sum())
    cost = float((np.abs(np.diff(np.concatenate([[0.0], pos]))) * TXN * price).sum())
    sh = 0.0
    pl = pos * nret
    if pl.std() > 1e-9:
        sh = float(pl.mean() / pl.std() * np.sqrt(len(pl)))
    return gross - cost, sh


def run_one(ticker, fold):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    from agent import train
    from run_cross import aligned_hourly
    from validate import fold_bounds
    out = OUT / f"{ticker}_f{fold}.csv"
    if out.exists():
        print(f"skip {ticker} f{fold}")
        return
    cfg = cfg_for(ticker)
    dev = aligned_hourly(ticker)
    n, window = len(dev), cfg.env.window
    tr_end, te_end = fold_bounds(n, K, window, fold)
    df_tr = dev.iloc[:tr_end].reset_index(drop=True)
    df_te = dev.iloc[tr_end:te_end].reset_index(drop=True)

    base = train(df_tr, cfg)
    Xtr, ptr, rtr, _pr, _ = rollout(base, df_tr, cfg)
    Xte, pte, rte, prte, bh = rollout(base, df_te, cfg)

    # 'wrong' label = held a position that lost money that bar.
    ytr = ((ptr != 0) & (ptr * rtr < 0)).astype(int)
    yte = ((pte != 0) & (pte * rte < 0)).astype(int)
    atr = ptr != 0      # active (in-position) training bars
    ate = pte != 0

    auc = float("nan")
    base_pnl, base_sh = pnl_of(pte, rte, prte)
    gated_pnl, gated_sh = base_pnl, base_sh
    n_gated = 0
    if atr.sum() > 30 and ytr[atr].sum() > 5 and len(np.unique(ytr[atr])) == 2:
        clf = GradientBoostingClassifier(max_depth=3, n_estimators=120, subsample=0.8)
        clf.fit(Xtr[atr], ytr[atr])
        proba = clf.predict_proba(Xte)[:, 1]
        if ate.sum() > 10 and len(np.unique(yte[ate])) == 2:
            auc = float(roc_auc_score(yte[ate], proba[ate]))
        # Overlay: flat out the bars the meta model flags as high-risk (top 30%).
        thresh = np.quantile(proba[ate], 0.70) if ate.sum() else 1.0
        gate = pte.copy()
        gate[(pte != 0) & (proba >= thresh)] = 0.0
        n_gated = int(((pte != 0) & (proba >= thresh)).sum())
        gated_pnl, gated_sh = pnl_of(gate, rte, prte)

    pd.DataFrame([{
        "ticker": ticker, "fold": fold, "meta_auc": auc,
        "base_pnl": base_pnl, "base_sharpe": base_sh,
        "gated_pnl": gated_pnl, "gated_sharpe": gated_sh,
        "buy_hold_pnl": bh, "n_gated": n_gated,
        "base_beats_bh": int(base_pnl > bh), "gated_beats_bh": int(gated_pnl > bh),
    }]).to_csv(out, index=False)
    print(f"DONE {ticker} f{fold}: AUC={auc:.3f} base={base_pnl:.1f} "
          f"gated={gated_pnl:.1f} B&H={bh:.1f}")


def launch():
    print(f"Stacked error-predictor QQQ/SPY @ {TIMESTEPS}")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"; env["MKL_NUM_THREADS"] = "2"
    env["CT_TORCH_THREADS"] = "2"; env["PYTHONWARNINGS"] = "ignore"
    procs = [subprocess.Popen(
        [sys.executable, "-u", "run_stacked.py", "--ticker", t, "--fold", str(f)],
        cwd=str(HERE), env=env) for t in TICKERS for f in range(K)]
    for p in procs:
        p.wait()
    report()


def report():
    files = sorted(OUT.glob("*_f*.csv"))
    if not files:
        print("no results.")
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print("\n=== STACKED ERROR-PREDICTOR — can we predict the base's mistakes? ===")
    for tk, s in d.groupby("ticker"):
        print(f"  {tk}: META AUC {s.meta_auc.mean():.3f} (0.50=coin flip)  | "
              f"base P&L {s.base_pnl.mean():7.2f} (beat-B&H {s.base_beats_bh.mean():.0%}) "
              f"-> gated P&L {s.gated_pnl.mean():7.2f} (beat {s.gated_beats_bh.mean():.0%})  "
              f"B&H {s.buy_hold_pnl.mean():.2f}")
    print(f"\n  POOLED: META AUC {d.meta_auc.mean():.3f}  "
          f"base beat-B&H {d.base_beats_bh.mean():.0%} -> gated {d.gated_beats_bh.mean():.0%}")
    print("  READ: AUC ~0.50 => errors are unpredictable (idea can't work, honest).")
    print("        AUC > ~0.55 AND gated > base => error-prediction adds real value.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.ticker and a.fold is not None:
        run_one(a.ticker, a.fold)
    else:
        launch()
