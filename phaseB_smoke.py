"""
phaseB_smoke.py — quick integration check for the risk-aware reward.

NOT the real evaluation (that's the queued dev-set A/B). This just trains two
short PPO agents on one small SYNTHETIC slice — pure capture vs risk-aware — to
confirm (a) PPO trains end-to-end with the new reward terms and (b) the terms
actually change the risk profile (Sharpe / drawdown). Tiny on purpose so it does
not compete with any running job. Models go under models/ (never the live path).
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pathlib import Path

from agent import evaluate, split_data, train
from config import default_config
from data import make_synthetic

Path("models").mkdir(exist_ok=True)


def run(name: str, **reward_kw):
    cfg = default_config()
    cfg.train.total_timesteps = 12_000      # tiny smoke
    cfg.train.n_steps = 1024
    cfg.train.device = "cpu"
    cfg.train.model_path = f"models/phaseB_smoke_{name}.zip"
    for k, v in reward_kw.items():
        setattr(cfg.reward, k, v)
    df = make_synthetic(2600, seed=11)
    tr, te = split_data(df, cfg.train.train_frac)
    model = train(tr, cfg, verbose=0)
    res = evaluate(model, te, cfg, name)
    return res


if __name__ == "__main__":
    print("=== Phase B smoke: pure capture vs risk-aware (synthetic, 12k steps) ===\n")
    control = run("control")
    risk = run("riskaware", use_diff_sharpe=True, diff_sharpe_w=0.3,
               dd_penalty_w=0.05, vol_penalty_w=0.02)
    print(control.block())
    print(risk.block())
    print("\n--- delta (risk-aware - control) ---")
    print(f"  Sharpe   : {risk.sharpe - control.sharpe:+.3f}")
    print(f"  max DD   : {risk.max_drawdown - control.max_drawdown:+.3f}")
    print(f"  raw P&L  : {risk.raw_pnl - control.raw_pnl:+.3f}")
    print(f"  trades   : {risk.n_trades - control.n_trades:+d}")
    print("\n(integration check only — real verdict is the queued dev-set A/B)")
