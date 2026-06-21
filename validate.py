"""
validate.py — Part 6: walk-forward validation and overfitting report.

What makes the system trustworthy. We split the series into K SEQUENTIAL folds
(never shuffled). For each fold we train on a past block and evaluate on the next
unseen block, collecting out-of-sample metrics. We report the train-vs-test gap
and flag likely overfitting when test Sharpe falls below ~50% of train Sharpe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agent import EvalResult, evaluate, random_baseline, train
from config import Config, default_config

OVERFIT_SHARPE_RATIO = 0.5  # warn if test_sharpe / train_sharpe < this


@dataclass
class FoldResult:
    fold: int
    train: EvalResult
    test: EvalResult
    random: EvalResult

    @property
    def buy_hold_pnl(self) -> float:
        return self.test.buy_hold_pnl

    @property
    def reward_gap(self) -> float:
        return self.train.capture_reward - self.test.capture_reward

    @property
    def sharpe_ratio(self) -> float:
        if abs(self.train.sharpe) < 1e-9:
            return 0.0
        return self.test.sharpe / self.train.sharpe


def walk_forward(df: pd.DataFrame, cfg: Config, k: int = 4) -> list[FoldResult]:
    """
    Expanding-window walk-forward: fold i trains on blocks [0..i] and tests on
    block i+1. Chronological throughout — earlier data only ever trains.
    """
    n = len(df)
    block = n // (k + 1)
    if block <= cfg.env.window + 5:
        raise ValueError("Series too short for the requested number of folds.")

    results: list[FoldResult] = []
    for i in range(k):
        train_end = block * (i + 1)
        test_end = block * (i + 2) if i < k - 1 else n
        df_tr = df.iloc[:train_end].reset_index(drop=True)
        df_te = df.iloc[train_end:test_end].reset_index(drop=True)
        if len(df_te) <= cfg.env.window + 2:
            continue

        print(f"[fold {i+1}/{k}] train={len(df_tr)} test={len(df_te)} — training ...")
        model = train(df_tr, cfg, verbose=0)
        res_tr = evaluate(model, df_tr, cfg, f"fold{i+1}-TRAIN")
        res_te = evaluate(model, df_te, cfg, f"fold{i+1}-TEST")
        res_rnd = random_baseline(df_te, cfg, f"fold{i+1}-RANDOM")
        results.append(FoldResult(i + 1, res_tr, res_te, res_rnd))
    return results


def report(results: list[FoldResult]) -> None:
    """Print a per-fold table, an aggregate summary, and the overfitting verdict."""
    print("\n" + "=" * 88)
    print("WALK-FORWARD REPORT")
    print("=" * 88)
    header = (
        f"{'fold':>4} | {'tr_reward':>9} {'te_reward':>9} {'gap':>7} | "
        f"{'tr_shrp':>8} {'te_shrp':>8} {'ratio':>6} | "
        f"{'te_rndR':>8} {'te_B&H':>8} {'te_MDD':>8} {'trades':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.fold:>4} | {r.train.capture_reward:>9.2f} {r.test.capture_reward:>9.2f} "
            f"{r.reward_gap:>7.2f} | {r.train.sharpe:>8.2f} {r.test.sharpe:>8.2f} "
            f"{r.sharpe_ratio:>6.2f} | {r.random.capture_reward:>8.2f} "
            f"{r.test.buy_hold_pnl:>8.2f} {r.test.max_drawdown:>8.2f} {r.test.n_trades:>6d}"
        )

    te_sharpes = np.array([r.test.sharpe for r in results])
    te_mdd = np.array([r.test.max_drawdown for r in results])
    te_reward = np.array([r.test.capture_reward for r in results])
    rnd_reward = np.array([r.random.capture_reward for r in results])
    ratios = np.array([r.sharpe_ratio for r in results])

    print("-" * len(header))
    print("AGGREGATE (out-of-sample):")
    print(f"  test Sharpe      : mean {te_sharpes.mean():.3f}  std {te_sharpes.std():.3f}")
    print(f"  test max drawdown: mean {te_mdd.mean():.3f}  std {te_mdd.std():.3f}")
    print(f"  test capture rwd : mean {te_reward.mean():.3f}  (random {rnd_reward.mean():.3f})")
    print(f"  mean te/tr Sharpe ratio : {ratios.mean():.3f}")

    # Overfitting verdict.
    print("\nOVERFITTING VERDICT:")
    mean_ratio = ratios.mean()
    beat_random = te_reward.mean() > rnd_reward.mean()
    if mean_ratio < OVERFIT_SHARPE_RATIO:
        print(f"  [WARN] test Sharpe is {mean_ratio:.0%} of train Sharpe "
              f"(< {OVERFIT_SHARPE_RATIO:.0%}) -- likely OVERFIT.")
    else:
        print(f"  [OK] test Sharpe is {mean_ratio:.0%} of train Sharpe "
              f"(>= {OVERFIT_SHARPE_RATIO:.0%}) -- no strong overfit signal.")
    if not beat_random:
        print("  [WARN] out-of-sample reward did not beat the random baseline.")
    else:
        print("  [OK] out-of-sample reward beat the random baseline.")


if __name__ == "__main__":
    from data import make_synthetic

    cfg = default_config()
    cfg.train.total_timesteps = 20_000  # short smoke run
    df = make_synthetic(4000, seed=cfg.data.synthetic_seed)
    print("=== validate.py self-test: walk-forward on synthetic ===")
    results = walk_forward(df, cfg, k=3)
    report(results)
    assert len(results) >= 1, "no folds produced"
    print("\nOK — walk-forward ran end to end")
