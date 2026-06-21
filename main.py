"""
main.py — CLI entry point for the Capture Trader research scaffold.

Pipeline: load data -> chronological split -> train PPO -> evaluate TRAIN + TEST
-> print the overfitting gap. Optionally run full walk-forward validation.

Reads POLYGON_API_KEY from the environment. With no key (and synthetic fallback
enabled), the whole pipeline runs on synthetic data.

Examples:
  python main.py                       # default config, synthetic or Polygon
  python main.py --timesteps 200000    # longer train
  python main.py --synthetic           # force synthetic data
  python main.py --walk-forward --folds 4
"""

from __future__ import annotations

# Enable the OS trust store BEFORE importing anything that pulls in urllib3
# (stable-baselines3, requests). Must run first to avoid a global-patch recursion.
try:  # pragma: no cover - environment dependent
    import truststore

    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

import argparse  # noqa: E402

from agent import evaluate, random_baseline, split_data, train  # noqa: E402
from config import default_config  # noqa: E402
from data import load_data, make_synthetic  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Capture Trader RL pipeline")
    p.add_argument("--timesteps", type=int, default=None,
                   help="override total_timesteps")
    p.add_argument("--synthetic", action="store_true",
                   help="force synthetic data (ignore Polygon key)")
    p.add_argument("--synthetic-n", type=int, default=None,
                   help="length of synthetic series")
    p.add_argument("--ticker", type=str, default=None, help="override ticker")
    p.add_argument("--walk-forward", action="store_true",
                   help="run walk-forward validation instead of single split")
    p.add_argument("--folds", type=int, default=4, help="walk-forward folds")
    args = p.parse_args()

    cfg = default_config()
    if args.timesteps is not None:
        cfg.train.total_timesteps = args.timesteps
    if args.synthetic_n is not None:
        cfg.data.synthetic_n = args.synthetic_n
    if args.ticker is not None:
        cfg.data.ticker = args.ticker

    print("=" * 70)
    print("CAPTURE TRADER")
    print("=" * 70)
    print(f"ticker        : {cfg.data.ticker}")
    print(f"bar           : {cfg.data.multiplier} {cfg.data.timespan}")
    print(f"date range    : {cfg.data.start_date} .. {cfg.data.end_date}")
    print(f"total_steps   : {cfg.train.total_timesteps}")
    print(f"polygon key   : {'set' if cfg.api_key else 'NOT set (synthetic)'}")

    # 1) Load data.
    if args.synthetic:
        print("\n[1/4] Loading SYNTHETIC data (forced) ...")
        df = make_synthetic(cfg.data.synthetic_n, cfg.data.synthetic_seed)
    else:
        print("\n[1/4] Loading data ...")
        df = load_data(cfg)
    print(f"      rows: {len(df)}  range: {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}")
    print(f"      price: {df['close'].min():.2f} .. {df['close'].max():.2f}")

    if args.walk_forward:
        from validate import report, walk_forward
        print("\n[2/2] Walk-forward validation ...")
        results = walk_forward(df, cfg, k=args.folds)
        report(results)
        return

    # 2) Chronological split.
    print("\n[2/4] Chronological split ...")
    df_train, df_test = split_data(df, cfg.train.train_frac)
    print(f"      train rows: {len(df_train)}   test rows: {len(df_test)}")

    # 3) Train.
    print("\n[3/4] Training PPO ...")
    model = train(df_train, cfg, verbose=0)
    print(f"      saved model -> {cfg.train.model_path}")

    # 4) Evaluate.
    print("\n[4/4] Evaluating ...\n")
    rnd_tr = random_baseline(df_train, cfg, "RANDOM (train)")
    rnd_te = random_baseline(df_test, cfg, "RANDOM (test)")
    res_tr = evaluate(model, df_train, cfg, "TRAIN")
    res_te = evaluate(model, df_test, cfg, "TEST")
    print(res_tr.block())
    print(rnd_tr.block())
    print(res_te.block())
    print(rnd_te.block())

    gap = res_tr.capture_reward - res_te.capture_reward
    ratio = (res_te.sharpe / res_tr.sharpe) if abs(res_tr.sharpe) > 1e-9 else 0.0
    print("\n" + "=" * 70)
    print("OVERFITTING GAP")
    print("=" * 70)
    print(f"  train reward - test reward : {gap:.3f}")
    print(f"  test Sharpe / train Sharpe : {ratio:.3f}")
    if ratio < 0.5:
        print("  [WARN] test Sharpe < 50% of train Sharpe -- likely overfit.")
    else:
        print("  [OK] no strong overfit signal on this split.")
    print(f"  test beats random?         : {res_te.capture_reward > rnd_te.capture_reward}")


if __name__ == "__main__":
    main()
