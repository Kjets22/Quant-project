"""
diagnose.py — WHY can't the agent beat buy-and-hold? (training-free analysis)

Three questions, answered with data (no RL training):
  1. Is there exploitable SIGNAL in 5-min returns? (autocorrelation + an honest
     out-of-sample linear predictor: R^2 and directional accuracy)
  2. What do TRANSACTION COSTS do? (cost per round-trip vs typical bar move ->
     the directional edge you'd need just to break even)
  3. Does ANY simple strategy beat B&H? (always-long, momentum, mean-revert,
     net of cost) — if not, the agent never had an easy target.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from basket import ticker_cfg
from lockbox import Lockbox, load_dev_ticker

TXN = 0.0002  # txn_cost_frac (per unit traded, fraction of price)


def analyze(ticker: str) -> None:
    dev = load_dev_ticker(ticker, Lockbox.load_or_build())
    close = dev["close"].to_numpy(float)
    r = np.diff(np.log(close))                      # 5-min log returns
    print(f"\n================  {ticker}  ================")
    print(f"bars={len(r)}  ret std={r.std()*1e4:.1f} bps  "
          f"mean={r.mean()*1e4:+.2f} bps/bar")

    # ---- 1. Predictability -------------------------------------------------
    print("\n[1] SIGNAL — autocorrelation of next-bar return:")
    for lag in (1, 2, 3, 5, 10):
        ac = np.corrcoef(r[:-lag], r[lag:])[0, 1]
        print(f"     lag {lag:>2}: {ac:+.4f}")
    # Honest OOS linear predictor: next return from past L returns.
    L = 20
    X = np.stack([r[i:len(r) - L + i] for i in range(L)], axis=1)
    y = r[L:]
    cut = int(len(y) * 0.7)
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    pred = np.hstack([Xte, np.ones((len(Xte), 1))]) @ coef
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    dir_acc = float((np.sign(pred) == np.sign(yte)).mean())
    print(f"     OOS linear predictor:  R^2 = {r2:+.5f}   "
          f"direction accuracy = {dir_acc:.1%}  (50% = no edge)")

    # ---- 2. Transaction costs ---------------------------------------------
    rt_cost_bps = 2 * TXN * 1e4                      # round-trip cost in bps
    typ_move_bps = r.std() * 1e4
    print("\n[2] COSTS:")
    print(f"     round-trip cost = {rt_cost_bps:.1f} bps   "
          f"typical bar move = {typ_move_bps:.1f} bps")
    print(f"     -> a round-trip eats {rt_cost_bps/typ_move_bps:.0%} of a typical "
          f"1-bar move; you must predict direction well ABOVE 50% to overcome it.")

    # ---- 3. Simple strategies (net of cost) vs B&H ------------------------
    print("\n[3] SIMPLE STRATEGIES on this series (net of cost), P&L in $:")
    price = close[1:]
    rets = close[1:] - close[:-1]                    # $ moves per bar (1 share)

    def pnl(position: np.ndarray) -> float:
        pos = position.astype(float)
        gross = float((pos[:-1] * rets[1:]).sum())
        traded = np.abs(np.diff(np.concatenate([[0.0], pos])))
        cost = float((traded * TXN * price).sum())
        return gross - cost, int((np.diff(pos) != 0).sum())

    n = len(price)
    bh = float(close[-1] - close[L])
    strategies = {
        "Buy & Hold (long)": np.ones(n),
        "Momentum (sign last ret)": np.sign(np.concatenate([[0], rets[:-1]])),
        "Momentum (sign last 12)": np.sign(np.convolve(rets, np.ones(12), "full")[:n]),
        "Mean-revert (-sign last)": -np.sign(np.concatenate([[0], rets[:-1]])),
        "Always flat": np.zeros(n),
    }
    print(f"     {'strategy':28} {'net P&L':>10} {'trades':>8}")
    for name, pos in strategies.items():
        p, tr = pnl(pos)
        flag = "  <- B&H" if name.startswith("Buy") else ""
        print(f"     {name:28} {p:>10.2f} {tr:>8d}{flag}")


if __name__ == "__main__":
    for tk in ("QQQ", "TLT"):
        analyze(tk)
    print("\n" + "=" * 50)
    print("Read: if R^2~0 and direction~50%, returns are ~unpredictable; if no")
    print("simple strategy beats B&H net of cost, the agent had no easy edge and")
    print("active trading mostly donates the round-trip cost on every flip.")
