"""
reward.py — Part 3: the capture-ratio reward (CORE).

The agent is graded on what fraction of the best achievable move it captured,
not on raw dollars. For each step we take the realized P&L of holding `position`
over the next bar, subtract transaction cost, and normalize by the oracle leg
range at the current bar (how big the move "should" have been).

Design intent:
  * Multiple trades allowed — position may flip every bar, so summed capture can
    exceed 1.0 across a leg before clipping; this is desired.
  * Flat bonus gives a small positive reward for sitting flat, discouraging the
    always-in-market overtrading trap.
  * Lookahead-safe — leg_range is future-derived and is used ONLY here, never in
    the observation (see environment.py and swings.py).

Numerical stability:
  * leg_range is already floored (> 0) in swings.build_leg_ranges, so the divide
    is safe. We additionally guard against non-finite values and clip the final
    reward to [-reward_clip, +reward_clip]. No NaN/Inf can leave step_reward.
"""

from __future__ import annotations

import numpy as np

from config import Config


class CaptureReward:
    def __init__(self, prices: np.ndarray, leg_range: np.ndarray, cfg: Config):
        self.prices = np.asarray(prices, dtype=np.float64)
        self.leg_range = np.asarray(leg_range, dtype=np.float64)
        self.cfg = cfg
        if self.prices.shape != self.leg_range.shape:
            raise ValueError("prices and leg_range must have the same shape")
        # Defensive floor in case a caller passes an unfloored range.
        self._eps = 1e-12
        self.leg_range = np.maximum(self.leg_range, self._eps)
        self.reset()

    def reset(self) -> None:
        """Zero the per-episode risk-term state (online Sharpe moments, equity)."""
        self._A = 0.0      # EWMA of returns
        self._B = 0.0      # EWMA of squared returns
        self._cum = 0.0    # cumulative normalized return (equity in capture units)
        self._peak = 0.0   # running max of _cum (for drawdown)

    def _risk_terms(self, R: float, rc) -> float:
        """
        Phase-B risk-aware add-ons. Returns 0.0 (and touches no state) when every
        weight is off, so the pure capture reward is recovered exactly.

        Lookahead-safe: every term is a function only of the agent's own realized
        return stream up to the current bar — never of the observation or future.
        """
        active = rc.use_diff_sharpe or rc.dd_penalty_w != 0.0 or rc.vol_penalty_w != 0.0
        if not active:
            return 0.0

        extra = 0.0
        dA = R - self._A
        dB = R * R - self._B
        denom = self._B - self._A * self._A

        # Differential Sharpe ratio (Moody & Saffell): uses the PREVIOUS moments.
        if rc.use_diff_sharpe and denom > 1e-12:
            D = (self._B * dA - 0.5 * self._A * dB) / (denom ** 1.5)
            if np.isfinite(D):
                extra += rc.diff_sharpe_w * D

        # Advance the online EWMA moments.
        eta = rc.diff_sharpe_eta
        self._A += eta * dA
        self._B += eta * dB

        # Equity & drawdown (capture units).
        self._cum += R
        self._peak = max(self._peak, self._cum)
        if rc.dd_penalty_w != 0.0:
            extra -= rc.dd_penalty_w * (self._peak - self._cum)   # drawdown >= 0

        # Volatility penalty on the running return variance.
        if rc.vol_penalty_w != 0.0:
            extra -= rc.vol_penalty_w * max(self._B - self._A * self._A, 0.0)

        return extra

    def step_reward(self, t: int, position: int, prev_position: int) -> float:
        """
        Reward for holding `position` over bar t -> t+1.

        position, prev_position in {-1, 0, +1}. `prev_position` is used to charge
        the cost of changing exposure (turnover) at this bar.
        """
        if t + 1 >= len(self.prices):
            return 0.0

        rc = self.cfg.reward

        price_now = self.prices[t]
        pnl = position * (self.prices[t + 1] - price_now)
        traded = abs(position - prev_position)
        cost = traded * rc.txn_cost_frac * price_now
        raw_pnl = pnl - cost

        R = raw_pnl / self.leg_range[t]            # normalized return (leg_range floored)
        reward = R
        if position == 0:
            reward += rc.flat_bonus

        # Phase-B risk-aware terms (no-op unless a weight is enabled).
        reward += self._risk_terms(R, rc)

        # Final safety: no NaN/Inf may ever reach the agent.
        if not np.isfinite(reward):
            reward = 0.0
        return float(np.clip(reward, -rc.reward_clip, rc.reward_clip))


if __name__ == "__main__":
    from config import default_config
    from data import make_synthetic
    from swings import alternate_swings, build_leg_ranges, detect_swings

    cfg = default_config()
    df = make_synthetic(2000, seed=cfg.data.synthetic_seed)
    close = df["close"].to_numpy()
    leg = build_leg_ranges(df, cfg)
    reward = CaptureReward(close, leg, cfg)

    # Find the first L -> H up-leg in the alternating sequence.
    swings = alternate_swings(detect_swings(df, cfg))
    up_leg = None
    for a, b in zip(swings[:-1], swings[1:]):
        if a[1] == "L" and b[1] == "H":
            up_leg = (a[0], b[0])
            break
    assert up_leg is not None, "no L->H leg found"
    lo_idx, hi_idx = up_leg

    # Simulate a perfect long held every bar from the low index to the high index.
    total = 0.0
    prev = 0
    for t in range(lo_idx, hi_idx):
        total += reward.step_reward(t, position=+1, prev_position=prev)
        prev = +1

    print("=== reward.py self-test: perfect long over first L->H leg ===")
    print(f"leg indices       : {lo_idx} -> {hi_idx} (len {hi_idx - lo_idx})")
    print(f"swing low price   : {close[lo_idx]:.4f}")
    print(f"swing high price  : {close[hi_idx]:.4f}")
    print(f"oracle leg range  : {leg[lo_idx]:.4f}")
    print(f"captured ratio    : {total:.4f}")
    # A close-trader paying entry cost, using close-to-close moves vs an oracle
    # built on wick extremes, should capture a sensible fraction near (usually a
    # bit below) 1.0 — but flips/clipping can push it modestly above 1.
    assert 0.3 <= total <= 2.0, f"capture ratio out of sane range: {total}"
    print("OK — capture ratio is a sensible fraction near 1.0")
