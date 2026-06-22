import numpy as np

from config import default_config
from reward import CaptureReward


def _prices_leg():
    prices = np.array([100, 101, 100.5, 102, 101.5, 103, 101, 104], dtype=float)
    leg = np.full(len(prices), 5.0)
    return prices, leg


def test_backward_compatible_when_weights_zero():
    """All Phase-B weights off => exactly the pure capture reward."""
    cfg = default_config()  # everything off by default
    prices, leg = _prices_leg()
    rw = CaptureReward(prices, leg, cfg)
    rc = cfg.reward
    for t in range(len(prices) - 1):
        for pos in (-1, 0, 1):
            r = rw.step_reward(t, pos, 0)
            pnl = pos * (prices[t + 1] - prices[t])
            cost = abs(pos) * rc.txn_cost_frac * prices[t]
            exp = (pnl - cost) / leg[t]
            if pos == 0:
                exp += rc.flat_bonus
            exp = float(np.clip(exp, -rc.reward_clip, rc.reward_clip))
            assert abs(r - exp) < 1e-12


def test_dd_penalty_reduces_reward_in_drawdown():
    prices, leg = _prices_leg()
    # Long the whole way; the series rises and dips, creating drawdowns.
    base = default_config()
    pen = default_config()
    pen.reward.dd_penalty_w = 0.5
    rw0 = CaptureReward(prices, leg, base)
    rw1 = CaptureReward(prices, leg, pen)
    tot0 = tot1 = 0.0
    prev = 0
    for t in range(len(prices) - 1):
        tot0 += rw0.step_reward(t, 1, prev)
        tot1 += rw1.step_reward(t, 1, prev)
        prev = 1
    # Penalty can only subtract -> never higher, and strictly lower given dips.
    assert tot1 < tot0


def test_diff_sharpe_is_finite_and_active():
    prices, leg = _prices_leg()
    cfg = default_config()
    cfg.reward.use_diff_sharpe = True
    cfg.reward.diff_sharpe_w = 1.0
    rw = CaptureReward(prices, leg, cfg)
    rw0 = CaptureReward(prices, leg, default_config())
    differs = False
    prev = 0
    for t in range(len(prices) - 1):
        r = rw.step_reward(t, 1, prev)
        r0 = rw0.step_reward(t, 1, prev)
        assert np.isfinite(r)
        if abs(r - r0) > 1e-9:
            differs = True
        prev = 1
    assert differs  # the differential-Sharpe term actually changed the reward


def test_no_nan_inf_with_all_terms_and_extreme_prices():
    prices = np.array([100.0, 1e6, 1.0, 100.0, 1e-3, 50.0], dtype=float)
    leg = np.ones_like(prices)
    cfg = default_config()
    cfg.reward.use_diff_sharpe = True
    cfg.reward.diff_sharpe_w = 1.0
    cfg.reward.dd_penalty_w = 0.3
    cfg.reward.vol_penalty_w = 0.2
    rw = CaptureReward(prices, leg, cfg)
    prev = 0
    for t in range(len(prices) - 1):
        for pos in (-1, 0, 1):
            r = rw.step_reward(t, pos, prev)
            assert np.isfinite(r)
            assert -cfg.reward.reward_clip <= r <= cfg.reward.reward_clip
        prev = 1


def test_reset_clears_state():
    prices, leg = _prices_leg()
    cfg = default_config()
    cfg.reward.use_diff_sharpe = True
    cfg.reward.diff_sharpe_w = 1.0
    cfg.reward.dd_penalty_w = 0.3
    rw = CaptureReward(prices, leg, cfg)

    run1 = [rw.step_reward(t, 1, 1 if t else 0) for t in range(len(prices) - 1)]
    rw.reset()
    run2 = [rw.step_reward(t, 1, 1 if t else 0) for t in range(len(prices) - 1)]
    assert np.allclose(run1, run2)
