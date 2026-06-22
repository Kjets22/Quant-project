import numpy as np

from config import default_config
from reward import CaptureReward


def _setup(mode):
    cfg = default_config()
    cfg.reward.reward_mode = mode
    prices = np.array([100.0, 101.0, 100.5, 102.0, 101.0], dtype=float)
    leg = np.full(len(prices), 5.0)
    return cfg, prices, leg


def test_profit_reward_matches_return_formula():
    cfg, prices, leg = _setup("profit")
    rw = CaptureReward(prices, leg, cfg)
    rc = cfg.reward
    for t in range(len(prices) - 1):
        for pos in (-1, 0, 1):
            r = rw.step_reward(t, pos, 0)
            pnl = pos * (prices[t + 1] - prices[t])
            cost = abs(pos) * rc.txn_cost_frac * prices[t]
            exp = (pnl - cost) / prices[t] * rc.profit_scale
            if pos == 0:
                exp += rc.flat_bonus
            exp = float(np.clip(exp, -rc.reward_clip, rc.reward_clip))
            assert abs(r - exp) < 1e-9


def test_profit_mode_ignores_leg_range():
    """Profit reward must NOT depend on the oracle leg_range."""
    cfg, prices, _ = _setup("profit")
    a = CaptureReward(prices, np.full(len(prices), 5.0), cfg)
    b = CaptureReward(prices, np.full(len(prices), 999.0), cfg)  # totally different oracle
    for t in range(len(prices) - 1):
        assert abs(a.step_reward(t, 1, 0) - b.step_reward(t, 1, 0)) < 1e-12


def test_capture_mode_still_default_and_uses_oracle():
    cfg, prices, _ = _setup("capture")
    assert cfg.reward.reward_mode == "capture"
    a = CaptureReward(prices, np.full(len(prices), 5.0), cfg)
    b = CaptureReward(prices, np.full(len(prices), 50.0), cfg)
    # Capture mode DOES depend on leg_range, so these differ.
    assert a.step_reward(0, 1, 0) != b.step_reward(0, 1, 0)
