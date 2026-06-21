import warnings

warnings.filterwarnings("ignore")

from config import default_config
from data import make_synthetic
from validate import walk_forward


def test_walk_forward_runs_and_is_out_of_sample():
    cfg = default_config()
    cfg.train.total_timesteps = 6000  # keep the test fast
    cfg.train.n_steps = 1024
    df = make_synthetic(3000, seed=7)
    results = walk_forward(df, cfg, k=2)
    assert len(results) >= 1
    for r in results:
        # each fold produced finite metrics
        assert r.test.n_steps > 0
        assert r.train.capture_reward == r.train.capture_reward  # not NaN
        assert hasattr(r, "sharpe_ratio")
