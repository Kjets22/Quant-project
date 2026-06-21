import warnings

warnings.filterwarnings("ignore")

from agent import evaluate, random_baseline, split_data, train
from config import default_config
from data import make_synthetic


def test_split_is_chronological():
    cfg = default_config()
    df = make_synthetic(1000, seed=7)
    tr, te = split_data(df, cfg.train.train_frac)
    assert len(tr) + len(te) == len(df)
    # train ends before test begins in time
    assert tr["timestamp"].iloc[-1] < te["timestamp"].iloc[0]


def test_trained_agent_beats_random_short():
    cfg = default_config()
    cfg.train.total_timesteps = 8000  # keep the test fast
    cfg.train.n_steps = 1024
    df = make_synthetic(2500, seed=7)
    df_tr, df_te = split_data(df, cfg.train.train_frac)
    model = train(df_tr, cfg, verbose=0)
    res = evaluate(model, df_te, cfg, "TEST")
    rnd = random_baseline(df_te, cfg, "RANDOM")
    assert res.capture_reward > rnd.capture_reward
