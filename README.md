# Capture Trader — oracle-normalized RL trading research scaffold

A reinforcement-learning trading research system whose defining feature is a
custom **oracle-normalized "capture-ratio" reward**: the agent is graded on what
fraction of the *best achievable move* it captured, not on raw dollars.

> ⚠️ **This is a research scaffold, not financial advice.** Synthetic and
> short-run numbers mean nothing. Always watch the TRAIN/TEST gap and paper-trade
> before risking real capital.

---

## File-by-file

| File | Part | Responsibility |
|------|------|----------------|
| [`config.py`](config.py) | 0 | All parameters as dataclasses; loads `.env`; reads `POLYGON_API_KEY`. |
| [`data.py`](data.py) | 1 | Polygon.io fetch (pagination + 429 backoff), CSV disk cache, regime-switching synthetic fallback. |
| [`fetch_data.py`](fetch_data.py) | 1 | Standalone script: download data once and cache it to `data_cache/`. |
| [`swings.py`](swings.py) | 2 | Swing detection + alternating sequence + **oracle leg ranges** (future-derived). |
| [`reward.py`](reward.py) | 3 | **The capture-ratio reward** (core). Normalizes P&L by the oracle leg range. |
| [`environment.py`](environment.py) | 4 | Gymnasium env. **Lookahead-safe** observation; oracle used only for reward. |
| [`agent.py`](agent.py) | 5 | PPO training + deterministic evaluation/metrics + random baseline. |
| [`validate.py`](validate.py) | 6 | Walk-forward validation + overfitting report. |
| [`main.py`](main.py) | — | CLI: load → split → train → evaluate → overfitting gap (or walk-forward). |
| `tests/` | — | pytest suite (one file per module) covering all acceptance criteria. |

---

## The lookahead rule (read this)

**Swing / leg / oracle information is derived from the future. It is used ONLY to
compute the reward. It NEVER appears in the observation the agent sees.**

- The observation (`environment.py::_get_obs`) is built exclusively from data at
  or before the current bar: scaled past log-returns, plus past-only rolling
  momentum / volatility / range-position, plus the agent's own current position.
- The oracle (`swings.build_leg_ranges`) is passed only to `CaptureReward`.
- `tests/test_environment.py::test_observation_has_no_oracle_leakage` proves it:
  scrambling `leg_range` does not change a single observation.

Other hard constraints honored: chronological splits only (never shuffled);
the oracle range is floored and the final reward is clipped (no NaN/Inf can reach
the agent — asserted in `environment.step`); the API key is never hardcoded;
HTTP errors and 429 rate limits are handled with backoff.

---

## Quick start

### 1. Install
```powershell
pip install -r requirements.txt
```

### 2. Provide a Polygon key (optional — synthetic fallback works without one)
Create a `.env` file next to `config.py`:
```
POLYGON_API_KEY=your_key_here
```
`config.py` loads it automatically (the cross-platform equivalent of `export`).
Or set it in your shell:
```powershell
# Windows (persistent, user-level):
[Environment]::SetEnvironmentVariable('POLYGON_API_KEY','your_key','User')
```
```bash
# macOS/Linux:
export POLYGON_API_KEY=your_key
```

> **TLS note:** on networks that intercept TLS (corporate proxy / antivirus),
> `truststore` (in `requirements.txt`) routes verification through the OS trust
> store. It is injected *before* `requests` is imported.

### 3. Download & cache the data (run once)
```powershell
python fetch_data.py                      # default AAPL 5-minute, range in config
python fetch_data.py --ticker MSFT --start 2024-01-01 --end 2024-06-30
python fetch_data.py --refresh            # ignore cache, re-download
```
Data is cached to `data_cache/<ticker>_<bar>_<start>_<end>.csv`. Training reads
the cache, so you only hit the API once.

### 4. Train & evaluate
```powershell
python main.py                            # uses cache if present, else fetch/synthetic
python main.py --timesteps 500000         # a real run (raise well above the 50k smoke default)
python main.py --synthetic                # force synthetic data, no key needed
python main.py --walk-forward --folds 4   # out-of-sample walk-forward + overfit verdict
```

### 5. Tests
```powershell
python -m pytest -q
```

---

## How the reward works (`reward.py`)

For each bar `t` with target `position ∈ {-1,0,+1}`:
```
pnl        = position * (price[t+1] - price[t])
cost       = |position - prev_position| * txn_cost_frac * price[t]
normalized = (pnl - cost) / leg_range[t]      # leg_range is floored > 0
if position == 0: normalized += flat_bonus    # small reward for sitting flat
reward     = clip(normalized, -reward_clip, +reward_clip)
```
- `leg_range[t]` is the oracle's best-achievable move for the leg bar `t` belongs
  to — so the reward is *what fraction of the move you captured*.
- The agent may flip every bar, so summed capture can exceed 1.0 over a leg
  before clipping. This is intended.
- The flat bonus discourages the always-in-market overtrading trap.

---

## Honest expectations

- **Synthetic numbers mean nothing.** The synthetic generator is a toy; good
  results on it are not evidence of edge.
- **Short runs overfit.** The smoke-test budgets (6k–50k steps) memorize the
  train slice. Expect a large TRAIN/TEST gap — the report will say so.
- **Watch the overfitting verdict**, not the train reward. If test Sharpe is
  below ~50% of train Sharpe, the model is likely overfit.
- **Baselines matter.** Every report shows buy-&-hold and a random agent. Beating
  random is the floor, not success; beating buy-&-hold out-of-sample is the bar.
- **Paper trade before real capital.** Transaction costs, slippage, market impact,
  and regime shifts are only crudely modeled here.

---

## Optional enhancements (Phase 7)

Not yet implemented; each is meant to live behind a config flag and be
independently testable: regime-adaptive reward parameters, overtrading-control
sweeps (`flat_bonus` / `txn_cost_frac`), continuous position sizing
(`Box(-1,1)` action), and training across many synthetic regimes with held-out
real data for generalization testing.
