"""bench.py — measure PPO training throughput across device / parallel-env configs."""
from __future__ import annotations

import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch

from agent import train
from basket import ticker_cfg
from lockbox import Lockbox, load_dev_ticker

STEPS = 20_000


def run(label, df, device, n_envs):
    cfg = ticker_cfg("SPY")
    cfg.train.total_timesteps = STEPS
    cfg.train.device = device
    cfg.train.n_envs = n_envs
    t0 = time.time()
    train(df, cfg, verbose=0)
    dt = time.time() - t0
    print(f"{label:28s} | {STEPS/dt:8.0f} steps/s | {dt:6.1f}s")
    return dt


if __name__ == "__main__":
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
    lb = Lockbox.load_or_build()
    dev = load_dev_ticker("SPY", lb)
    # Use a modest slice so env construction isn't the timed bottleneck.
    df = dev.iloc[:40000].reset_index(drop=True)
    print(f"benchmark slice: {len(df)} rows, {STEPS} steps each\n")
    print(f"{'config':28s} | {'throughput':>10} | {'wall':>6}")
    print("-" * 52)
    run("CPU  n_envs=1", df, "cpu", 1)
    if torch.cuda.is_available():
        run("CUDA n_envs=1", df, "cuda", 1)
    run("CPU  n_envs=8", df, "cpu", 8)
    run("CPU  n_envs=16", df, "cpu", 16)
    if torch.cuda.is_available():
        run("CUDA n_envs=16", df, "cuda", 16)
