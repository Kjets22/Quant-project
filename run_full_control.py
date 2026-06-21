"""
run_full_control.py — Phase A: full-rigor control via ticker-level parallelism.

The benchmark showed single-env CPU is the fastest per-process configuration
(parallel SubprocVecEnv and GPU add overhead for this tiny MLP + cheap env). So
the throughput win is to run the 6 tickers as INDEPENDENT processes in parallel,
each doing its own 6-fold x 500k walk-forward on one CPU core's rollout loop.

Each ticker writes runs/control_<TICKER>.csv; we then merge into
runs/control_baseline.csv and print the pooled summary.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from basket import BASKET_TICKERS
from control import CONTROL_CSV, RUNS_DIR, _summary

LOG_DIR = RUNS_DIR / "logs"


def launch(tickers: list[str], folds: int, timesteps: int,
           threads_per_proc: int = 4) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent

    procs = []
    for t in tickers:
        env = os.environ.copy()
        # Limit intra-op threads so N parallel processes don't oversubscribe.
        env["OMP_NUM_THREADS"] = str(threads_per_proc)
        env["MKL_NUM_THREADS"] = str(threads_per_proc)
        env["CT_TORCH_THREADS"] = str(threads_per_proc)
        out_csv = RUNS_DIR / f"control_{t}.csv"
        log_path = LOG_DIR / f"control_{t}.log"
        cmd = [sys.executable, "control.py",
               "--tickers", t, "--folds", str(folds),
               "--timesteps", str(timesteps), "--out", str(out_csv)]
        log_f = open(log_path, "w", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=here, env=env, stdout=log_f,
                             stderr=subprocess.STDOUT)
        procs.append((t, p, log_f, out_csv))
        print(f"launched {t} (pid {p.pid}) -> {log_path.name}")

    print(f"\nwaiting for {len(procs)} ticker processes ...")
    t0 = time.time()
    results = {}
    for t, p, log_f, out_csv in procs:
        rc = p.wait()
        log_f.close()
        results[t] = rc
        print(f"  {t}: exit {rc}  ({(time.time()-t0)/60:.1f} min elapsed)")

    # Merge per-ticker CSVs.
    frames = []
    for t, _p, _f, out_csv in procs:
        if out_csv.exists():
            frames.append(pd.read_csv(out_csv))
        else:
            print(f"  WARNING: no output for {t} (check {LOG_DIR / f'control_{t}.log'})")
    if not frames:
        print("No results produced. See logs in", LOG_DIR)
        return
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(CONTROL_CSV, index=False)
    _summary(merged, CONTROL_CSV)
    print(f"\ntotal wall time: {(time.time()-t0)/60:.1f} min")


def merge_only(tickers: list[str]) -> None:
    """Merge already-written per-ticker CSVs into the baseline + print summary."""
    frames = []
    missing = []
    for t in tickers:
        out_csv = RUNS_DIR / f"control_{t}.csv"
        if out_csv.exists():
            frames.append(pd.read_csv(out_csv))
        else:
            missing.append(t)
    if missing:
        print(f"still missing: {missing}")
    if not frames:
        print("no per-ticker CSVs yet.")
        return
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(CONTROL_CSV, index=False)
    _summary(merged, CONTROL_CSV)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Full-rigor parallel control baseline")
    p.add_argument("--tickers", type=str, default=None)
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--threads-per-proc", type=int, default=4)
    p.add_argument("--merge-only", action="store_true",
                   help="skip launching; just merge existing per-ticker CSVs")
    args = p.parse_args()
    tickers = ([s.strip().upper() for s in args.tickers.split(",")]
               if args.tickers else BASKET_TICKERS)
    if args.merge_only:
        merge_only(tickers)
    else:
        launch(tickers, args.folds, args.timesteps, args.threads_per_proc)
