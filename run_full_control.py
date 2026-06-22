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
           threads_per_proc: int = 4, outdir: Path = RUNS_DIR, tag: str = "control",
           position_features: bool = False, phase: str = "A",
           reward_json: str | None = None) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent

    procs = []
    for t in tickers:
        out_csv = outdir / f"{tag}_{t}.csv"
        # Skip tickers already fully complete (resume-friendly).
        if out_csv.exists():
            try:
                ndone = int((pd.read_csv(out_csv)["fold"]).nunique())
            except Exception:  # noqa: BLE001
                ndone = 0
            if ndone >= folds:
                print(f"skip {t}: already complete ({ndone}/{folds} folds)")
                continue
            elif ndone > 0:
                print(f"resume {t}: {ndone}/{folds} folds done, continuing")
        env = os.environ.copy()
        # Limit intra-op threads so N parallel processes don't oversubscribe.
        env["OMP_NUM_THREADS"] = str(threads_per_proc)
        env["MKL_NUM_THREADS"] = str(threads_per_proc)
        env["CT_TORCH_THREADS"] = str(threads_per_proc)
        log_path = LOG_DIR / f"{tag}_{t}.log"
        cmd = [sys.executable, "-u", "control.py",
               "--tickers", t, "--folds", str(folds),
               "--timesteps", str(timesteps), "--out", str(out_csv),
               "--phase", phase]
        if position_features:
            cmd.append("--position-features")
        if reward_json:
            cmd += ["--reward-json", reward_json]
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

    merged_path = outdir / f"{tag}_baseline.csv"
    # Merge per-ticker CSVs.
    frames = []
    for t, _p, _f, out_csv in procs:
        if out_csv.exists():
            frames.append(pd.read_csv(out_csv))
        else:
            print(f"  WARNING: no output for {t} (check {LOG_DIR / f'{tag}_{t}.log'})")
    if not frames:
        print("No results produced. See logs in", LOG_DIR)
        return
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(merged_path, index=False)
    _summary(merged, merged_path)
    print(f"\ntotal wall time: {(time.time()-t0)/60:.1f} min")


def merge_only(tickers: list[str], outdir: Path = RUNS_DIR, tag: str = "control") -> None:
    """Merge already-written per-ticker CSVs into the baseline + print summary."""
    outdir = Path(outdir)
    merged_path = outdir / f"{tag}_baseline.csv"
    frames = []
    missing = []
    for t in tickers:
        out_csv = outdir / f"{tag}_{t}.csv"
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
    merged.to_csv(merged_path, index=False)
    _summary(merged, merged_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Full-rigor parallel control baseline")
    p.add_argument("--tickers", type=str, default=None)
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--threads-per-proc", type=int, default=4)
    p.add_argument("--merge-only", action="store_true",
                   help="skip launching; just merge existing per-ticker CSVs")
    p.add_argument("--outdir", type=str, default=str(RUNS_DIR))
    p.add_argument("--tag", type=str, default="control")
    p.add_argument("--position-features", action="store_true")
    p.add_argument("--reward-json", type=str, default=None,
                   help="JSON dict of RewardConfig overrides for the variant run")
    p.add_argument("--phase", type=str, default="A")
    args = p.parse_args()
    tickers = ([s.strip().upper() for s in args.tickers.split(",")]
               if args.tickers else BASKET_TICKERS)
    outdir = Path(args.outdir)
    if args.merge_only:
        merge_only(tickers, outdir=outdir, tag=args.tag)
    else:
        launch(tickers, args.folds, args.timesteps, args.threads_per_proc,
               outdir=outdir, tag=args.tag, position_features=args.position_features,
               phase=args.phase, reward_json=args.reward_json)
