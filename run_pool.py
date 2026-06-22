"""
run_pool.py — fold-level parallel runner for the Phase-3 A/B evaluations.

Speeds up the A/Bs by running independent (tag, ticker, fold) jobs across many
processes to fill idle CPU cores (the per-ticker scheme left ~half the cores
idle). Fully resumable: completed jobs (and migrated legacy per-ticker
checkpoints) are skipped. When all jobs finish it merges per-tag baselines and
writes the A/B verdicts vs the control.
"""

from __future__ import annotations

import datetime
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
from run_job import job_path

HERE = Path(__file__).parent
RUNS = HERE / "runs"
PHASE3 = RUNS / "phase3"
JOBS_DIR = PHASE3 / "jobs"
LOG = PHASE3 / "pool.log"
PY = sys.executable
K = 6
TIMESTEPS = 500_000
MAX_PARALLEL = 16           # push more of the 28 logical cores (single-core rollouts)
THREADS_PER_PROC = 2

RISK_JSON = ('{"use_diff_sharpe": true, "diff_sharpe_w": 0.3, '
             '"dd_penalty_w": 0.05, "vol_penalty_w": 0.02}')

CONFIGS = [
    {"tag": "alpha", "extra": ["--alphatrend"], "phase": "AlphaTrend",
     "label": "alphatrend"},
    {"tag": "regimeC", "extra": ["--regime"], "phase": "C",
     "label": "regime"},
]


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def migrate_legacy() -> None:
    """Split any legacy per-ticker CSV (tag_TICKER.csv) into per-fold job files."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in CONFIGS:
        for t in BASKET_TICKERS:
            legacy = PHASE3 / f"{cfg['tag']}_{t}.csv"
            if not legacy.exists():
                continue
            d = pd.read_csv(legacy)
            for _, r in d.iterrows():
                jp = job_path(cfg["tag"], t, int(r["fold"]))
                if not jp.exists():
                    pd.DataFrame([r.to_dict()]).to_csv(jp, index=False)


def pending_jobs() -> list[dict]:
    jobs = []
    for cfg in CONFIGS:
        for t in BASKET_TICKERS:
            for f in range(1, K + 1):
                if not job_path(cfg["tag"], t, f).exists():
                    jobs.append({"tag": cfg["tag"], "ticker": t, "fold": f,
                                 "extra": cfg["extra"], "phase": cfg["phase"]})
    return jobs


def run_pool(jobs: list[dict]) -> None:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(THREADS_PER_PROC)
    env["MKL_NUM_THREADS"] = str(THREADS_PER_PROC)
    env["CT_TORCH_THREADS"] = str(THREADS_PER_PROC)
    env["PYTHONWARNINGS"] = "ignore"

    running: list[tuple[dict, subprocess.Popen]] = []
    queue = list(jobs)
    total = len(jobs)
    done = 0
    t0 = time.time()

    while queue or running:
        while queue and len(running) < MAX_PARALLEL:
            j = queue.pop(0)
            cmd = [PY, "-u", "run_job.py", "--tag", j["tag"], "--ticker", j["ticker"],
                   "--fold", str(j["fold"]), "--timesteps", str(TIMESTEPS),
                   "--phase", j["phase"], *j["extra"]]
            p = subprocess.Popen(cmd, cwd=str(HERE), env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            running.append((j, p))
        # Reap finished.
        still = []
        for j, p in running:
            if p.poll() is None:
                still.append((j, p))
            else:
                done += 1
                log(f"[{done}/{total}] finished {j['tag']}/{j['ticker']}/f{j['fold']} "
                    f"rc={p.returncode}  ({(time.time()-t0)/60:.1f} min)")
        running = still
        time.sleep(3)


def merge_and_verdict() -> None:
    for cfg in CONFIGS:
        tag = cfg["tag"]
        files = sorted(JOBS_DIR.glob(f"{tag}__*.csv"))
        if not files:
            continue
        merged = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        base = PHASE3 / f"{tag}_baseline.csv"
        merged.to_csv(base, index=False)
        log(f"merged {len(files)} job files -> {base.name}")
        vtxt = PHASE3 / f"{tag}_verdict.txt"
        with open(vtxt, "w", encoding="utf-8") as f:
            subprocess.run([PY, "-u", "ab_compare.py", "--variant",
                            str(base.relative_to(HERE)), "--label", cfg["label"]],
                           cwd=str(HERE), stdout=f, stderr=subprocess.STDOUT)
        log(f"verdict -> {vtxt.name}")


def main() -> None:
    log(f"=== fold-pool started (MAX_PARALLEL={MAX_PARALLEL}) ===")
    migrate_legacy()
    jobs = pending_jobs()
    log(f"{len(jobs)} pending jobs (of {len(CONFIGS) * len(BASKET_TICKERS) * K} total)")
    run_pool(jobs)
    merge_and_verdict()
    (PHASE3 / "PHASE3_DONE.txt").write_text(
        datetime.datetime.now().isoformat(), encoding="utf-8")
    log("=== fold-pool DONE ===")


if __name__ == "__main__":
    main()
