"""
phase3_run_all.py — resumable orchestrator for the Phase-3 A/B evaluations.

Runs SEQUENTIALLY (never more than 6 workers at once, to bound thermal load):
  1. Enhancement-1 position-features A/B   -> runs/phase3/posfeat_baseline.csv
  2. Phase-B risk-aware reward A/B         -> runs/phase3/riskB_baseline.csv
Then writes an A/B verdict for each vs the control baseline.

Everything is resumable: each underlying run skips completed tickers/folds, so
if the machine crashes you just launch this script again and it continues from
the last checkpoint. Safe to re-run any number of times.
"""

from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "runs"
PHASE3 = RUNS / "phase3"
PHASE3.mkdir(parents=True, exist_ok=True)
LOG = PHASE3 / "runall.log"
PY = sys.executable

RISK_JSON = ('{"use_diff_sharpe": true, "diff_sharpe_w": 0.3, '
             '"dd_penalty_w": 0.05, "vol_penalty_w": 0.02}')


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str]) -> int:
    log("RUN: " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=str(HERE)).returncode


def verdict(variant_csv: str, label: str, out_txt: Path) -> None:
    with open(out_txt, "w", encoding="utf-8") as f:
        subprocess.run([PY, "-u", "ab_compare.py", "--variant", variant_csv,
                        "--label", label], cwd=str(HERE), stdout=f,
                       stderr=subprocess.STDOUT)
    log(f"verdict -> {out_txt.name}")


def main() -> None:
    log("=== Phase-3 orchestrator started (resumable, 6-way) ===")

    # 1) Enhancement-1 position features.
    log("--- Enhancement 1: position features A/B ---")
    run([PY, "-u", "run_full_control.py", "--tag", "posfeat",
         "--outdir", "runs/phase3", "--position-features", "--phase", "Enh1"])
    if (PHASE3 / "posfeat_baseline.csv").exists():
        verdict("runs/phase3/posfeat_baseline.csv", "position_features",
                PHASE3 / "enh1_verdict.txt")

    # 2) Phase-B risk-aware reward.
    log("--- Phase B: risk-aware reward A/B ---")
    run([PY, "-u", "run_full_control.py", "--tag", "riskB",
         "--outdir", "runs/phase3", "--reward-json", RISK_JSON, "--phase", "B"])
    if (PHASE3 / "riskB_baseline.csv").exists():
        verdict("runs/phase3/riskB_baseline.csv", "risk_reward",
                PHASE3 / "phaseB_verdict.txt")

    (PHASE3 / "PHASE3_DONE.txt").write_text(
        datetime.datetime.now().isoformat(), encoding="utf-8")
    log("=== Phase-3 orchestrator DONE ===")


if __name__ == "__main__":
    main()
