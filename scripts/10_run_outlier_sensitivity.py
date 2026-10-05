"""
Parameter sweep for the outlier sensitivity analysis (Supplementary
Materials S7.1). Varies one dimension at a time around the baseline
(12 iterations, eps = 1..12, MinPts = 10):
    - number of iterations: 8, 10, 12, 14, 16 (eps = 1..n)
    - eps schedule shape:   ramp 1-8, ramp 2-14, ramp 1-12 (12 iterations)
    - MinPts:               5, 8, 10, 12, 15

Runs 09_outlier_frequency.py once per combination; existing outputs are
skipped, so the sweep can be resumed.

Usage:
    python scripts/10_run_outlier_sensitivity.py [--period PRE POST]

Outputs: outputs/outliers/sensitivity/<PERIOD>_<dimension>=<value>_mp<MinPts>.csv
"""
import argparse
import subprocess
import sys
from pathlib import Path

import config

SCRIPT = Path(__file__).resolve().parent / "09_outlier_frequency.py"

BASELINE_SCHEDULE = ",".join(str(i) for i in range(1, 13))
BASELINE_MIN_PTS = 10

variations = []
for n_iter in [8, 10, 12, 14, 16]:
    sched = ",".join(str(i) for i in range(1, n_iter + 1))
    variations.append(("iters", n_iter, sched, BASELINE_MIN_PTS))

schedules_named = {
    "ramp1-8": ",".join(str(int(round(1 + 7 * i / 11))) for i in range(12)),
    "ramp2-14": ",".join(str(int(round(2 + 12 * i / 11))) for i in range(12)),
    "ramp1-12": BASELINE_SCHEDULE,
}
for name, sched in schedules_named.items():
    variations.append(("sched", name, sched, BASELINE_MIN_PTS))

for mp in [5, 8, 10, 12, 15]:
    variations.append(("mp", mp, BASELINE_SCHEDULE, mp))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", nargs="+", choices=["PRE", "POST"],
                    default=["PRE", "POST"])
    args = ap.parse_args()

    out_dir = config.ensure_dir(config.SENSITIVITY_DIR)
    for period in args.period:
        for dim, tag, sched, mp in variations:
            outfile = out_dir / f"{period}_{dim}={tag}_mp{mp}.csv"
            if outfile.exists() and outfile.stat().st_size > 100:
                print(f"Skipping {outfile.name} (already done)")
                continue
            cmd = [sys.executable, str(SCRIPT), "--period", period,
                   "--schedule", sched, "--min-pts", str(mp),
                   "--out", str(outfile)]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)
