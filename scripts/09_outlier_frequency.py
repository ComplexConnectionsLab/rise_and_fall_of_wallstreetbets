"""
Outlier frequency for one parameter combination (Methods, step 2, and the
sensitivity analysis in Supplementary Materials S7.1).

Runs iterative DBSCAN on every week of a period and counts, for each user,
the number of weeks in which they are still classified as noise after the
last iteration. Persistent outliers are the users flagged in more than
15 (PRE) or 20 (POST) weeks.

Usage:
    python scripts/09_outlier_frequency.py --period PRE \
        --schedule 1,2,3,4,5,6,7,8,9,10,11,12 --min-pts 10 \
        --out outputs/outliers/sensitivity/PRE_iters=12_mp10.csv

Output columns: author, n_weeks_flagged
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

import config
from common import iterative_dbscan_noise, load_week


def main(period: str, eps_schedule: list, min_pts: int, out_path: Path):
    freq = Counter()
    for i in range(config.N_WEEKS[period]):
        # As in the original analysis, ticker columns are dropped only for PRE
        features = load_week(period, i, drop_tickers=(period == "PRE"))
        for author in iterative_dbscan_noise(features, eps_schedule, min_pts):
            freq[author] += 1
    out = pd.DataFrame(list(freq.items()), columns=["author", "n_weeks_flagged"])
    out = out.sort_values("n_weeks_flagged", ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} authors to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", required=True, choices=["PRE", "POST"])
    ap.add_argument("--schedule",
                    default=",".join(map(str, config.OUTLIER_EPS_SCHEDULE)),
                    help="Comma-separated eps values, one per iteration")
    ap.add_argument("--min-pts", type=int, default=config.OUTLIER_MIN_PTS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    schedule = [float(x) for x in args.schedule.split(",")]
    main(args.period, schedule, args.min_pts, args.out)
