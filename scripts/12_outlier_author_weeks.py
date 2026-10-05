"""
Baseline iterative-DBSCAN run (eps = 1..12, MinPts = 10) that saves every
(author, week) pair in which a user is a final outlier. Used for the
calendar-day persistence analysis (13_leader_calendar_persistence.py).

Outputs: outputs/outliers/PRE_baseline_pairs.csv
         outputs/outliers/POST_baseline_pairs.csv
"""
from pathlib import Path

import pandas as pd

import config
from common import iterative_dbscan_noise, load_week


def main(period: str, out_path: Path):
    rows = []
    for i in range(config.N_WEEKS[period]):
        features = load_week(period, i, drop_tickers=(period == "PRE"))
        for author in iterative_dbscan_noise(features, config.OUTLIER_EPS_SCHEDULE,
                                             config.OUTLIER_MIN_PTS):
            rows.append({"author": author, "week": i})
    out = pd.DataFrame(rows, columns=["author", "week"])
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} (author, week) pairs to {out_path}")


if __name__ == "__main__":
    out_dir = config.ensure_dir(config.OUTLIERS_DIR)
    for period in ["PRE", "POST"]:
        main(period, out_dir / f"{period}_baseline_pairs.csv")
