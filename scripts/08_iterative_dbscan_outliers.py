"""
Iterative DBSCAN outlier extraction (Methods, "Outliers with iterative
DBSCAN clustering", step 1), saving the intermediate results.

For each week, DBSCAN (min_samples = 10) is applied 12 times with
eps = 1, 2, ..., 12; after each iteration users assigned to a cluster are
removed and only the noise points are clustered again.

Usage:
    python scripts/08_iterative_dbscan_outliers.py [--period PRE|POST]

Output: outputs/outliers/iterative/<period>/week_<i>_eps_<eps>.csv
        users still classified as noise after the iteration with that eps.
"""
import argparse

import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm

import config
from common import load_week, pca_project

ap = argparse.ArgumentParser()
ap.add_argument("--period", choices=["PRE", "POST"], default="PRE")
args = ap.parse_args()

out = config.ensure_dir(config.OUTLIERS_DIR / "iterative" / args.period.lower())

for i in tqdm(range(config.N_WEEKS[args.period])):
    features = load_week(args.period, i, drop_tickers=(args.period == "PRE")).fillna(0)
    for eps in config.OUTLIER_EPS_SCHEDULE:
        if len(features) < config.OUTLIER_MIN_PTS:
            break
        labels = DBSCAN(eps=eps, min_samples=config.OUTLIER_MIN_PTS).fit(
            pca_project(features)).labels_
        features = features[labels == -1].reset_index(drop=True)
        features.to_csv(out / f"week_{i}_eps_{eps}.csv")
print(f"Saved results to {out}")
