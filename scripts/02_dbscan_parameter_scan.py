"""
DBSCAN parameter selection (Supplementary Materials S9, Figure S11).

For every (eps, min_samples) pair, every week and N_REALIZATIONS random
samples, record the number of noise points and the Davies-Bouldin index
(computed on non-noise points).

Usage:
    python scripts/02_dbscan_parameter_scan.py                  # PRE, full 3x3 grid
    python scripts/02_dbscan_parameter_scan.py --period POST    # Figure S11B
    python scripts/02_dbscan_parameter_scan.py --eps 1 --min-samples 5 10 15

Outputs: outputs/clustering/dbscan_parameter_scan/<period>/eps_<eps>_minpts_<m>/
    noisepoints.csv, index.csv   (rows = weeks, columns = realizations)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN
from tqdm import tqdm

import config
from common import load_week, pca_project

EPS_GRID = [0.5, 1, 1.5]
MIN_SAMPLES_GRID = [5, 10, 15]


def scan_dir(period, eps, min_samples):
    return config.DBSCAN_SCAN_DIR / period.lower() / f"eps_{eps:g}_minpts_{min_samples}"


def run(period, eps, min_samples, rng):
    weeks = list(range(config.N_WEEKS[period]))
    cols = [str(j) for j in range(config.N_REALIZATIONS)]
    n_noise = pd.DataFrame(0, index=weeks, columns=cols)
    db_index = pd.DataFrame(np.nan, index=weeks, columns=cols)

    for i in tqdm(weeks, desc=f"{period} eps={eps:g}, min_samples={min_samples}"):
        week = load_week(period, i, drop_tickers=(period == "PRE"))
        for j in cols:
            features = week.sample(config.SAMPLE_SIZE, random_state=rng)
            features = features.reset_index(drop=True)
            X = pca_project(features)
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_

            n_noise.loc[i, j] = int((labels == -1).sum())

            in_cluster = labels != -1
            try:
                db_index.loc[i, j] = metrics.davies_bouldin_score(
                    X[in_cluster], labels[in_cluster])
            except ValueError:  # fewer than 2 clusters
                db_index.loc[i, j] = np.nan

    out = config.ensure_dir(scan_dir(period, eps, min_samples))
    n_noise.to_csv(out / "noisepoints.csv")
    db_index.to_csv(out / "index.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period", choices=["PRE", "POST"], default="PRE")
    ap.add_argument("--eps", type=float, nargs="+", default=EPS_GRID)
    ap.add_argument("--min-samples", type=int, nargs="+", default=MIN_SAMPLES_GRID)
    args = ap.parse_args()

    rng = np.random.default_rng(config.SEED)
    for eps in args.eps:
        for m in args.min_samples:
            run(args.period, eps, m, rng)
    print(f"Saved results to {config.DBSCAN_SCAN_DIR}")
