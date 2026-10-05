"""
Figure 2A (insets) and 2B: DBSCAN on repeated random samples of users, for
every week of the pre-squeeze period.

For each week and each of N_REALIZATIONS random samples of SAMPLE_SIZE users,
project the features on the top 10 PCA components, run DBSCAN and record the
number of clusters, the number of noise points and the cluster sizes.
The mean/std over samples give the number of clusters and fraction of noise
(Figure 2A) and the sizes of the three largest clusters over time (Figure 2B).

Outputs (outputs/clustering/dbscan_ensemble/):
    cluster_numbers.csv  rows = weeks, columns = realizations
    noisepoints.csv      rows = weeks, columns = realizations
    cluster_sizes.csv    rows = weeks, columns = realizations (list of sizes)
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm

import config
from common import load_week, pca_project

EPS = 1
MIN_SAMPLES = 15

weeks = range(config.N_WEEKS["PRE"])
cols = [str(j) for j in range(config.N_REALIZATIONS)]
rng = np.random.default_rng(config.SEED)

n_clusters = pd.DataFrame(0, index=list(weeks), columns=cols)
n_noise = pd.DataFrame(0, index=list(weeks), columns=cols)
sizes = pd.DataFrame(None, index=list(weeks), columns=cols, dtype=object)

for i in tqdm(weeks):
    week = load_week("PRE", i)
    for j in cols:
        features = week.sample(config.SAMPLE_SIZE, random_state=rng)
        features = features.reset_index(drop=True)

        X = pca_project(features)
        labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X).labels_

        n_clusters.loc[i, j] = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise.loc[i, j] = int((labels == -1).sum())
        count = pd.Series(labels).value_counts()
        sizes.at[i, j] = list(count[count.index != -1].values)

out = config.ensure_dir(config.DBSCAN_ENSEMBLE_DIR)
n_clusters.to_csv(out / "cluster_numbers.csv")
n_noise.to_csv(out / "noisepoints.csv")
sizes.to_csv(out / "cluster_sizes.csv")
print(f"Saved results to {out}")
