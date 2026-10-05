"""
Figure 2C: feature profiles of the three largest DBSCAN clusters
(commenters, active, posters) in the week of 24-30 January 2021
(mean +/- std).

Requires: 05_dbscan_tsne.py
Output:   outputs/figures/avg_features_clusters_pointplot.pdf
"""
import matplotlib
matplotlib.use("Agg")
import pandas as pd

import config
from common import cluster_feature_pointplot, set_style

set_style()

WEEK = config.FOCUS_WEEK
fig_dir = config.ensure_dir(config.FIGURES_DIR)

dbs = pd.read_csv(config.REPRESENTATIVE_DIR / "dbscan" / f"features_sample_week_{WEEK}.csv", index_col=0)
top3 = dbs[dbs.labels.isin([0, 1, 2])]  # labels are ordered by cluster size

cluster_feature_pointplot(top3, fig_dir / "avg_features_clusters_pointplot.pdf")

print(f"Saved: {fig_dir / 'avg_features_clusters_pointplot.pdf'}")
