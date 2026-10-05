"""
Figure 2A: t-SNE projections of the DBSCAN clusters for the six
representative weeks.

DBSCAN (eps=1, min_samples=15) is run on the same user samples used for
spectral clustering, so that the two methods can be compared. The three
largest clusters are coloured green (commenters), pink (active) and orange
(posters); noise is grey. Cluster labels are renumbered by size (0 = largest)
so that downstream scripts can refer to the top three clusters as 0, 1, 2.

Requires: 04_spectral_representative_weeks.py
Outputs:  outputs/clustering/representative_weeks/dbscan/features_sample_week_<week>.csv
          outputs/figures/dbscan_tsne/dbscan_clustering_no-axis_week_<week>.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

import config
from common import pca_project, relabel_by_size, set_style

EPS = 1
MIN_SAMPLES = 15

# First three: commenters, active, posters; then smaller clusters
COLORS = config.CLUSTER_COLORS + ['#ADA1CE', '#388D72', '#00FFF2', '#F2FF00',
                                  '#A600FF', '#F6DBFF', '#77685D']
GREY = (0.5, 0.5, 0.5)

set_style()

spectral_dir = config.REPRESENTATIVE_DIR / "spectral"
out_dir = config.ensure_dir(config.REPRESENTATIVE_DIR / "dbscan")
fig_dir = config.ensure_dir(config.FIGURES_DIR / "dbscan_tsne")

for i, week_label in zip(config.REPRESENTATIVE_WEEKS, config.REPRESENTATIVE_LABELS):
    features = pd.read_csv(spectral_dir / f"features_sample_final_{i}.csv", index_col=0)
    features = features.drop(columns=["labels"])

    X = pca_project(features)
    projection = TSNE().fit_transform(X)
    labels = relabel_by_size(DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X).labels_)

    features["labels"] = labels
    features.to_csv(out_dir / f"features_sample_week_{i}.csv")

    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    colors = [COLORS[x % len(COLORS)] if x >= 0 else GREY for x in labels]

    fig, ax = plt.subplots(facecolor='none', figsize=(10, 10))
    ax.scatter(*projection.T, s=50, linewidth=0, c=colors, alpha=0.25)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(fig_dir / f"dbscan_clustering_no-axis_week_{i}.png",
                dpi=500, transparent=True)
    plt.close(fig)

    print(f'{week_label}, clusters: {n_clusters}, '
          f'noise: {n_noise / len(features) * 100:.2f}%')
