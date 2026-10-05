"""
Spectral clustering of one random sample of users for each representative
week (input to the ARI comparison, Supplementary Materials S12). The same samples are then clustered with DBSCAN
(05_dbscan_tsne.py) so that the two methods can be compared user by user.

Clusters are the connected components (>= MIN_COMPONENT_SIZE nodes) of the
graph with affinity S = exp(-D^2), thresholded at THRESHOLD. Label 0 is the
largest component; noise is labelled -1.

Output: outputs/clustering/representative_weeks/spectral/features_sample_final_<week>.csv
        (sampled users + spectral `labels`)
"""
import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances

import config
from common import load_week, pca_project

THRESHOLD = config.SPECTRAL_THRESHOLD
MIN_COMPONENT_SIZE = config.SPECTRAL_MIN_COMPONENT_SIZE

out = config.ensure_dir(config.REPRESENTATIVE_DIR / "spectral")
rng = np.random.default_rng(config.SEED)

for i in config.REPRESENTATIVE_WEEKS:
    features = load_week("PRE", i).sample(config.SAMPLE_SIZE, random_state=rng)
    features = features.reset_index(drop=True)
    X = pca_project(features)

    s = np.exp(-pairwise_distances(X) ** 2)
    np.fill_diagonal(s, 0)
    s[s < THRESHOLD] = 0
    g = nx.from_numpy_array(s)

    components = sorted(nx.connected_components(g), key=len, reverse=True)
    noise = set(nx.isolates(g))
    labels = np.full(len(features), -1)
    k = 0
    for comp in components:
        if len(comp) < MIN_COMPONENT_SIZE:
            noise |= comp
        else:
            labels[list(comp)] = k
            k += 1

    features["labels"] = labels
    features.to_csv(out / f"features_sample_final_{i}.csv")

print(f"Saved results to {out}")
