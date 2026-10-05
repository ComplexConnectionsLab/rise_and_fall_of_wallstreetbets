"""
Agreement between DBSCAN and spectral clustering on the same user samples:
adjusted Rand index for each representative week (Figure S13).

Requires: 04_spectral_representative_weeks.py, 05_dbscan_tsne.py
Output:   outputs/figures/ARI_vs_t.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score

import config
from common import set_style

set_style()

sc_dir = config.REPRESENTATIVE_DIR / "spectral"
dbs_dir = config.REPRESENTATIVE_DIR / "dbscan"

ari = []
for i in config.REPRESENTATIVE_WEEKS:
    dbs = pd.read_csv(dbs_dir / f"features_sample_week_{i}.csv", index_col=0)
    sc = pd.read_csv(sc_dir / f"features_sample_final_{i}.csv", index_col=0)
    ari.append(adjusted_rand_score(dbs.labels, sc.labels))

for label, value in zip(config.REPRESENTATIVE_LABELS, ari):
    print(f"{label}: ARI = {value:.3f}")

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(config.REPRESENTATIVE_LABELS, ari, linestyle='-', marker='o',
        markersize=15, linewidth=8, c='#5603AD')
ax.set_xlabel('Week', fontsize=20)
ax.set_ylabel('Adjusted Rand Index', fontsize=20)
ax.tick_params(axis='x', labelsize=15, rotation=45)
ax.tick_params(axis='y', labelsize=15, width=2)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_visible(True)
    ax.spines[axis].set_linewidth(2)
ax.set_ylim(0, 1)
plt.tight_layout()
out = config.ensure_dir(config.FIGURES_DIR) / "ARI_vs_t.png"
fig.savefig(out, dpi=300)
print(f"Saved: {out}")
