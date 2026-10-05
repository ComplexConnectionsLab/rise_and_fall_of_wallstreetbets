"""
Figure S11: fraction of noise points and Davies-Bouldin index over time for
each (eps, min_samples) pair of the DBSCAN parameter scan. A 7-week rolling
mean is applied; shaded bands are +/- one std over samples.

Usage:
    python scripts/03_plot_dbscan_parameter_scan.py [--period PRE|POST]

Requires: 02_dbscan_parameter_scan.py (same period)
Output:   outputs/figures/DBSCAN_cluster_evaluation_<period>.pdf
"""
import argparse
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import config
from common import set_style

scan = importlib.import_module("02_dbscan_parameter_scan")

ap = argparse.ArgumentParser()
ap.add_argument("--period", choices=["PRE", "POST"], default="PRE")
period = ap.parse_args().period

set_style()

week = np.arange(config.N_WEEKS[period])
ticks = [0, 50, 100, 150]
start = config.PERIOD_START[period]
tick_labels = [f"{(start + pd.Timedelta(days=t)):%d/%m}-"
               f"{(start + pd.Timedelta(days=t + config.WINDOW_DAYS - 1)):%d/%m}"
               for t in ticks]
ROLL = 7

eps_values = scan.EPS_GRID
min_values = scan.MIN_SAMPLES_GRID
fig, ax = plt.subplots(len(eps_values), len(min_values), figsize=(15, 10),
                       sharex=True, sharey=True, squeeze=False)

for i, eps in enumerate(eps_values):
    for j, mins in enumerate(min_values):
        d = scan.scan_dir(period, eps, mins)
        noise = pd.read_csv(d / "noisepoints.csv", index_col=0) / config.SAMPLE_SIZE
        db_index = pd.read_csv(d / "index.csv", index_col=0)

        noise_mean = noise.mean(axis=1).rolling(ROLL).mean()
        noise_std = noise.std(axis=1).rolling(ROLL).mean()
        index_mean = db_index.mean(axis=1).rolling(ROLL).mean()
        index_std = db_index.std(axis=1).rolling(ROLL).mean()

        a = ax[i][j]
        a.plot(week, noise_mean, linewidth=4, c='blue', label='noise',
               solid_capstyle='round')
        a.fill_between(week, noise_mean - noise_std, noise_mean + noise_std,
                       alpha=0.2, edgecolor='blue', facecolor='blue')

        a2 = a.twinx()
        a2.plot(week, index_mean, linewidth=4, c='red', label='D-B index',
                solid_capstyle='round')
        a2.fill_between(week, index_mean - index_std, index_mean + index_std,
                        alpha=0.2, edgecolor='red', facecolor='red')
        a2.set_ylim(0.2, 1.5)

        if j < len(min_values) - 1:
            a2.yaxis.set_tick_params(labelright=False)
        else:
            a2.set_ylabel('D-B index', fontsize=20)
        if i == len(eps_values) - 1:
            a.set_xlabel('Week', fontsize=20)
            a.xaxis.set_major_locator(mticker.FixedLocator(ticks))
            a.set_xticklabels(tick_labels, rotation=45)
        if j == 0:
            a.set_ylabel('Fraction of noise', fontsize=20)

        a2.legend(loc='upper right', frameon=False)
        a.legend(loc='upper left', frameon=False)
        a.set_title(f'eps = {eps:g}, min_samples = {mins}')

plt.tight_layout()
out = config.ensure_dir(config.FIGURES_DIR) / f'DBSCAN_cluster_evaluation_{period}.pdf'
plt.savefig(out, dpi=300)
print(f"Saved: {out}")
