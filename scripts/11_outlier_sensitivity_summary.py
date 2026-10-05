"""
Figure S8: opinion-leader persistence across the outlier sensitivity
analysis (Supplementary Materials S7.1).

Reads all files produced by 10_run_outlier_sensitivity.py and reports, for
every configuration and a range of persistence thresholds, whether each of
the three opinion leaders is still a persistent outlier
(n_weeks_flagged >= threshold).

Requires: 10_run_outlier_sensitivity.py, leaders.local.json (see README)
Outputs:  outputs/outliers/stage2_survival_long.csv
          outputs/outliers/stage2_survival_wide.csv
          outputs/figures/stage2_survival_heatmap.png
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from common import set_style

set_style()

LEADERS = config.load_leaders()  # {label: username}
LABELS = {label: f"{label} ({config.LEADER_TICKERS[label]})" for label in LEADERS}

# Thresholds to sweep around the baselines (15 PRE, 20 POST)
THRESHOLDS = {"PRE": [10, 15, 20, 25, 30], "POST": [10, 15, 20, 25, 30]}


def parse_filename(name: str):
    """Parse e.g. 'PRE_iters=12_mp10.csv' into its parameters."""
    m = re.match(r"(PRE|POST)_(\w+)=([\w\-]+)_mp(\d+)\.csv$", name)
    if not m:
        return None
    period, dim, val, mp = m.groups()
    return {"period": period, "dimension": dim, "value": val, "mp": int(mp)}


def load_all_files() -> pd.DataFrame:
    """One row per (file, opinion leader) with the leader's n_weeks_flagged."""
    rows = []
    for path in sorted(config.SENSITIVITY_DIR.glob("*.csv")):
        meta = parse_filename(path.name)
        if meta is None:
            print(f"Skipping unparseable filename: {path.name}")
            continue
        df = pd.read_csv(path)
        counts = dict(zip(df["author"], df["n_weeks_flagged"]))
        for label, username in LEADERS.items():
            rows.append({**meta, "file": path.name, "leader_label": LABELS[label],
                         "n_weeks_flagged": int(counts.get(username, 0))})
    return pd.DataFrame(rows)


def survival_table(long_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in long_df.iterrows():
        for thr in THRESHOLDS[row["period"]]:
            out.append({**row.to_dict(), "threshold": thr,
                        "survives": row["n_weeks_flagged"] >= thr})
    return pd.DataFrame(out)


def survival_heatmap(surv: pd.DataFrame, period: str, ax):
    """Rows = configurations, columns = (leader x threshold); cell colour =
    n_weeks_flagged, annotation = survives or not."""
    sub = surv[surv["period"] == period].copy()
    if sub.empty:
        ax.set_axis_off()
        return
    sub["config"] = sub["dimension"] + "=" + sub["value"] + " mp" + sub["mp"].astype(str)
    sub["col"] = sub["leader_label"] + "\nthr=" + sub["threshold"].astype(str)
    val_grid = sub.pivot_table(index="config", columns="col",
                               values="n_weeks_flagged", aggfunc="first")
    surv_grid = sub.pivot_table(index="config", columns="col",
                                values="survives", aggfunc="first")

    im = ax.imshow(val_grid.values, aspect="auto", cmap="viridis",
                   interpolation="nearest", rasterized=True)
    ax.set_xticks(range(val_grid.shape[1]))
    ax.set_xticklabels(val_grid.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(val_grid.shape[0]))
    ax.set_yticklabels(val_grid.index, fontsize=8)
    ax.set_title(f"Period {period}: opinion-leader persistence across configurations\n"
                 f"cell = n_weeks_flagged, marker = survives threshold")
    plt.colorbar(im, ax=ax, label="n_weeks_flagged")
    vmax = np.nanmax(val_grid.values)
    for i in range(val_grid.shape[0]):
        for j in range(val_grid.shape[1]):
            v = val_grid.values[i, j]
            marker = "✓" if surv_grid.values[i, j] else "✗"
            color = "white" if v < vmax / 2 else "black"
            ax.text(j, i, f"{int(v)}\n{marker}", ha="center", va="center",
                    fontsize=7, color=color)


long_df = load_all_files()
print(f"Loaded {long_df['file'].nunique()} files, {len(long_df)} (file x leader) rows\n")

surv = survival_table(long_df)
out_dir = config.ensure_dir(config.OUTLIERS_DIR)
surv.to_csv(out_dir / "stage2_survival_long.csv", index=False)
surv.pivot_table(index=["period", "dimension", "value", "mp"],
                 columns=["leader_label", "threshold"],
                 values="survives", aggfunc="first").to_csv(out_dir / "stage2_survival_wide.csv")

for period in ["PRE", "POST"]:
    for label in LABELS.values():
        s = surv[(surv.period == period) & (surv.leader_label == label)]
        print(f"{period:4s}  {label:15s}: survives {int(s['survives'].sum())}/{len(s)} "
              f"(config x threshold) cells")

fig, axes = plt.subplots(2, 1, figsize=(13, 12))
survival_heatmap(surv, "PRE", axes[0])
survival_heatmap(surv, "POST", axes[1])
fig.tight_layout()
out = config.ensure_dir(config.FIGURES_DIR) / "stage2_survival_heatmap.png"
fig.savefig(out, bbox_inches="tight", dpi=500)
print(f"Saved: {out}")
