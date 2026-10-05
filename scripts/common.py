"""Helpers shared by several analysis scripts."""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import config


def load_week(period: str, week: int, drop_tickers: bool = True) -> pd.DataFrame:
    """Read the feature file of one week.

    Ticker columns (not used for clustering) are dropped by default, as is a
    stale `labels` column if one is present.
    """
    df = pd.read_csv(config.week_file(period, week), index_col=0)
    if drop_tickers:
        df = df.drop(columns=config.TICKER_COLS, errors="ignore")
    if "labels" in df.columns:
        df = df.drop(columns=["labels"])
    return df


def pca_project(features: pd.DataFrame,
                n_components: int = config.N_PCA_COMPONENTS) -> np.ndarray:
    """PCA projection of all feature columns (everything except `author`)."""
    X = features.drop(columns=["author"], errors="ignore")
    return PCA(n_components=n_components).fit_transform(X)


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Relabel clusters so that 0 is the largest, 1 the second largest, ...

    Noise (-1) is left unchanged.
    """
    labels = np.asarray(labels)
    counts = pd.Series(labels[labels != -1]).value_counts()
    mapping = {old: new for new, old in enumerate(counts.index)}
    mapping[-1] = -1
    return np.array([mapping[l] for l in labels])


def iterative_dbscan_noise(features: pd.DataFrame,
                           eps_schedule: list,
                           min_pts: int) -> set:
    """Iterative DBSCAN with cluster removal (Methods, step 1).

    At each iteration, users assigned to a cluster are removed and DBSCAN is
    re-run (with the next eps of the schedule) on the remaining noise points.
    Returns the set of authors still classified as noise after the last
    iteration.

    features: one week of standardized features with an `author` column and
              ticker columns already dropped.
    """
    df = features.copy().fillna(0)
    for eps in eps_schedule:
        if len(df) < min_pts:
            return set(df["author"])
        X = df.drop(columns=["author"])
        n_comp = min(config.N_PCA_COMPONENTS, X.shape[1], len(X))
        X_pca = PCA(n_components=n_comp).fit_transform(X)
        labels = DBSCAN(eps=eps, min_samples=min_pts, n_jobs=1).fit_predict(X_pca)
        df = df.loc[labels == -1].reset_index(drop=True)
        del X, X_pca, labels
        gc.collect()
        if df.empty:
            return set()
    return set(df["author"])


def set_style() -> None:
    """Apply the seaborn 'talk' style (renamed in matplotlib >= 3.6)."""
    import matplotlib.pyplot as plt
    try:
        plt.style.use("seaborn-v0_8-talk")
    except OSError:
        plt.style.use("seaborn-talk")


def cluster_feature_pointplot(df: pd.DataFrame, out_path) -> None:
    """Mean +/- std of each feature for the three largest clusters
    (commenters, active, posters), as in Figure 2C.

    df: one row per user, with a `labels` column in {0, 1, 2} and the
        feature columns listed in config.PLOT_FEATURES.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    tot = pd.melt(df[config.PLOT_FEATURES + ["labels"]], "labels",
                  var_name="Features")
    tot["labels"] = tot["labels"].astype(int)

    plt.rcParams['lines.solid_capstyle'] = 'round'
    fig, ax = plt.subplots(figsize=(22, 8))
    ax.axhline(0, color='black', linewidth=1, ls='--')
    sns.pointplot(data=tot, x="Features", y="value", hue="labels",
                  order=config.PLOT_FEATURES, hue_order=[0, 1, 2],
                  linestyle="none", dodge=.8 - .8 / 6,
                  palette=config.CLUSTER_COLORS, markers="o", markersize=8,
                  errorbar='sd', err_kws={"linewidth": 3}, capsize=0, ax=ax)
    ax.set_yscale('symlog')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(width=2)
    y_ticks = np.append(-10, np.append(ax.get_yticks(), 1))
    ax.set_yticks(np.unique(y_ticks))
    ax.set_xlabel('Features', fontsize=20)
    ax.set_ylabel('Standardized values', fontsize=20)
    ax.tick_params(axis='x', labelsize=20, rotation=45)
    ax.tick_params(axis='y', labelsize=20)
    ax.yaxis.grid(True, alpha=0.5, linewidth=0.8)
    handles = ax.get_legend().legend_handles
    ax.legend(handles=handles, labels=config.CLUSTER_NAMES, fontsize=20,
              frameon=False, ncol=3, loc=(0, 1.02))
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
# Ticker mentions (opinion-leader analyses)
# --------------------------------------------------------------------------
def weekly_ticker_totals(tickers) -> dict:
    """Community-wide ticker mentions per week (pre-squeeze period).

    Returns {ticker: array over weeks} for each requested ticker (lower case)
    plus "all": the total mentions of all tickers in that week.
    """
    n = config.N_WEEKS["PRE"]
    totals = {t: np.zeros(n) for t in list(tickers) + ["all"]}
    for i in range(n):
        path = config.TICKER_COUNTS_DIR / f"week_{i}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        week = df.select_dtypes(include=[np.number]).sum(axis=0)
        for t in tickers:
            totals[t][i] = week.get(t, 0)
        totals["all"][i] = week.sum()
    return totals


def load_ticker_activity(ticker: str) -> pd.DataFrame:
    """Posts + comments mentioning a ticker, as a long dataframe with columns
    date (daily), author, kind. Deleted accounts are dropped."""
    base = config.TICKER_TEXT_DIR / ticker
    posts = pd.read_csv(base / f"{ticker}_posts.csv", usecols=["author", "time"])
    comments = pd.read_csv(base / f"{ticker}_comments.csv", usecols=["author", "time"])
    posts["kind"] = "post"
    comments["kind"] = "comment"
    df = pd.concat([posts, comments], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "author"])
    df["date"] = df["time"].dt.floor("D")
    df = df[~df["author"].isin(["[deleted]", "[removed]"])]
    return df[["date", "author", "kind"]]


def load_ticker_posts(ticker: str) -> pd.DataFrame:
    """Posts mentioning a ticker (author, time)."""
    posts = pd.read_csv(config.TICKER_TEXT_DIR / ticker / f"{ticker}_posts.csv",
                        usecols=["author", "time"])
    posts["time"] = pd.to_datetime(posts["time"], errors="coerce")
    return posts.dropna(subset=["time"])


def daily_series(df: pd.DataFrame, leader_username: str) -> pd.DataFrame:
    """Daily mentions and daily unique users mentioning the ticker
    (unique users exclude the opinion leader). Missing days are zeros."""
    full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    mentions = df.groupby("date").size().reindex(full_idx, fill_value=0)
    others = df[df["author"] != leader_username]
    unique_users = others.groupby("date")["author"].nunique().reindex(full_idx, fill_value=0)
    out = pd.DataFrame({"mentions": mentions, "unique_users": unique_users})
    out.index.name = "date"
    return out
