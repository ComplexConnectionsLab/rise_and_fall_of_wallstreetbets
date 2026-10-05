"""
Shared configuration for all analysis scripts.

Every path is relative to two root folders, which can be overridden with
environment variables:

    WSB_DATA_DIR    input data   (default: <repo>/data)
    WSB_OUTPUT_DIR  results      (default: <repo>/outputs)

See data/README.md for the expected layout and file formats.

The Reddit usernames of the three opinion leaders are NOT stored in this
repository. To run the opinion-leader scripts on the original data, copy
`leaders.example.json` to `leaders.local.json` (git-ignored) and fill in the
usernames. Every output only ever uses the pseudonymous labels User1/2/3.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("WSB_DATA_DIR", REPO_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("WSB_OUTPUT_DIR", REPO_ROOT / "outputs"))

# Optional local override of individual folders (git-ignored), e.g. to run
# on data stored elsewhere. Keys: features_pre, features_post, ticker_counts,
# ticker_text, raw_features, prices, output.
PATHS_FILE = REPO_ROOT / "paths.local.json"
_local = json.loads(PATHS_FILE.read_text()) if PATHS_FILE.exists() else {}


def _path(key: str, default: Path) -> Path:
    return Path(_local[key]) if key in _local else default


OUTPUT_DIR = _path("output", OUTPUT_DIR)

# Inputs ------------------------------------------------------------------
# Weekly standardized user features: one file per rolling week, week_<i>.csv
FEATURES_DIR = {
    "PRE": _path("features_pre", DATA_DIR / "features" / "pre"),    # Aug 2020 - Jan 2021
    "POST": _path("features_post", DATA_DIR / "features" / "post"),  # Feb 2021 - Jul 2021
}
# Weekly per-user ticker-mention counts (pre-squeeze period), week_<i>.csv
TICKER_COUNTS_DIR = _path("ticker_counts", DATA_DIR / "tickers" / "pre")
# Posts/comments mentioning each leader's ticker: <TICKER>/<TICKER>_posts.csv
TICKER_TEXT_DIR = _path("ticker_text", DATA_DIR / "ticker_text")
# Weekly non-standardized user features (counts), week_<i>.csv
RAW_FEATURES_DIR = _path("raw_features", DATA_DIR / "user_features_raw" / "pre")
# Daily closing prices: <NAME>.csv or <NAME>/<NAME>.csv, and SP500.csv
PRICES_DIR = _path("prices", DATA_DIR / "prices")


def price_file(name: str) -> Path:
    nested = PRICES_DIR / name / f"{name}.csv"
    return nested if nested.exists() else PRICES_DIR / f"{name}.csv"

# Outputs -----------------------------------------------------------------
CLUSTERING_DIR = OUTPUT_DIR / "clustering"
DBSCAN_ENSEMBLE_DIR = CLUSTERING_DIR / "dbscan_ensemble"
DBSCAN_SCAN_DIR = CLUSTERING_DIR / "dbscan_parameter_scan"
REPRESENTATIVE_DIR = CLUSTERING_DIR / "representative_weeks"
OUTLIERS_DIR = OUTPUT_DIR / "outliers"
SENSITIVITY_DIR = OUTLIERS_DIR / "sensitivity"
FIGURES_DIR = OUTPUT_DIR / "figures"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def week_file(period: str, week: int) -> Path:
    return FEATURES_DIR[period] / f"week_{week}.csv"


# --------------------------------------------------------------------------
# Time windows
# --------------------------------------------------------------------------
# Data are split into 7-day windows shifted by one day ("weeks").
# Week i covers the days [PERIOD_START + i, PERIOD_START + i + 6].
PERIOD_START = {
    "PRE": pd.Timestamp("2020-08-01"),
    "POST": pd.Timestamp("2021-02-01"),
}
WINDOW_DAYS = 7
SHIFT_DAYS = 1
N_WEEKS = {
    "PRE": int(os.environ.get("WSB_N_WEEKS_PRE", 178)),
    "POST": int(os.environ.get("WSB_N_WEEKS_POST", 176)),
}

# Representative weeks shown in Figure 2A (24-30 of each month, Aug-Jan)
REPRESENTATIVE_WEEKS = [23, 54, 84, 115, 145, 176]
REPRESENTATIVE_LABELS = ['24-30 Aug', '24-30 Sept', '24-30 Oct',
                         '24-30 Nov', '24-30 Dec', '24-30 Jan']
FOCUS_WEEK = 176  # 24-30 Jan 2021, used in Figure 2C

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
# The 16 user features (see Methods). Column names in the week_<i>.csv files.
FEATURES = [
    'num_comm', 'comm_score', 'comm_entropy', 'comm_jargon', 'comm_sent',
    'first_children',
    'num_post', 'post_score', 'post_entropy', 'post_jargon', 'post_sent',
    'post_comms',
    'k_out', 'k_in', 's_out', 's_in',
]
# Subset shown in the cluster-profile figures
PLOT_FEATURES = ['num_comm', 'comm_score', 'first_children', 'comm_entropy',
                 'num_post', 'post_score', 'post_comms', 'post_entropy',
                 'k_in', 'k_out']

# Per-ticker columns present in the PRE feature files; they are NOT used for
# clustering and are dropped before PCA.
TICKER_COLS = ['spy', 'amd', 'tsla', 'mu', 'aapl', 'amzn', 'msft', 'snap',
               'nvda', 'spce', 'fb', 'dis', 'bynd', 'nflx', 'jnug', 'ge',
               'rad', 'sq', 'atvi', 'uso', 'twtr', 'amc', 'bb', 'nok',
               'pltr', 'gme']

# --------------------------------------------------------------------------
# Clustering parameters
# --------------------------------------------------------------------------
N_PCA_COMPONENTS = 10
SAMPLE_SIZE = int(os.environ.get("WSB_SAMPLE_SIZE", 10_000))
N_REALIZATIONS = int(os.environ.get("WSB_N_REALIZATIONS", 100))
# Seed for the random subsamples. The original analysis used no fixed seed;
# set WSB_SEED to an integer to make runs reproducible.
SEED = int(os.environ["WSB_SEED"]) if os.environ.get("WSB_SEED") else None

# Spectral clustering: affinity S = exp(-D^2), entries below the threshold
# are set to 0; components smaller than MIN_COMPONENT_SIZE are noise
SPECTRAL_THRESHOLD = 0.5
SPECTRAL_MIN_COMPONENT_SIZE = 5

# Iterative DBSCAN for outliers (Methods: "Outliers with iterative DBSCAN")
OUTLIER_EPS_SCHEDULE = list(range(1, 13))  # eps = 1, 2, ..., 12
OUTLIER_MIN_PTS = 10
OUTLIER_THRESHOLD = {"PRE": 15, "POST": 20}  # min. number of weeks flagged

# Cluster colours (commenters, active, posters)
CLUSTER_COLORS = ['#04e756', '#de0074', '#f7b500']
CLUSTER_NAMES = ['Commenters', 'Active', 'Posters']

# --------------------------------------------------------------------------
# Opinion leaders
# --------------------------------------------------------------------------
LEADER_TICKERS = {"User1": "GME", "User2": "CRSR", "User3": "PLTR"}
LEADERS_FILE = REPO_ROOT / "leaders.local.json"


def load_leaders() -> dict:
    """Return {label: username} for the three opinion leaders.

    Reads leaders.local.json (git-ignored). If it does not exist, the
    placeholder names "User1", "User2", "User3" are used, which match the
    authors in the synthetic dataset.
    """
    if LEADERS_FILE.exists():
        with open(LEADERS_FILE) as fh:
            mapping = json.load(fh)
    else:
        mapping = {label: label for label in LEADER_TICKERS}
    missing = set(LEADER_TICKERS) - set(mapping)
    if missing:
        raise KeyError(f"{LEADERS_FILE.name} is missing {sorted(missing)}")
    return {label: mapping[label] for label in LEADER_TICKERS}
