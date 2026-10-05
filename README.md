# The rise and fall of WallStreetBets

Code accompanying the paper

> A. Mancini, A. Desiderio, G. Palermo, R. Di Clemente, G. Cimini,
> *The rise and fall of WallStreetBets: social roles and opinion leaders across the GameStop saga*.

The paper identifies the social roles of users in the r/wallstreetbets community (August 2020 – July 2021). It clusters each user's weekly behavioural features with PCA followed by DBSCAN, and validates the clusters with spectral clustering. It then extracts persistent outliers with an iterative DBSCAN procedure and characterises the candidate opinion leaders among them.

## Data availability

The Reddit data, retrieved from Pushshift, **cannot be redistributed** and is not included here. The code runs on any data with the same layout, which is documented in [`data/README.md`](data/README.md). For testing, `tools/make_synthetic_data.py` generates a random dataset with that layout (see [Quick start](#quick-start)).

The three candidate opinion leaders are called `User1`, `User2` and `User3` throughout, as in the paper. Their Reddit usernames are not part of this repository (see [Opinion leaders](#opinion-leaders)).

## Repository structure

```
scripts/
  config.py        paths, time windows, feature names, parameters
  common.py        shared helpers (data loading, PCA, iterative DBSCAN, plots)
  01_ ... 07_      user clustering (Figure 2, SI S9, S12)
  08_ ... 13_      iterative-DBSCAN outliers and robustness checks (SI S6.2, S7)
  14_ ... 17_      opinion-leader analyses (SI S6)
tools/
  make_synthetic_data.py
data/              input data (not distributed; see data/README.md)
outputs/           results and figures (created by the scripts)
leaders.example.json
```

## Installation

Python ≥ 3.10.

```bash
git clone https://github.com/ComplexConnectionsLab/rise_and_fall_of_wallstreetbets.git
cd rise_and_fall_of_wallstreetbets
pip install -r requirements.txt
```

## Quick start

Generate a synthetic dataset and run the pipeline on small samples:

```bash
python tools/make_synthetic_data.py                    # writes data/ (~100 MB)
export WSB_SAMPLE_SIZE=500 WSB_N_REALIZATIONS=2        # small runs for testing
python scripts/01_dbscan_ensemble.py
python scripts/04_spectral_representative_weeks.py
python scripts/05_dbscan_tsne.py
...
```

The synthetic data is random. It only checks that the code runs, and it does not reproduce the paper's results.

## Scripts

Run the scripts from the repository root, in numerical order within each group. Each script's docstring lists what it needs and what it writes.

### 1. User roles: clustering (Figure 2, SI S9 and S12)

| Script | What it does | Paper |
|---|---|---|
| `01_dbscan_ensemble.py` | DBSCAN on 100 random samples of 10,000 users for every week: number of clusters, noise points, cluster sizes | Fig. 2A insets, 2B |
| `02_dbscan_parameter_scan.py` | Noise and Davies–Bouldin index over a grid of `eps` × `min_samples` (`--period PRE` or `POST`) | SI S9 |
| `03_plot_dbscan_parameter_scan.py` | Plot of the parameter scan | Fig. S11 |
| `04_spectral_representative_weeks.py` | Spectral clustering of one user sample per representative week | SI S12 |
| `05_dbscan_tsne.py` | DBSCAN on the same samples, with t-SNE projections | Fig. 2A |
| `06_adjusted_rand_index.py` | Agreement between DBSCAN and spectral clustering (ARI) | Fig. S13 |
| `07_plot_cluster_features_week.py` | Feature profiles of commenters, active users and posters, 24–30 Jan 2021 | Fig. 2C |

### 2. Outliers: iterative DBSCAN (Methods, SI S6.2 and S7.1)

| Script | What it does | Paper |
|---|---|---|
| `08_iterative_dbscan_outliers.py` | Iterative DBSCAN (12 iterations, `eps` = 1…12, MinPts = 10), saving the noise points left after each iteration | Methods |
| `09_outlier_frequency.py` | Number of weeks in which each user is a final outlier, for one parameter set | Methods |
| `10_run_outlier_sensitivity.py` | Runs `09_` over the sensitivity grid (number of iterations, `eps` schedule, MinPts) | SI S7.1 |
| `11_outlier_sensitivity_summary.py` | Whether the opinion leaders remain persistent outliers across configurations and thresholds | Fig. S8 |
| `12_outlier_author_weeks.py` | Every (user, week) outlier pair for the baseline configuration | SI S6.2 |
| `13_leader_calendar_persistence.py` | Converts the leaders' flagged 7-day windows into distinct calendar days and bursts | SI S6.2 |

### 3. Opinion leaders (SI S6)

| Script | What it does | Paper |
|---|---|---|
| `14_leader_precedence.py` | Daily mentions and unique users around each leader's first post | Fig. S5, Table S2 |
| `15_leader_event_study.py` | Event study around all of each leader's posts | Fig. S6 |
| `16_table_s3_baselines.py` | Change in ticker mentions, comments under posts and stock price around User2's and User3's first investment post, against community / S&P 500 baselines, with bootstrap CIs | Table S3 |
| `17_plot_ticker_attention_share.py` | Share of community ticker mentions for each leader's stock over time | Fig. S7 |

The repository does not include the data collection and cleaning steps, the construction of the reply networks and user features, or the analyses in Figures 1, 3 and 4 and SI S1–S5, S8, S10, S11, S13 and S14.

## Configuration

All paths and shared parameters live in `scripts/config.py`. These can be overridden with environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `WSB_DATA_DIR` | `data/` | input data folder |
| `WSB_OUTPUT_DIR` | `outputs/` | results folder |
| `WSB_SAMPLE_SIZE` | `10000` | users per random sample |
| `WSB_N_REALIZATIONS` | `100` | number of random samples |
| `WSB_SEED` | unset | set to an integer for reproducible sampling |
| `WSB_N_WEEKS_PRE`, `WSB_N_WEEKS_POST` | `178`, `176` | number of rolling weeks per period |

Parameters specific to a single analysis (for example `eps` and `min_samples` for one clustering step) are defined at the top of the corresponding script.

## Opinion leaders

Scripts 11, 13, 14, 15 and 16 need the Reddit usernames of User1–User3 to find them in the data. These names are not stored in the repository. To run the scripts on the original data, create the local file:

```bash
cp leaders.example.json leaders.local.json   # then fill in the usernames
```

`leaders.local.json` is listed in `.gitignore`. Without it, the scripts use the placeholder names that appear in the synthetic data.

## Citation

If you use this code, please cite the paper (full reference to be added on publication).

## License

Released under the MIT License (see [`LICENSE`](LICENSE)).
