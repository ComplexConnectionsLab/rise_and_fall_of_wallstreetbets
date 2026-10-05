# Input data

The original data (r/wallstreetbets posts and comments, August 2020 – July 2021, retrieved from Pushshift) cannot be redistributed. This folder is empty in the repository. This file describes the layout the scripts expect, so the code can be run on equivalent data. `tools/make_synthetic_data.py` creates a random dataset with exactly this layout.

```
data/
  features/
    pre/week_0.csv ... week_177.csv     Aug 2020 - Jan 2021
    post/week_0.csv ... week_175.csv    Feb 2021 - Jul 2021
  tickers/
    pre/week_0.csv ... week_177.csv
  ticker_text/
    GME/GME_posts.csv, GME_comments.csv
    CRSR/CRSR_posts.csv, CRSR_comments.csv
    PLTR/PLTR_posts.csv, PLTR_comments.csv
  user_features_raw/
    pre/week_0.csv ... week_177.csv
  prices/
    CRSR.csv, PLTR.csv, SP500.csv
```

## Weeks

The data is divided into 7-day windows shifted by one day. Window `i` covers the days from `start + i` to `start + i + 6`, where `start` is 2020-08-01 for the pre-squeeze period and 2021-02-01 for the post-squeeze period.

## `features/<period>/week_<i>.csv`

One row per active user in window `i`. The first column is an unnamed row index. All feature values are standardized within the week.

| Column | Description |
|---|---|
| `author` | user identifier (any string) |
| `num_comm` | number of comments |
| `comm_score` | average score of comments |
| `comm_entropy` | average normalized word entropy of comments |
| `comm_jargon` | average number of jargon terms in comments |
| `comm_sent` | average VADER sentiment of comments |
| `first_children` | average number of direct replies to the user's comments |
| `num_post` | number of posts |
| `post_score` | average score of posts |
| `post_entropy` | average normalized word entropy of posts |
| `post_jargon` | average number of jargon terms in posts |
| `post_sent` | average VADER sentiment of posts |
| `post_comms` | average number of comments under the user's posts |
| `k_out`, `k_in` | out- and in-degree in the weekly reply-to network |
| `s_out`, `s_in` | out- and in-strength in the weekly reply-to network |

See the Methods section of the paper for how each feature is computed.

The pre-squeeze files also contain one column per ticker (`spy`, `amd`, `tsla`, …, `gme`; the full list is `TICKER_COLS` in `scripts/config.py`). These columns are dropped before clustering. The post-squeeze files do not contain them.

The clustering uses **every column except `author`** (after dropping ticker columns). Any extra columns therefore enter the PCA, so remove them first.

## `tickers/pre/week_<i>.csv`

One row per user in window `i`. It has an unnamed row index, an `author` column and one column per ticker (lower case, for example `gme`, `crsr`, `pltr`) with the number of mentions of that ticker in the user's posts. The paper uses 157 manually validated tickers (see Methods). Community-wide totals are obtained by summing over users.

## `ticker_text/<TICKER>/<TICKER>_{posts,comments}.csv`

All posts or comments mentioning the ticker. The scripts use only two columns:

| Column | Description |
|---|---|
| `author` | user identifier (`[deleted]` and `[removed]` are ignored) |
| `time` | timestamp, parseable by `pandas.to_datetime` |

## `user_features_raw/pre/week_<i>.csv`

The same weekly user features **before standardization**. Only these columns are used (by `16_table_s3_baselines.py`):

| Column | Description |
|---|---|
| `author` | user identifier |
| `num_comm` | number of comments written in the window |
| `num_post` | number of posts written in the window |
| `post_comms` | average number of comments under the user's posts |

## `prices/<NAME>.csv`

Daily closing prices of CRSR and PLTR (Yahoo Finance) and of the S&P 500. Two column layouts are accepted: `Date`, `Close` (Yahoo Finance), or `observation_date`, `SP500` (FRED).

