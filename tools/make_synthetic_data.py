"""
Generate a small SYNTHETIC dataset with the same layout and columns as the
original (non-shareable) data, so that every script can be run end to end.

The numbers are random and carry no information about WallStreetBets:
users are drawn from three Gaussian blobs (mimicking commenters, active
users and posters) plus a handful of far-away outliers, among which the
placeholder opinion leaders "User1", "User2" and "User3".

Usage:
    python tools/make_synthetic_data.py [--users 800] [--out data]

Then run the scripts with a matching sample size, e.g.
    WSB_SAMPLE_SIZE=500 WSB_N_REALIZATIONS=2 python scripts/01_dbscan_ensemble.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import config  # noqa: E402

LEADERS = ["User1", "User2", "User3"]
OTHER_TICKERS = ["spy", "tsla", "amd", "aapl", "nok", "bb", "amc"]


def make_features(n_users: int, rng, with_tickers: bool) -> pd.DataFrame:
    n_feat = len(config.FEATURES)
    centers = rng.normal(0, 4, size=(3, n_feat))
    weights = [0.6, 0.3, 0.1]
    n_out = max(20, n_users // 50)
    n_in = n_users - n_out

    blob = rng.choice(3, size=n_in, p=weights)
    X_in = centers[blob] + rng.normal(0, 0.3, size=(n_in, n_feat))
    X_out = rng.normal(0, 25, size=(n_out, n_feat))
    X = np.vstack([X_in, X_out])
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    authors = [f"user_{k}" for k in range(n_in)]
    authors += LEADERS + [f"outlier_{k}" for k in range(n_out - len(LEADERS))]
    df = pd.DataFrame(X, columns=config.FEATURES)
    df.insert(0, "author", authors)
    if with_tickers:
        for t in config.TICKER_COLS:
            df[t] = rng.poisson(0.2, size=n_users)
    return df


def make_ticker_counts(n_users: int, week: int, rng) -> pd.DataFrame:
    df = pd.DataFrame({"author": [f"user_{k}" for k in range(n_users)]})
    boost = {"gme": 1 + week / 20, "crsr": 1 + 3 * (week >= 107),
             "pltr": 1 + 3 * (week >= 118)}
    for t in ["gme", "crsr", "pltr"] + OTHER_TICKERS:
        df[t] = rng.poisson(0.1 * boost.get(t, 1), size=n_users)
    return df


def make_raw_features(n_users: int, rng) -> pd.DataFrame:
    authors = [f"user_{k}" for k in range(n_users - len(LEADERS))] + LEADERS
    return pd.DataFrame({
        "author": authors,
        "num_comm": rng.poisson(20, size=n_users),
        "num_post": rng.poisson(1, size=n_users),
        "post_comms": rng.gamma(2, 20, size=n_users),
    })


def make_prices(p0: float, rng, vol: float = 0.04) -> pd.DataFrame:
    days = pd.bdate_range("2020-07-01", "2021-02-28")
    close = p0 * np.exp(np.cumsum(rng.normal(0, vol, size=len(days))))
    return pd.DataFrame({"Date": days.strftime("%Y-%m-%d"), "Close": close.round(2)})


def make_ticker_text(ticker: str, leader: str, first_post: str, rng) -> tuple:
    days = pd.date_range("2020-06-01", "2021-02-28", freq="D")
    rows = []
    for d in days:
        rate = 5 + (40 if d >= pd.Timestamp(first_post) else 0)
        for _ in range(rng.poisson(rate)):
            t = d + pd.Timedelta(seconds=int(rng.integers(0, 86_400)))
            rows.append((f"user_{rng.integers(0, 3000)}", t))
    comments = pd.DataFrame(rows, columns=["author", "time"])
    posts = comments.sample(frac=0.1, random_state=0).copy()
    leader_days = pd.date_range(first_post, "2021-01-31", freq="9D")
    leader_posts = pd.DataFrame({"author": leader, "time": leader_days + pd.Timedelta(hours=15)})
    posts = pd.concat([posts, leader_posts]).sort_values("time")
    return posts, comments


def main(n_users: int, out: Path, seed: int):
    rng = np.random.default_rng(seed)
    for period, folder in [("PRE", "pre"), ("POST", "post")]:
        d = out / "features" / folder
        d.mkdir(parents=True, exist_ok=True)
        for i in range(config.N_WEEKS[period]):
            make_features(n_users, rng, with_tickers=(period == "PRE")).to_csv(d / f"week_{i}.csv")

    d = out / "tickers" / "pre"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(config.N_WEEKS["PRE"]):
        make_ticker_counts(n_users, i, rng).to_csv(d / f"week_{i}.csv")

    d = out / "user_features_raw" / "pre"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(config.N_WEEKS["PRE"]):
        make_raw_features(n_users, rng).to_csv(d / f"week_{i}.csv")

    d = out / "prices"
    d.mkdir(parents=True, exist_ok=True)
    for name, p0 in [("CRSR", 30), ("PLTR", 10), ("GME", 5)]:
        make_prices(p0, rng).to_csv(d / f"{name}.csv", index=False)
    make_prices(3500, rng, vol=0.01).rename(
        columns={"Date": "observation_date", "Close": "SP500"}).to_csv(d / "SP500.csv", index=False)

    for ticker, leader, first in [("GME", "User1", "2020-09-01"),
                                  ("CRSR", "User2", "2020-11-16"),
                                  ("PLTR", "User3", "2020-11-27")]:
        d = out / "ticker_text" / ticker
        d.mkdir(parents=True, exist_ok=True)
        posts, comments = make_ticker_text(ticker, leader, first, rng)
        posts.to_csv(d / f"{ticker}_posts.csv", index=False)
        comments.to_csv(d / f"{ticker}_comments.csv", index=False)
    print(f"Synthetic data written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=800, help="users per week")
    ap.add_argument("--out", type=Path, default=config.DATA_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.users, args.out, args.seed)
