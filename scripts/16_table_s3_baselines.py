"""
Table S3: baseline-corrected percentage changes around the first investment
post of User2 (CRSR) and User3 (PLTR) (Supplementary Materials S6.4).

All three quantities use the same window around the post day t0:
    before = value at day t0 - PRE_WINDOW
    after  = mean of the values on days t0, t0+1, ..., t0 + POST_WINDOW - 1
             (post day included)
    change = 100 * (after - before) / before

For the weekly quantities (ticker mentions, comments under posts), "value at
day d" is the 7-day rolling window starting on day d (window index = days
since 2020-08-01). For prices it is the daily close; when the market is
closed on day t0 - PRE_WINDOW, the last close before that day is used
(e.g. Friday's close for a Sunday), and
the "after" mean uses the trading days in [t0, t0 + POST_WINDOW - 1].

| Row                  | Leader                                   | Community baseline               |
|----------------------|------------------------------------------|----------------------------------|
| Ticker mentions      | mentions of the leader's ticker          | mentions of all tracked tickers  |
| Comments under posts | leader's average comments per post       | community comments per post      |
| Stock price          | close of the leader's stock              | S&P 500 close                    |

95% CIs (ticker mentions, comments under posts) come from bootstrap
resampling of the "after" values (N_BOOT replicates). Prices have no CI.
User1 is excluded: his first investment predates the observation window.

Requires: data/tickers/pre/week_<i>.csv
          data/user_features_raw/pre/week_<i>.csv
          data/prices/<TICKER>.csv, data/prices/SP500.csv
          leaders.local.json (see README)
Output:   outputs/opinion_leaders/table_s3.csv
"""
import numpy as np
import pandas as pd

import config
from common import weekly_ticker_totals

PRE_WINDOW = 5
POST_WINDOW = 5
N_BOOT = 1000
RNG = np.random.default_rng(0)

# First investment post: index of the rolling window starting on the post day
INVESTMENTS = {
    "User2": {"ticker": "CRSR", "t0": 107},  # 16 Nov 2020
    "User3": {"ticker": "PLTR", "t0": 118},  # 27 Nov 2020
}
LEADERS = config.load_leaders()
START = config.PERIOD_START["PRE"]


# --- Percentage changes -----------------------------------------------------
def windows(series: pd.Series, t0: int):
    """(before, after values) for a series indexed by window/day number."""
    before = series.get(t0 - PRE_WINDOW, np.nan)
    after = series.reindex(range(t0, t0 + POST_WINDOW)).to_numpy(dtype=float)
    return before, after[~np.isnan(after)]


def pct(before: float, after_mean: float) -> float:
    if before == 0 or np.isnan(before) or np.isnan(after_mean):
        return np.nan
    return 100 * (after_mean - before) / before


def change_with_ci(series: pd.Series, t0: int):
    before, after = windows(series, t0)
    change = pct(before, after.mean() if len(after) else np.nan)
    if len(after) < 2 or np.isnan(change):
        return change, (np.nan, np.nan)
    boots = [pct(before, after[RNG.integers(0, len(after), len(after))].mean())
             for _ in range(N_BOOT)]
    return change, tuple(np.percentile(boots, [2.5, 97.5]))


def change(series: pd.Series, t0: int) -> float:
    before, after = windows(series, t0)
    return pct(before, after.mean() if len(after) else np.nan)


# --- Data -------------------------------------------------------------------
def comments_per_post() -> pd.DataFrame:
    """Per rolling window: community comments per post and each leader's
    average number of comments under their posts (post_comms)."""
    rows = []
    for i in range(config.N_WEEKS["PRE"]):
        path = config.RAW_FEATURES_DIR / f"week_{i}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        n_post = df["num_post"].sum()
        row = {"week": i,
               "community": df["num_comm"].sum() / n_post if n_post > 0 else np.nan}
        for label in INVESTMENTS:
            leader = df.loc[df["author"] == LEADERS[label], "post_comms"]
            row[label] = leader.iloc[0] if len(leader) == 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("week")


def load_close(name: str) -> pd.Series:
    """Daily close indexed by date. Accepts Yahoo Finance (Date, Close) or
    FRED (observation_date, SP500) column layouts."""
    df = pd.read_csv(config.price_file(name))
    df = df.rename(columns={"observation_date": "Date", "SP500": "Close"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return pd.to_numeric(df.set_index("Date")["Close"], errors="coerce").dropna().sort_index()


def price_by_day(close: pd.Series, t0: int) -> pd.Series:
    """Close prices re-indexed by day number (days since 2020-08-01), with the
    'before' day filled by the last available close on or before it."""
    days = (close.index - START).days
    series = pd.Series(close.values, index=days)
    before_day = t0 - PRE_WINDOW
    prior = series[series.index <= before_day]
    if len(prior):
        series.loc[before_day] = prior.iloc[-1]
    return series.sort_index()


# --- Run --------------------------------------------------------------------
tickers = weekly_ticker_totals([info["ticker"].lower() for info in INVESTMENTS.values()])
mentions = {k: pd.Series(v) for k, v in tickers.items()}
cpp = comments_per_post()
sp500 = load_close("SP500")

rows = []
for label, info in INVESTMENTS.items():
    t0, ticker = info["t0"], info["ticker"]

    m, m_ci = change_with_ci(mentions[ticker.lower()], t0)
    c, c_ci = change_with_ci(cpp[label], t0)
    p = change(price_by_day(load_close(ticker), t0), t0)

    rows += [
        {"leader": f"{label} ({ticker})", "quantity": "Ticker mentions",
         "change_pct": m, "ci_low": m_ci[0], "ci_high": m_ci[1],
         "community_pct": change(mentions["all"], t0)},
        {"leader": f"{label} ({ticker})", "quantity": "Comments under posts",
         "change_pct": c, "ci_low": c_ci[0], "ci_high": c_ci[1],
         "community_pct": change(cpp["community"], t0)},
        {"leader": f"{label} ({ticker})", "quantity": "Stock price",
         "change_pct": p, "ci_low": np.nan, "ci_high": np.nan,
         "community_pct": change(price_by_day(sp500, t0), t0)},
    ]

table = pd.DataFrame(rows)
out = config.ensure_dir(config.OUTPUT_DIR / "opinion_leaders") / "table_s3.csv"
table.to_csv(out, index=False)

print(f"Window: day t0-{PRE_WINDOW} vs mean of days t0..t0+{POST_WINDOW - 1}\n")
for _, r in table.iterrows():
    ci = "" if np.isnan(r.ci_low) else f" [{r.ci_low:+.0f}, {r.ci_high:+.0f}]"
    print(f"{r.leader:13s} {r.quantity:21s} {r.change_pct:+6.0f}%{ci:14s}"
          f"  community {r.community_pct:+.0f}%")
print(f"\nSaved: {out}")
