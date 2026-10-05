"""
Calendar-day persistence of the three opinion leaders as outliers
(Supplementary Materials S6.2).

Because consecutive "weeks" are 7-day windows shifted by one day, the number
of flagged windows overstates persistence. This script converts each
leader's flagged windows into the set of distinct calendar days covered and
counts the separate bursts of activity.

Requires: 12_outlier_author_weeks.py, leaders.local.json (see README)
"""
import pandas as pd

import config

LEADERS = config.load_leaders()


def week_to_days(period: str, week_i: int) -> set:
    """The 7 calendar dates covered by rolling window `week_i` in `period`."""
    start = config.PERIOD_START[period] + pd.Timedelta(days=week_i * config.SHIFT_DAYS)
    return {start + pd.Timedelta(days=d) for d in range(config.WINDOW_DAYS)}


for period in ["PRE", "POST"]:
    df = pd.read_csv(config.OUTLIERS_DIR / f"{period}_baseline_pairs.csv")
    print(f"\n=== {period} (period starts {config.PERIOD_START[period].date()}) ===")
    for label, username in LEADERS.items():
        name = f"{label} ({config.LEADER_TICKERS[label]})"
        user_weeks = sorted(df.loc[df["author"] == username, "week"].tolist())
        all_days = set()
        for w in user_weeks:
            all_days |= week_to_days(period, w)
        if all_days:
            days = sorted(all_days)
            n_bursts = 1 + sum((days[k + 1] - days[k]).days > 1
                               for k in range(len(days) - 1))
            date_range = f"{days[0].date()} -> {days[-1].date()}"
        else:
            n_bursts, date_range = 0, "n/a"
        print(f"  {name:15s}: {len(user_weeks):4d} windows -> "
              f"{len(all_days):4d} calendar days, {n_bursts} burst(s), {date_range}")
