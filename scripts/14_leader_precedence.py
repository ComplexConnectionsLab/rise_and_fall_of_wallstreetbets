"""
Figure S5 and Table S2: aggregate temporal precedence for the three opinion
leaders (Supplementary Materials S6.1).

For each leader (User1/GME, User2/CRSR, User3/PLTR):
    - finds their first post about the target ticker (for User1, the first
      one inside the observation window, since his original investment
      dates back to 2019),
    - computes daily ticker mentions (posts + comments) and daily unique
      users mentioning the ticker (excluding the leader),
    - plots both series (7-day centred mean) with the first-post date and
      the August 2020 baseline level marked, and prints a summary table.

Requires: data/ticker_text/<TICKER>/<TICKER>_{posts,comments}.csv,
          leaders.local.json (see README)
Output:   outputs/figures/precedence.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from common import daily_series, load_ticker_activity, load_ticker_posts, set_style

set_style()

LEADERS = config.load_leaders()
OBSERVATION_START = config.PERIOD_START["PRE"]
BASELINE_START = pd.Timestamp("2020-08-01")
BASELINE_END = pd.Timestamp("2020-08-31")
SHARED_START = pd.Timestamp("2020-08-01")
SHARED_END = pd.Timestamp("2021-02-01")
SMOOTH = 7  # centred rolling window, days


def first_post_date(ticker: str, username: str) -> pd.Timestamp:
    posts = load_ticker_posts(ticker)
    mask = posts["author"] == username
    if ticker == "GME":
        mask &= posts["time"] >= OBSERVATION_START
    leader_posts = posts.loc[mask].sort_values("time")
    if leader_posts.empty:
        raise ValueError(f"No posts found for the {ticker} leader. "
                         f"Check leaders.local.json.")
    return leader_posts["time"].iloc[0].floor("D")


fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
summary_rows = []
first_ax2 = None

for i, (ax, (label, username)) in enumerate(zip(axes, LEADERS.items())):
    ticker = config.LEADER_TICKERS[label]
    df = load_ticker_activity(ticker)
    daily = daily_series(df, leader_username=username)
    base = daily.loc[BASELINE_START:BASELINE_END].mean()

    view = daily.rolling(SMOOTH, center=True, min_periods=1).mean().loc[SHARED_START:SHARED_END]
    daily_raw = daily.loc[SHARED_START:SHARED_END]
    t0 = first_post_date(ticker, username)

    ax.axvspan(BASELINE_START, BASELINE_END, color="gray", alpha=0.15, zorder=0,
               label="Aug 2020 baseline period")

    color_m = "#c0392b"
    ax.plot(view.index, view["mentions"], color=color_m, lw=2,
            label="Mentions / day (7-day mean)")
    ax.axhline(base["mentions"], color=color_m, ls=":", lw=1.2, alpha=0.8,
               label="Mentions baseline level")
    ax.set_ylabel("Mentions / day", color=color_m)
    ax.tick_params(axis="y", labelcolor=color_m)
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.9)

    ax2 = ax.twinx()
    color_u = "#2c3e50"
    ax2.fill_between(view.index, view["unique_users"], color=color_u, alpha=0.15)
    ax2.plot(view.index, view["unique_users"], color=color_u, lw=1.5,
             label="Unique users / day (7-day mean)")
    ax2.axhline(base["unique_users"], color=color_u, ls=":", lw=1.2, alpha=0.8,
                label="Unique users baseline level")
    ax2.set_ylabel("Unique users / day", color=color_u)
    ax2.tick_params(axis="y", labelcolor=color_u)
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=0.9)

    ax.axvline(t0, color="black", ls="--", lw=1.5,
               label="Leader's first post" if i == 0 else None)
    ax.set_title(f"{label} — {ticker}")
    if i == 0:
        first_ax2 = ax2

    peak_date = daily_raw["mentions"].idxmax()
    in_view = t0 in daily_raw.index
    summary_rows.append({
        "user": label, "ticker": ticker, "first_post": t0.date(),
        "aug_baseline_mentions": round(base["mentions"], 1),
        "aug_baseline_users": round(base["unique_users"], 1),
        "mentions_at_first_post": int(daily_raw["mentions"].loc[t0]) if in_view else np.nan,
        "unique_users_at_first_post": int(daily_raw["unique_users"].loc[t0]) if in_view else np.nan,
        "peak_date": peak_date.date(),
        "mentions_at_peak": int(daily_raw["mentions"].loc[peak_date]),
        "unique_users_at_peak": int(daily_raw["unique_users"].loc[peak_date]),
        "lead_time_days": (peak_date - t0).days,
    })

axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[-1].set_xlim(SHARED_START, SHARED_END)

handles_l, labels_l = axes[0].get_legend_handles_labels()
handles_r, labels_r = first_ax2.get_legend_handles_labels()
axes[0].legend(handles_l + handles_r, labels_l + labels_r, loc="upper left",
               bbox_to_anchor=(0.0, -0.02), frameon=True, fontsize=9, ncol=2)

fig.suptitle("Opinion leaders' first post vs. community engagement "
             "with the target ticker", y=1.00, fontsize=14)
fig.tight_layout()
out = config.ensure_dir(config.FIGURES_DIR) / "precedence.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")

print("\nPrecedence summary (Table S2):")
print(pd.DataFrame(summary_rows).to_string(index=False))

