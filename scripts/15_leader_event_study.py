"""
Event study around every post of each opinion leader about the target
ticker (Figure S6, Supplementary Materials S6.3).

For each post (and for bursts deduplicated with a >= 7-day gap), daily
mentions and unique users in [-EVENT_WINDOW, +EVENT_WINDOW] days are
standardized by the pre-event baseline (days -3..-1). The plot shows the
mean with a 95% bootstrap band (Figure S6).

Requires: data/ticker_text/<TICKER>/<TICKER>_{posts,comments}.csv,
          leaders.local.json (see README)
Output:   outputs/figures/layer2_eventstudy.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from common import daily_series, load_ticker_activity, load_ticker_posts, set_style

set_style()

LEADERS = config.load_leaders()
OBSERVATION_START = config.PERIOD_START["PRE"]

EVENT_WINDOW = 3
BASELINE_LO, BASELINE_HI = -3, -1
N_BOOT = 1000
RNG = np.random.default_rng(0)


def leader_post_dates(ticker: str, username: str) -> pd.Series:
    """All dates on which the leader posted about the ticker."""
    posts = load_ticker_posts(ticker)
    dates = posts.loc[posts["author"] == username, "time"].dt.floor("D")
    if ticker == "GME":
        dates = dates[dates >= OBSERVATION_START]
    return dates.sort_values().reset_index(drop=True)


def event_windows(series: pd.Series, event_dates, lo: int, hi: int) -> np.ndarray:
    """(n_events, hi-lo+1) array; row k = series at days event_k+lo..event_k+hi."""
    out = np.full((len(event_dates), hi - lo + 1), np.nan)
    for k, d in enumerate(event_dates):
        window = pd.date_range(d + pd.Timedelta(days=lo), d + pd.Timedelta(days=hi), freq="D")
        out[k, :] = series.reindex(window).values
    return out


def standardize_by_pre(mat: np.ndarray, lo: int) -> np.ndarray:
    """Standardize each event by its own pre-event baseline."""
    pre = mat[:, np.arange(BASELINE_LO - lo, BASELINE_HI - lo + 1)]
    mu = np.nanmean(pre, axis=1, keepdims=True)
    sd = np.nanstd(pre, axis=1, keepdims=True)
    sd = np.where(sd == 0, np.nan, sd)
    return (mat - mu) / sd


def deduplicate_events(dates: pd.Series, min_gap_days: int = 7) -> pd.Series:
    """Keep the first post of each burst of posts closer than min_gap_days."""
    kept, last = [], None
    for d in dates:
        if last is None or (d - last).days >= min_gap_days:
            kept.append(d)
            last = d
    return pd.Series(kept, dtype="datetime64[ns]")


def bootstrap_ci(mat: np.ndarray, n_boot: int, alpha: float = 0.05):
    """Mean over events and bootstrap CI (resampling events)."""
    boot = np.empty((n_boot, mat.shape[1]))
    for b in range(n_boot):
        idx = RNG.integers(0, mat.shape[0], size=mat.shape[0])
        boot[b] = np.nanmean(mat[idx], axis=0)
    return (np.nanmean(mat, axis=0),
            np.nanpercentile(boot, 100 * alpha / 2, axis=0),
            np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0))


fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True, sharey="row")
offsets = np.arange(-EVENT_WINDOW, EVENT_WINDOW + 1)

for row, (label, username) in enumerate(LEADERS.items()):
    ticker = config.LEADER_TICKERS[label]
    daily = daily_series(load_ticker_activity(ticker), leader_username=username)
    all_dates = leader_post_dates(ticker, username)
    dedup_dates = deduplicate_events(all_dates, min_gap_days=7)

    for col, (dates, tag) in enumerate([(all_dates, "all posts"),
                                        (dedup_dates, "deduplicated (≥7d gap)")]):
        ax = axes[row, col]
        n = len(dates)
        m_mat = event_windows(daily["mentions"], dates, -EVENT_WINDOW, EVENT_WINDOW)
        u_mat = event_windows(daily["unique_users"], dates, -EVENT_WINDOW, EVENT_WINDOW)
        m_std = standardize_by_pre(m_mat, -EVENT_WINDOW)
        u_std = standardize_by_pre(u_mat, -EVENT_WINDOW)

        if n <= 8:  # few events: also show individual traces
            for k in range(n):
                ax.plot(offsets, m_std[k], color="#c0392b", alpha=0.25, lw=0.8,
                        label="Individual event traces" if k == 0 else None)

        m_mean, m_lo, m_hi = bootstrap_ci(m_std, N_BOOT)
        u_mean, u_lo, u_hi = bootstrap_ci(u_std, N_BOOT)
        ax.plot(offsets, m_mean, color="#c0392b", lw=2, label="Mentions (standardized)")
        ax.fill_between(offsets, m_lo, m_hi, color="#c0392b", alpha=0.2)
        ax.plot(offsets, u_mean, color="#2c3e50", lw=2, label="Unique users (standardized)")
        ax.fill_between(offsets, u_lo, u_hi, color="#2c3e50", alpha=0.2)
        ax.axvline(0, color="black", ls="--", lw=1.2)
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_title(f"{label} — {ticker}  ({tag}, n={n})")
        if row == 2:
            ax.set_xlabel("Days from post")
        if col == 0:
            ax.set_ylabel("Std. dev. above pre-event baseline")

axes[0, 0].legend(loc="upper left", fontsize=9, frameon=True)
fig.suptitle("Event study around opinion leaders' posts about the target ticker\n"
             "(mean and 95% bootstrap CI, standardized by pre-event baseline)",
             y=1.00, fontsize=13)
fig.tight_layout()
out = config.ensure_dir(config.FIGURES_DIR) / "layer2_eventstudy.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")

