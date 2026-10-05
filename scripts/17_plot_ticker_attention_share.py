"""
Figure S7: share of community ticker mentions captured by each opinion
leader's ticker over time, with the leader's first investment post marked
(Supplementary Materials S6.5).

Requires: data/tickers/pre/week_<i>.csv
Output:   outputs/figures/attention_share.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import config
from common import set_style, weekly_ticker_totals

set_style()

# t0 = index of the rolling week of the first investment post
LEADERS = [
    {"user": "User1", "ticker": "gme", "t0": None,
     "note": "first investment predates observation window"},
    {"user": "User2", "ticker": "crsr", "t0": 107},  # 16 Nov 2020
    {"user": "User3", "ticker": "pltr", "t0": 118},  # 27 Nov 2020
]

totals = weekly_ticker_totals(["gme", "crsr", "pltr"])
with np.errstate(divide="ignore", invalid="ignore"):
    shares = {t: np.where(totals["all"] > 0, totals[t] / totals["all"], np.nan)
              for t in ["gme", "crsr", "pltr"]}

weeks = np.arange(config.N_WEEKS["PRE"])
tick_labels = ['01/08-07/08', '26/08-01/09', '20/09-26/09', '15/10-21/10',
               '09/11-15/11', '04/12-10/12', '29/12-04/01', '23/01-29/01']

fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
for ax, leader in zip(axes, LEADERS):
    ticker, t0 = leader["ticker"], leader["t0"]
    ax.plot(weeks, 100 * shares[ticker], linewidth=4, color="#2D936C", alpha=0.85,
            label=f"{ticker.upper()} share of community attention")
    if t0 is not None:
        ax.axvline(t0, linewidth=4, ls="--", color="#9D1B8C", alpha=0.8,
                   label="First investment post")
        ax.axvspan(t0, t0 + 5, alpha=0.15, color="#9D1B8C",
                   label="5-day post-investment window")
    ax.set_ylabel("% of community\nticker mentions", fontsize=22)
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(axis="x", labelsize=18, rotation=45)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1.8)
    ax.tick_params(width=1.8)
    ax.legend(fontsize=18, frameon=False, loc="upper left")
    ax.set_title(f"{leader['user']}: {ticker.upper()}"
                 + (f"  ({leader['note']})" if t0 is None else ""), fontsize=22)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 100)

# One tick every 25 windows (dates of the corresponding window)
axes[-1].xaxis.set_major_locator(mticker.FixedLocator(np.arange(0, 176, 25)))
axes[-1].set_xticklabels(tick_labels)
axes[-1].set_xlabel("Weeks", fontsize=24)

fig.tight_layout()
out = config.ensure_dir(config.FIGURES_DIR) / "attention_share.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved: {out}")
