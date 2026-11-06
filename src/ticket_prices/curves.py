"""Booking curves: how the cheapest fare moves as departure approaches."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ticket_prices.backtest import ROUTE_KEYS, Config, relative_curves

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e8e7e3"
SERIES = "#2a78d6"


def _describe(relative: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "median": relative.median(),
            "p25": relative.quantile(0.25),
            "p75": relative.quantile(0.75),
            "mean": relative.mean(),
            "share_below_first_day": (relative < 1).sum() / relative.notna().sum(),
            "trips": relative.notna().sum(),
        }
    )
    summary.index.name = "days_to_departure"
    return summary


def booking_curve(curves: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _describe(relative_curves(curves, cfg))


def booking_curve_by_route(curves: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    relative = relative_curves(curves, cfg)
    frames = {route: _describe(trips) for route, trips in relative.groupby(level=ROUTE_KEYS)}
    return pd.concat(frames, names=ROUTE_KEYS)


def plot_booking_curve(curve: pd.DataFrame, trips: int, cfg: Config, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    leads = curve.index
    ax.axhline(100, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 4)))
    ax.fill_between(leads, 100 * curve["p25"], 100 * curve["p75"], color=SERIES, alpha=0.15, linewidth=0)
    ax.plot(leads, 100 * curve["median"], color=SERIES, linewidth=2)

    lowest = curve["median"].idxmin()
    if curve.loc[lowest, "median"] < 1:  # with no dip below the first day there is nothing to mark
        ax.plot(lowest, 100 * curve.loc[lowest, "median"], "o", markersize=8, color=SERIES,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(
            f"Lowest median: {100 * curve.loc[lowest, 'median']:.0f} at {lowest} days out",
            (lowest, 100 * curve.loc[lowest, "median"]),
            xytext=(0, -18), textcoords="offset points", ha="center", color=TEXT_PRIMARY, fontsize=10,
        )
    ax.annotate(
        "Shaded: middle 50% of trips", (leads[len(leads) // 10], 100 * curve["p75"].iloc[len(leads) // 10]),
        xytext=(0, 8), textcoords="offset points", ha="left", color=TEXT_SECONDARY, fontsize=10,
    )

    ax.set_xlim(cfg.start_lead, cfg.end_lead)
    ax.set_xlabel("Days before departure", color=TEXT_SECONDARY)
    ax.set_ylabel(f"Cheapest fare (price {cfg.start_lead} days out = 100)", color=TEXT_SECONDARY)
    ax.set_title(
        f"Cheapest fare as departure approaches, {trips:,} training trips",
        loc="left", color=TEXT_PRIMARY, fontsize=13, pad=12,
    )
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
