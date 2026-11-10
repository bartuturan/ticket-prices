"""Dips near departure: does the cheapest fare sometimes fall in the last days before a flight?

A departure is a route and departure date. Its price path has one quote per days-to-departure
(a lead). A window is a range of leads, such as 7 down to 1. Inside a window the reference is
the first quote and a dip is a later quote that is lower by at least `min_drop_pct` and
`min_drop_abs`. The same measure on windows further from departure (placebo windows) shows
whether dips are really more common near departure.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ticket_prices.curves import GRID, SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY

MUTED = "#a8a7a1"
TRIP_KEYS = ["origin", "destination", "flightDate"]

Window = tuple[int, int]  # (start lead, end lead), start > end


@dataclass(frozen=True)
class DipConfig:
    min_drop_pct: float = 0.05
    min_drop_abs: float = 3.0
    bootstrap_samples: int = 2000
    seed: int = 0


def window_name(window: Window) -> str:
    return f"{window[0]}-{window[1]} days out"


def wide_curves(series: pd.DataFrame, max_lead: int) -> pd.DataFrame:
    """One row per departure, one column per lead from `max_lead` down to 1."""
    leads = list(range(max_lead, 0, -1))
    return series.pivot_table(
        index=TRIP_KEYS, columns="days_to_departure", values="price", aggfunc="min"
    ).reindex(columns=leads)


def ratio_ci(flag: np.ndarray, groups, cfg: DipConfig) -> tuple[float, float]:
    """95% interval for a share, resampling groups (not rows) because rows in a group move together."""
    codes, _ = pd.factorize(pd.Series(groups))
    hits = np.bincount(codes, weights=np.asarray(flag, dtype=float))
    counts = np.bincount(codes)
    if len(hits) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(cfg.seed)
    picks = rng.integers(len(hits), size=(cfg.bootstrap_samples, len(hits)))
    shares = hits[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return float(np.quantile(shares, 0.025)), float(np.quantile(shares, 0.975))


def late_dips(
    curves: pd.DataFrame, window: Window, cfg: DipConfig, min_obs: int | None = None, require_end: bool = True
) -> pd.DataFrame:
    """Dip measures for every departure in one window, with the rows of `curves` kept in order.

    A departure is eligible when it has at least `min_obs` quotes in the window (80% of the
    window by default) and, with `require_end`, a quote at the window's last lead. Without that,
    a window near departure would include flights only watched while still far away. The
    reference is its first quote there. `rebounded` means the quote after
    the lowest one is back at or above the reference, so a dip you could only have caught on one
    day. A dip at the window's last lead has no next quote and counts as not rebounded.
    """
    start, end = window
    leads = sorted((c for c in curves.columns if end <= c <= start), reverse=True)
    values = curves[leads].to_numpy(dtype=float)
    n = len(values)
    min_obs = max(2, int(np.ceil(0.8 * len(leads)))) if min_obs is None else max(2, min_obs)

    observed = ~np.isnan(values)
    eligible = observed.sum(axis=1) >= min_obs
    if require_end:
        eligible &= observed[:, -1]
    rows = np.arange(n)
    position = np.arange(len(leads))[None, :]

    first = observed.argmax(axis=1)
    reference = values[rows, first]
    later = np.where((position > first[:, None]) & observed, values, np.inf)
    lowest_at = later.argmin(axis=1)
    lowest = later[rows, lowest_at]

    after_lowest = (position > lowest_at[:, None]) & observed
    has_next = after_lowest.any(axis=1)
    next_price = np.where(has_next, values[rows, after_lowest.argmax(axis=1)], np.nan)

    drop = reference - lowest
    drop_pct = drop / reference
    dipped = eligible & np.isfinite(lowest) & (drop_pct >= cfg.min_drop_pct) & (drop >= cfg.min_drop_abs)
    end_price = values[:, -1]

    return pd.DataFrame(
        {
            "eligible": eligible,
            "ref_lead": np.array(leads)[first],
            "ref_price": reference,
            "min_price": np.where(np.isfinite(lowest), lowest, np.nan),
            "min_lead": np.array(leads)[lowest_at],
            "drop": drop,
            "drop_pct": drop_pct,
            "dipped": dipped,
            "rebounded": dipped & has_next & (next_price >= reference),
            "dip_at_end": dipped & ~has_next,
            "end_observed": ~np.isnan(end_price),
            "last_day_drop": eligible & ~np.isnan(end_price) & (end_price < reference),
        },
        index=curves.index,
    )


def dip_rates(
    curves: pd.DataFrame,
    windows: dict[str, Window],
    cfg: DipConfig,
    groups,
    min_obs: int | None = None,
) -> pd.DataFrame:
    """Dip share, last-day-drop share and dip size for each window, with confidence intervals."""
    groups = np.asarray(groups)
    rows = []
    for name, window in windows.items():
        result = late_dips(curves, window, cfg, min_obs)
        keep = result["eligible"].to_numpy()
        found, g = result[keep], groups[keep]
        ended = found["end_observed"].to_numpy()
        dips = found[found["dipped"]]
        low, high = ratio_ci(found["dipped"], g, cfg)
        d_low, d_high = ratio_ci(found["last_day_drop"].to_numpy()[ended], g[ended], cfg)
        rows.append(
            {
                "window": name,
                "start_lead": window[0],
                "end_lead": window[1],
                "min_drop_pct": cfg.min_drop_pct,
                "departures": len(found),
                "dip_rate": found["dipped"].mean(),
                "dip_ci_low": low,
                "dip_ci_high": high,
                "last_day_drop_rate": found["last_day_drop"].to_numpy()[ended].mean() if ended.any() else np.nan,
                "last_day_ci_low": d_low,
                "last_day_ci_high": d_high,
                "median_dip_pct": dips["drop_pct"].median() if len(dips) else np.nan,
                "median_dip_abs": dips["drop"].median() if len(dips) else np.nan,
                "rebound_share": dips["rebounded"].mean() if len(dips) else np.nan,
                "share_dips_at_end": dips["dip_at_end"].mean() if len(dips) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("window")


def daily_change_by_lead(curves: pd.DataFrame, cfg: DipConfig) -> pd.DataFrame:
    """How the price moved from lead L+1 to L, for departures quoted on both days.

    `share_down_big` is the share that fell by the dip thresholds in one day. Confidence
    intervals resample search days, which is when each change happened.
    """
    flight_dates = curves.index.get_level_values("flightDate")
    rows = []
    for lead in sorted(c for c in curves.columns if c + 1 in curves.columns):
        before, after = curves[lead + 1].to_numpy(), curves[lead].to_numpy()
        both = ~np.isnan(before) & ~np.isnan(after)
        if both.sum() == 0:
            continue
        before, after = before[both], after[both]
        change = after - before
        big_down = (change <= -cfg.min_drop_pct * before) & (change <= -cfg.min_drop_abs)
        big_up = (change >= cfg.min_drop_pct * before) & (change >= cfg.min_drop_abs)
        searched = flight_dates[both] - pd.Timedelta(days=lead)
        low, high = ratio_ci(big_down, searched, cfg)
        rows.append(
            {
                "lead": lead,
                "pairs": int(both.sum()),
                "share_down": float((change < 0).mean()),
                "share_down_big": float(big_down.mean()),
                "share_down_big_ci_low": low,
                "share_down_big_ci_high": high,
                "share_up_big": float(big_up.mean()),
                "median_log_change": float(np.median(np.log(after / before))),
                "mean_log_change": float(np.mean(np.log(after / before))),
            }
        )
    return pd.DataFrame(rows).set_index("lead")


def google_cases(fares: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Google Flights has few departures tracked to the end, so show their paths and the
    spread between the runs of one day instead of rates.

    The first result has every run for departures quoted at a lead of 1 or less. The second
    has, for each lead, how much the cheapest fare differed between runs on the same day.
    """
    keys = [*TRIP_KEYS]
    closest = fares.groupby(keys)["days_to_departure"].min()
    cases = closest[closest <= 1].index.to_frame(index=False)
    paths = fares.merge(cases, on=keys)[
        [*keys, "snapshot_at", "days_to_departure", "price", "airline", "stops"]
    ].sort_values([*keys, "snapshot_at"])

    day = fares.groupby([*keys, "searchDate", "days_to_departure"])["price"].agg(["min", "max", "count"])
    day = day[day["count"] >= 2].reset_index()
    day["spread_pct"] = (day["max"] - day["min"]) / day["min"]
    spread = day.groupby("days_to_departure").agg(
        departure_days=("spread_pct", "size"),
        share_with_spread=("spread_pct", lambda s: (s > 0).mean()),
        mean_spread_pct=("spread_pct", "mean"),
        max_spread_pct=("spread_pct", "max"),
    )
    return paths, spread


def plot_dip_rates(rates: pd.DataFrame, final: str, title: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = rates.sort_values("start_lead", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.arange(len(ordered))
    heights = 100 * ordered["dip_rate"].to_numpy()
    colors = [SERIES if name == final else MUTED for name in ordered.index]
    ax.bar(x, heights, width=0.5, color=colors)
    lower = heights - 100 * ordered["dip_ci_low"].to_numpy()
    upper = 100 * ordered["dip_ci_high"].to_numpy() - heights
    ax.errorbar(x, heights, yerr=[lower, upper], fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1.2, capsize=4)
    for position, height, top in zip(x, heights, 100 * ordered["dip_ci_high"].to_numpy()):
        ax.annotate(f"{height:.0f}%", (position, np.nan_to_num(top, nan=height)), xytext=(0, 6),
                    textcoords="offset points", ha="center", color=TEXT_PRIMARY, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{name}\n{count:,} departures" for name, count in zip(ordered.index, ordered["departures"])]
    )
    ax.set_ylabel("Departures with a dip (%)", color=TEXT_SECONDARY)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=13, pad=12)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_daily_change(change: pd.DataFrame, title: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    leads = change.index
    ax.fill_between(leads, 100 * change["share_down_big_ci_low"], 100 * change["share_down_big_ci_high"],
                    color=SERIES, alpha=0.15, linewidth=0)
    ax.plot(leads, 100 * change["share_down_big"], color=SERIES, linewidth=2)
    ax.annotate("Shaded: 95% interval", (leads[len(leads) // 2], 100 * change["share_down_big_ci_high"].iloc[len(leads) // 2]),
                xytext=(0, 8), textcoords="offset points", ha="center", color=TEXT_SECONDARY, fontsize=10)

    ax.set_xlim(leads.max(), leads.min())  # departure is on the right
    ax.set_xlabel("Days before departure (the price after the move)", color=TEXT_SECONDARY)
    ax.set_ylabel("Fell since the day before (%)", color=TEXT_SECONDARY)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=13, pad=12)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path)


def analyse_dataset(
    name: str,
    curves: pd.DataFrame,
    windows: dict[str, Window],
    cfg: DipConfig,
    groups,
    out_dir: Path,
    min_obs: int | None = None,
    unit: str = "",
) -> dict:
    """Write the rate, daily-change and per-departure files for one dataset, and return headline numbers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final = next(iter(windows))
    sensitivity = sorted({cfg.min_drop_pct, 0.02, 0.10})
    rates = pd.concat(
        [dip_rates(curves, windows, replace(cfg, min_drop_pct=pct), groups, min_obs) for pct in sensitivity]
    )
    _write(rates, out_dir / f"{name}_dip_rates.csv")
    main = rates[rates["min_drop_pct"] == cfg.min_drop_pct]

    change = daily_change_by_lead(curves, cfg)
    _write(change, out_dir / f"{name}_daily_change.csv")

    departures = late_dips(curves, windows[final], cfg, min_obs)
    _write(departures[departures["eligible"]], out_dir / f"{name}_departures.csv")

    label = name.replace("_", " ").title()
    plot_dip_rates(main, final,
                   f"{label}: departures whose fare dipped {cfg.min_drop_pct:.0%}+ inside each window",
                   out_dir / f"{name}_dip_rates.png")
    plot_daily_change(change, f"{label}: fares that fell {cfg.min_drop_pct:.0%}+ in one day",
                      out_dir / f"{name}_daily_change.png")
    return {
        "departures_in_curves": len(curves),
        "final_window": final,
        "dip_rate_by_window": {w: round(float(r), 4) for w, r in main["dip_rate"].items()},
        "departures_by_window": {w: int(c) for w, c in main["departures"].items()},
    }


def write_summary(summary: dict, out: Path) -> None:
    out.write_text(json.dumps(summary, indent=2, default=str))
