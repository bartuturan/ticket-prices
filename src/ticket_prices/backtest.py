"""Backtest simple buying rules, which any later model has to beat.

A trip is a route and departure date. The traveller starts shopping `start_lead` days
before departure and must buy by `end_lead` days before departure.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

ROUTE_KEYS = ["origin", "destination"]
TRIP_KEYS = [*ROUTE_KEYS, "flightDate"]


@dataclass(frozen=True)
class Config:
    start_lead: int = 30
    end_lead: int = 1
    min_coverage: float = 0.8
    min_route_trips: int = 20
    bootstrap_samples: int = 2000
    seed: int = 0


def default_split_date(series: pd.DataFrame, fraction: float = 0.6) -> pd.Timestamp:
    first = pd.Timestamp(series["searchDate"].min())
    last = pd.Timestamp(series["searchDate"].max())
    return (first + (last - first) * fraction).normalize()


def trip_curves(series: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Prices with one row per usable trip and one column per lead, latest purchase last.

    A trip must have quotes on its first and last buying days and on at least
    `min_coverage` of the days between. This is what removes trips cut off by the edges
    of the collection window.
    """
    leads = list(range(cfg.start_lead, cfg.end_lead - 1, -1))
    in_window = series[series["days_to_departure"].between(cfg.end_lead, cfg.start_lead)]
    wide = in_window.pivot_table(
        index=TRIP_KEYS, columns="days_to_departure", values="price", aggfunc="min"
    ).reindex(columns=leads)
    usable = (
        wide[cfg.start_lead].notna()
        & wide[cfg.end_lead].notna()
        & (wide.notna().mean(axis=1) >= cfg.min_coverage)
    )
    return wide[usable]


def split_trips(curves: pd.DataFrame, split: pd.Timestamp, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep every training quote before `split` and every test quote on or after it.

    Trips on nearby dates share price shocks on the same search day, so splitting by
    departure date alone would let the training data see the test period. Trips that
    straddle the split are dropped.
    """
    flight_dates = pd.to_datetime(curves.index.get_level_values("flightDate"))
    train = curves[flight_dates < split + pd.Timedelta(days=cfg.end_lead)]
    test = curves[flight_dates >= split + pd.Timedelta(days=cfg.start_lead)]
    return train, test


def relative_curves(curves: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Prices as a ratio of the first-day price, so cheap and dear routes compare."""
    return curves.div(curves[cfg.start_lead], axis=0)


def purchase_prices(curves: pd.DataFrame) -> pd.DataFrame:
    """The price paid when planning to buy at each lead.

    If a day has no quote the purchase slips to the next quoted day. A forward fill would
    wrongly let the traveller buy at an older price.
    """
    return curves.bfill(axis=1)


def best_fixed_lead(train: pd.DataFrame, cfg: Config) -> int:
    return int(relative_curves(purchase_prices(train), cfg).mean().idxmin())


def best_lead_by_route(train: pd.DataFrame, cfg: Config) -> dict[tuple[str, str], int]:
    """Best lead for each route with enough trips. Other routes use the overall best."""
    return {
        route: best_fixed_lead(trips, cfg)
        for route, trips in train.groupby(level=ROUTE_KEYS)
        if len(trips) >= cfg.min_route_trips
    }


def policy_prices(train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """What each rule pays on each test trip. Rules are chosen from training trips only."""
    purchase = purchase_prices(test)
    overall = best_fixed_lead(train, cfg)
    by_route = best_lead_by_route(train, cfg)

    route_leads = [
        by_route.get((origin, destination), overall)
        for origin, destination, _ in test.index
    ]
    route_columns = purchase.columns.get_indexer(route_leads)

    prices = pd.DataFrame(
        {
            "buy_first_day": purchase[cfg.start_lead],
            "buy_last_day": purchase[cfg.end_lead],
            "best_fixed_lead": purchase[overall],
            "best_lead_by_route": purchase.to_numpy()[np.arange(len(purchase)), route_columns],
            "hindsight": test.min(axis=1),
        },
        index=test.index,
    )
    return prices, {"overall": overall, "by_route": by_route}


def summarize(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Dollar results per rule, against buying on the first day.

    Confidence intervals resample departure dates, not trips, because trips leaving on
    the same day are not independent.
    """
    base = prices["buy_first_day"]
    possible = (base - prices["hindsight"]).mean()
    flight_dates = prices.index.get_level_values("flightDate")
    rng = np.random.default_rng(cfg.seed)

    rows = []
    for policy in prices.columns:
        saving = base - prices[policy]
        by_date = saving.groupby(flight_dates).agg(["sum", "count"]).to_numpy()
        picks = rng.integers(len(by_date), size=(cfg.bootstrap_samples, len(by_date)))
        means = by_date[picks, 0].sum(axis=1) / by_date[picks, 1].sum(axis=1)
        rows.append(
            {
                "policy": policy,
                "trips": len(prices),
                "mean_price": prices[policy].mean(),
                "mean_saving": saving.mean(),
                "saving_ci_low": np.quantile(means, 0.025),
                "saving_ci_high": np.quantile(means, 0.975),
                "mean_saving_pct": 100 * saving.mean() / base.mean(),
                "share_of_possible_saving": saving.mean() / possible if possible > 0 else np.nan,
                "share_trips_cheaper": (saving > 0).mean(),
                "share_trips_dearer": (saving < 0).mean(),
                "p95_overpay": (-saving).quantile(0.95),
            }
        )
    return pd.DataFrame(rows).set_index("policy")
