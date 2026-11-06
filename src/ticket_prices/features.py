"""Turn the daily series into one row per trip per day, using only what is known that day."""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from ticket_prices.backtest import ROUTE_KEYS, TRIP_KEYS

LEAD = "days_to_departure"
CATEGORICAL = ["flight_weekday", "cheapest_airline"]
NUMERIC = [
    "days_to_holiday",
    "vs_trip_low",
    "vs_trip_high",
    "vs_trip_first",
    "vs_neighbors",
    "vs_route_today",
    "route_change_7d",
    "change_1d",
    "change_3d",
    "change_7d",
    "days_since_change",
    "n_flights",
    "n_airlines",
    "next_airline_gap",
    "cheapest_nonstop",
    "seats_remaining",
    "log_distance",
    "log_route_median",
]
FEATURES = NUMERIC + CATEGORICAL
NEIGHBOR_DAYS = (-3, -2, -1, 1, 2, 3)
MAX_HOLIDAY_DAYS = 14


def _days_since_change(known: np.ndarray) -> np.ndarray:
    since = np.full(known.shape, np.nan)
    previous_price = np.full(known.shape[0], np.nan)
    previous_since = np.full(known.shape[0], np.nan)
    for column in range(known.shape[1]):
        price = known[:, column]
        changed = ~np.isnan(price) & (np.isnan(previous_price) | (price != previous_price))
        previous_since = np.where(changed, 0, previous_since + 1)
        since[:, column] = previous_since
        previous_price = price
    return since


def _trip_history(series: pd.DataFrame) -> pd.DataFrame:
    """Features from this trip's own past quotes. Columns run oldest quote to newest."""
    wide = series.pivot(index=TRIP_KEYS, columns=LEAD, values="price")
    wide = wide.reindex(columns=range(int(wide.columns.max()), int(wide.columns.min()) - 1, -1))
    known = wide.ffill(axis=1)  # the latest quote seen so far, on days without a quote
    first = wide.bfill(axis=1).iloc[:, 0]

    frames = {
        "vs_trip_low": known / known.cummin(axis=1),
        "vs_trip_high": known / known.cummax(axis=1),
        "vs_trip_first": known.div(first, axis=0),
        "change_1d": known / known.shift(1, axis=1),
        "change_3d": known / known.shift(3, axis=1),
        "change_7d": known / known.shift(7, axis=1),
    }
    frames["days_since_change"] = pd.DataFrame(
        _days_since_change(known.to_numpy()), index=wide.index, columns=wide.columns
    )

    rows = wide.index.get_indexer(pd.MultiIndex.from_frame(series[TRIP_KEYS]))
    columns = wide.columns.get_indexer(series[LEAD])
    return pd.DataFrame(
        {name: frame.to_numpy()[rows, columns] for name, frame in frames.items()},
        index=series.index,
    )


def _neighbor_median(series: pd.DataFrame) -> pd.Series:
    """Median price the same day for the same route, departing up to 3 days either side."""
    keys = [*ROUTE_KEYS, "searchDate", "flightDate"]
    lookup = series[[*keys, "price"]]
    neighbors = {}
    for days in NEIGHBOR_DAYS:
        shifted = lookup.assign(flightDate=lookup["flightDate"] + pd.Timedelta(days=days))
        neighbors[days] = series[keys].merge(shifted, how="left", on=keys)["price"].to_numpy()
    return pd.DataFrame(neighbors, index=series.index).median(axis=1)


def _route_change_7d(series: pd.DataFrame, route_median: pd.Series) -> pd.Series:
    """How the route's median quote has moved over the past week, across all departures."""
    keys = [*ROUTE_KEYS, "searchDate"]
    daily = series[keys].assign(route_median=route_median).drop_duplicates(keys)
    week_ago = daily.assign(searchDate=daily["searchDate"] + pd.Timedelta(days=7))
    past = series[keys].merge(week_ago, how="left", on=keys)["route_median"].to_numpy()
    return route_median / past


def _days_to_holiday(flight_dates: pd.Series) -> pd.Series:
    span = pd.Timedelta(days=60)
    holidays = USFederalHolidayCalendar().holidays(flight_dates.min() - span, flight_dates.max() + span)
    days = pd.Series(flight_dates.unique())
    if len(holidays) == 0:
        distance = pd.Series(MAX_HOLIDAY_DAYS, index=days)
    else:
        gaps = np.abs((days.to_numpy()[:, None] - holidays.to_numpy()[None, :]) / np.timedelta64(1, "D"))
        distance = pd.Series(gaps.min(axis=1), index=days).clip(upper=MAX_HOLIDAY_DAYS)
    return flight_dates.map(distance)


def build_panel(series: pd.DataFrame) -> pd.DataFrame:
    """One row per trip per quoted day, with the price and the features known that day.

    The weekday of the search is left out: models are fitted per lead, so it is fixed by
    the departure weekday.
    """
    series = series.reset_index(drop=True)
    panel = pd.concat([series, _trip_history(series)], axis=1)

    route_median = series.groupby([*ROUTE_KEYS, "searchDate"])["price"].transform("median")
    panel["vs_route_today"] = series["price"] / route_median
    panel["log_route_median"] = np.log(route_median)
    panel["route_change_7d"] = _route_change_7d(series, route_median)
    panel["vs_neighbors"] = series["price"] / _neighbor_median(series)
    panel["days_to_holiday"] = _days_to_holiday(series["flightDate"])
    panel["next_airline_gap"] = series["next_airline_fare"] / series["price"]
    panel["log_distance"] = np.log(series["distance"].where(series["distance"] > 0))
    panel["cheapest_nonstop"] = series["cheapest_nonstop"].astype(float)
    panel["seats_remaining"] = series["seats_remaining"].astype(float)
    panel["n_flights"] = series["n_flights"].astype(float)
    panel["n_airlines"] = series["n_airlines"].astype(float)
    panel["flight_weekday"] = pd.Categorical(series["flightDate"].dt.day_of_week, categories=range(7))
    panel["cheapest_airline"] = pd.Categorical(series["cheapest_airline"].astype(str))

    return panel.set_index([*TRIP_KEYS, LEAD]).sort_index()[["price", *FEATURES]]
