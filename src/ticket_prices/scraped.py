"""Load the fare files written by the germany-flight-fares scraper.

The result has the same columns as the Expedia daily series (origin, destination, flightDate,
searchDate, days_to_departure, price), so the dip analysis runs on either.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TRIP_KEYS = ["origin", "destination", "flightDate"]
DAY_KEYS = [*TRIP_KEYS, "searchDate"]
COLUMNS = [
    *DAY_KEYS,
    "days_to_departure",
    "price",
    "flight_number",
    "airline",
    "stops",
    "snapshot_at",
]


def _stamp_from_name(name: str) -> pd.Timestamp:
    """The run time in a file name: `2026-11-04T0120Z`, or `2026-11-03` for the first files."""
    if len(name) == 10:
        return pd.Timestamp(name, tz="UTC")
    return pd.to_datetime(name, format="%Y-%m-%dT%H%MZ", utc=True)


def load_scraped_fares(data_dir: Path, source: str, skip_legacy: bool = False) -> pd.DataFrame:
    """Every fare row of one source, one row per run, route and departure.

    The first files are named by date alone and have no `snapshot_at`, so it is filled from the
    file name. `skip_legacy` drops them, which Ryanair needs: its first file holds the cheapest
    fare per week, not per day.
    """
    frames = []
    for path in sorted(Path(data_dir).glob(f"fares/{source}/*/*.csv.gz")):
        name = path.name.removesuffix(".csv.gz")
        if skip_legacy and len(name) == 10:
            continue
        frame = pd.read_csv(path)
        stamp = _stamp_from_name(name)
        recorded = frame["snapshot_at"] if "snapshot_at" in frame else pd.Series(pd.NaT, index=frame.index)
        frame["snapshot_at"] = pd.to_datetime(recorded, utc=True).fillna(stamp)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No {source} fare files under {Path(data_dir) / 'fares' / source}")

    fares = pd.concat(frames, ignore_index=True)
    fares["flightDate"] = pd.to_datetime(fares["departure_date"])
    fares["searchDate"] = pd.to_datetime(fares["snapshot_date"])
    fares["days_to_departure"] = (fares["flightDate"] - fares["searchDate"]).dt.days
    return fares[COLUMNS].sort_values([*DAY_KEYS, "snapshot_at"]).reset_index(drop=True)


def daily_series(fares: pd.DataFrame) -> pd.DataFrame:
    """One price per route, departure and search day, from that day's last run.

    This matches how the scraper's own daily summary is built, so days stay comparable. If a
    route has several flights in a run, the cheapest one is kept.
    """
    last_run = fares.groupby("searchDate")["snapshot_at"].transform("max")
    latest = fares[fares["snapshot_at"] == last_run]
    return latest.sort_values([*DAY_KEYS, "price"]).drop_duplicates(DAY_KEYS).reset_index(drop=True)


def same_flight_only(series: pd.DataFrame) -> pd.DataFrame:
    """Keep quotes for each departure's most common flight number.

    A fall in the cheapest fare can just be a different flight becoming the cheapest. Prices
    that stay on one flight number can't be explained that way.
    """
    modal = series.groupby(TRIP_KEYS)["flight_number"].agg(
        lambda numbers: numbers.mode().iloc[0] if numbers.notna().any() else np.nan
    )
    modal.name = "modal_flight"
    joined = series.merge(modal, left_on=TRIP_KEYS, right_index=True)
    return joined[joined["flight_number"] == joined["modal_flight"]].drop(columns="modal_flight")


def missing_last_quote(series: pd.DataFrame, end_lead: int = 1) -> dict:
    """Departures whose last buying day was searched but which have no quote that day.

    These left the results early (sold out or taken off sale). Leaving them out of a dip
    analysis would quietly favour departures that stayed on sale, so they are counted.
    """
    trips = series.groupby(TRIP_KEYS)["days_to_departure"].agg(last_lead="min")
    searched = series["searchDate"]
    due_date = trips.index.get_level_values("flightDate") - pd.Timedelta(days=end_lead)
    due = (due_date >= searched.min()) & (due_date <= searched.max())
    vanished = due & (trips["last_lead"].to_numpy() > end_lead)
    return {
        "departures_due": int(due.sum()),
        "missing_at_end_lead": int(vanished.sum()),
        "share_missing": float(vanished.sum() / due.sum()) if due.sum() else float("nan"),
    }
