import csv
from datetime import date

import pandas as pd
import pytest

from ticket_prices.checks import run_checks
from ticket_prices.prepare import build_daily_series, convert_csv

HEADER = [
    "legId", "searchDate", "flightDate", "startingAirport", "destinationAirport", "fareBasisCode",
    "travelDuration", "elapsedDays", "isBasicEconomy", "isRefundable", "isNonStop", "baseFare",
    "totalFare", "seatsRemaining", "totalTravelDistance", "segmentsAirlineCode", "segmentsCabinCode",
]


def row(leg, search, fare, basic="False", cabin="coach", nonstop="True", flight="2022-06-11", airline="DL", seats="5"):
    return [
        leg, search, flight, "JFK", "LAX", "ABC", "PT5H", "0", basic, "False", nonstop,
        fare, fare, seats, "2475", airline, cabin,
    ]


@pytest.fixture
def itineraries(tmp_path):
    rows = [
        row("a", "2022-06-01", "200"),
        row("a", "2022-06-01", "150"),  # second fare, same flight and day
        row("a", "2022-06-02", "180"),
        row("a", "2022-06-04", "180"),  # no search on 06-03
        row("b", "2022-06-01", "100", basic="True"),
        row("c", "2022-06-01", "90", cabin="coach||first"),
        row("d", "2022-06-01", "170", cabin="coach||coach", nonstop="False"),
        row("e", "2022-06-01", "not-a-number"),
        row("f", "2022-06-12", "120"),  # searched after departure
    ]
    csv_path = tmp_path / "itineraries.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    out = tmp_path / "itineraries.parquet"
    convert_csv(csv_path, out, tmp_path / "tmp")
    return out


def test_convert_types_columns_and_keeps_bad_values_as_nulls(itineraries):
    df = pd.read_parquet(itineraries)
    assert len(df) == 9
    assert df["days_to_departure"].iloc[0] == 10
    assert df.loc[df["legId"] == "e", "totalFare"].isna().all()
    assert not itineraries.with_suffix(".partial").exists()


def test_daily_series_takes_cheapest_comparable_fare(itineraries, tmp_path):
    out = tmp_path / "series.parquet"
    build_daily_series(itineraries, out, tmp_path / "tmp")
    df = pd.read_parquet(out).sort_values("searchDate").reset_index(drop=True)

    # Basic economy (100), first class (90), the bad fare and the late search are all out.
    searched = pd.to_datetime(df["searchDate"]).dt.date.tolist()
    assert searched == [date(2022, 6, 1), date(2022, 6, 2), date(2022, 6, 4)]
    assert df["price"].tolist() == [150, 180, 180]
    assert df["n_flights"].tolist() == [2, 1, 1]


def test_daily_series_nonstop_only(itineraries, tmp_path):
    out = tmp_path / "series.parquet"
    build_daily_series(itineraries, out, tmp_path / "tmp", nonstop_only=True)
    df = pd.read_parquet(out).sort_values("searchDate")
    assert df["n_flights"].tolist() == [1, 1, 1]


def test_checks_report_repeats_gaps_and_bad_rows(itineraries):
    report = run_checks(itineraries)
    assert report["rows"] == 9
    assert report["distinct_flights"] == 6
    assert report["null_counts"]["totalFare"] == 1
    assert report["bad_rows"]["searched_after_departure"] == 1
    assert "2022-06-03" in report["missing_search_days"]
    assert report["max_days_to_departure"] == 10
    # Of 5 flights with a valid fare, only flight "a" appears on 2+ days and changes price.
    assert report["flights_with_valid_fare"] == 5
    assert report["search_days_per_flight"]["share_seen_2_plus_days"] == pytest.approx(0.2)
    assert report["share_of_flights_whose_price_changed"] == pytest.approx(0.2)


def test_daily_series_excludes_airlines_on_any_segment_and_adds_competition(tmp_path):
    rows = [
        row("a", "2022-06-01", "200", airline="DL", seats="4"),
        row("b", "2022-06-01", "230", airline="UA"),
        row("c", "2022-06-01", "260", airline="UA"),
        row("d", "2022-06-01", "90", airline="NK"),
        row("e", "2022-06-01", "95", airline="DL||F9", cabin="coach||coach"),
        row("f", "2022-06-02", "300", airline="AA"),
    ]
    csv_path = tmp_path / "itineraries.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    parquet = tmp_path / "itineraries.parquet"
    convert_csv(csv_path, parquet, tmp_path / "tmp")

    out = tmp_path / "series.parquet"
    build_daily_series(parquet, out, tmp_path / "tmp")
    df = pd.read_parquet(out).sort_values("searchDate").reset_index(drop=True)

    first = df.iloc[0]
    assert first["price"] == 200
    assert first["cheapest_airline"] == "DL"
    assert first["seats_remaining"] == 4
    assert first["n_flights"] == 3
    assert first["n_airlines"] == 2
    assert first["next_airline_fare"] == 230
    assert first["distance"] == 2475
    assert pd.isna(df.iloc[1]["next_airline_fare"])  # only one airline that day

    everyone = tmp_path / "everyone.parquet"
    build_daily_series(parquet, everyone, tmp_path / "tmp", exclude_airlines=())
    assert pd.read_parquet(everyone).sort_values("searchDate")["price"].iloc[0] == 90
