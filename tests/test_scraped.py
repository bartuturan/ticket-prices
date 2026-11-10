import gzip

import pandas as pd

from ticket_prices.scraped import (
    daily_series,
    load_scraped_fares,
    missing_last_quote,
    same_flight_only,
)

HEADER = "snapshot_date,source,origin,destination,departure_date,departure_time,airline,flight_number,stops,price,currency,snapshot_at"


def write_run(root, source, name, rows, stamp=""):
    folder = root / "fares" / source / "2026"
    folder.mkdir(parents=True, exist_ok=True)
    lines = [HEADER] + [
        f"{snap},{source},BER,PMI,{flight},10:00,Ryanair,{number},0,{price},EUR,{stamp}"
        for snap, flight, number, price in rows
    ]
    (folder / f"{name}.csv.gz").write_bytes(gzip.compress("\n".join(lines).encode()))


def test_loader_fills_snapshot_time_from_the_file_name_and_computes_leads(tmp_path):
    write_run(tmp_path, "ryanair", "2026-11-03", [("2026-11-03", "2026-11-09", "FR1", 30)])
    write_run(
        tmp_path, "ryanair", "2026-11-04T0517Z", [("2026-11-04", "2026-11-09", "FR1", 35)],
        stamp="2026-11-04T05:17:00Z",
    )

    fares = load_scraped_fares(tmp_path, "ryanair")

    assert fares["days_to_departure"].tolist() == [6, 5]
    assert fares["snapshot_at"].tolist() == [
        pd.Timestamp("2026-11-03", tz="UTC"),
        pd.Timestamp("2026-11-04T05:17", tz="UTC"),
    ]
    assert list(fares.columns[:5]) == ["origin", "destination", "flightDate", "searchDate", "days_to_departure"]


def test_loader_handles_a_first_file_with_no_snapshot_column(tmp_path):
    folder = tmp_path / "fares" / "google_flights" / "2026"
    folder.mkdir(parents=True)
    legacy = HEADER.removesuffix(",snapshot_at") + "\n2026-11-03,google_flights,FRA,BER,2026-11-10,08:30,Condor,,0,110,EUR\n"
    (folder / "2026-11-03.csv.gz").write_bytes(gzip.compress(legacy.encode()))

    fares = load_scraped_fares(tmp_path, "google_flights")

    assert fares["snapshot_at"].tolist() == [pd.Timestamp("2026-11-03", tz="UTC")]
    assert fares["days_to_departure"].tolist() == [7]


def test_loader_can_skip_the_first_files_named_by_date_alone(tmp_path):
    write_run(tmp_path, "ryanair", "2026-11-03", [("2026-11-03", "2026-11-09", "FR1", 30)])
    write_run(
        tmp_path, "ryanair", "2026-11-04T0517Z", [("2026-11-04", "2026-11-09", "FR1", 35)],
        stamp="2026-11-04T05:17:00Z",
    )

    fares = load_scraped_fares(tmp_path, "ryanair", skip_legacy=True)

    assert fares["price"].tolist() == [35]


def test_daily_series_keeps_the_last_run_of_each_day_and_the_cheapest_flight(tmp_path):
    write_run(
        tmp_path, "google_flights", "2026-11-04T0120Z", [("2026-11-04", "2026-11-10", "A", 100)],
        stamp="2026-11-04T01:20:00Z",
    )
    write_run(
        tmp_path, "google_flights", "2026-11-04T1259Z",
        [("2026-11-04", "2026-11-10", "A", 120), ("2026-11-04", "2026-11-10", "B", 90)],
        stamp="2026-11-04T12:59:00Z",
    )

    series = daily_series(load_scraped_fares(tmp_path, "google_flights"))

    assert series["price"].tolist() == [90]
    assert series["flight_number"].tolist() == ["B"]


def test_same_flight_only_drops_quotes_where_another_flight_was_cheapest(tmp_path):
    write_run(
        tmp_path, "ryanair", "2026-11-04T0517Z",
        [("2026-11-04", "2026-11-10", "FR1", 50)], stamp="2026-11-04T05:17:00Z",
    )
    write_run(
        tmp_path, "ryanair", "2026-11-05T0517Z",
        [("2026-11-05", "2026-11-10", "FR1", 55)], stamp="2026-11-05T05:17:00Z",
    )
    write_run(
        tmp_path, "ryanair", "2026-11-06T0517Z",
        [("2026-11-06", "2026-11-10", "FR2", 30)], stamp="2026-11-06T05:17:00Z",
    )
    series = daily_series(load_scraped_fares(tmp_path, "ryanair"))

    assert same_flight_only(series)["price"].tolist() == [50, 55]


def test_missing_last_quote_counts_departures_that_left_the_results_early(tmp_path):
    # Searches run on Nov 4 and Nov 5. Flight Nov 6 is due a quote on Nov 5 and has it;
    # flight Nov 5 is due one on Nov 4 and has it; flight Nov 6 (second route) is due and doesn't.
    write_run(tmp_path, "ryanair", "2026-11-04T0517Z", [
        ("2026-11-04", "2026-11-05", "FR1", 40),
        ("2026-11-04", "2026-11-06", "FR2", 40),
    ], stamp="2026-11-04T05:17:00Z")
    write_run(tmp_path, "ryanair", "2026-11-05T0517Z", [
        ("2026-11-05", "2026-11-06", "FR2", 45),
    ], stamp="2026-11-05T05:17:00Z")
    series = daily_series(load_scraped_fares(tmp_path, "ryanair"))
    series = pd.concat([series, series.assign(destination="FCO", flightDate=pd.Timestamp("2026-11-06")).head(1).assign(
        searchDate=pd.Timestamp("2026-11-04"), days_to_departure=2)])

    result = missing_last_quote(series)

    assert result == {"departures_due": 3, "missing_at_end_lead": 1, "share_missing": 1 / 3}
