"""Confirm the data can answer the question before anything is modelled on it."""

from pathlib import Path

import duckdb

from ticket_prices.prepare import quote_path

KEY_COLUMNS = ["legId", "searchDate", "flightDate", "origin", "destination", "totalFare"]
LEAD_BUCKETS = [(0, 0), (1, 7), (8, 14), (15, 30), (31, 45), (46, 60), (61, 90), (91, 100000)]


def run_checks(itineraries: Path) -> dict:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW it AS SELECT * FROM read_parquet({quote_path(itineraries)})")

    def one(query: str) -> tuple:
        return con.execute(query).fetchone()

    rows, legs, routes = one(
        "SELECT count(*), count(DISTINCT legId), count(DISTINCT (origin, destination)) FROM it"
    )
    first_search, last_search, first_flight, last_flight, max_lead = one(
        "SELECT min(searchDate), max(searchDate), min(flightDate), max(flightDate),"
        " max(days_to_departure) FROM it"
    )

    null_counts = dict(
        zip(KEY_COLUMNS, one(f"SELECT {', '.join(f'count(*) - count({c})' for c in KEY_COLUMNS)} FROM it"))
    )

    con.execute(
        """
        CREATE TEMP TABLE leg_daily AS
        SELECT legId, searchDate, min(totalFare) AS price, count(*) AS n_fares
        FROM it WHERE totalFare > 0 GROUP BY legId, searchDate
        """
    )
    median_days, p10_days, p90_days, multi_day, week_plus, price_moved, total_legs = one(
        """
        SELECT median(d), quantile_cont(d, 0.1), quantile_cont(d, 0.9),
            avg((d >= 2)::INT), avg((d >= 7)::INT), avg((prices >= 2)::INT), count(*)
        FROM (
            SELECT legId, count(*) AS d, count(DISTINCT price) AS prices
            FROM leg_daily GROUP BY legId
        )
        """
    )

    missing_search_days = [
        row[0].isoformat()
        for row in con.execute(
            """
            SELECT day::DATE FROM generate_series(
                (SELECT min(searchDate) FROM it), (SELECT max(searchDate) FROM it), INTERVAL 1 DAY
            ) AS days(day)
            WHERE day::DATE NOT IN (SELECT DISTINCT searchDate FROM it WHERE searchDate IS NOT NULL)
            ORDER BY 1
            """
        ).fetchall()
    ]

    rows_by_lead = {}
    for low, high in LEAD_BUCKETS:
        label = f"{low}" if low == high else f"{low}-{high}"
        rows_by_lead[label] = one(
            f"SELECT count(*) FROM it WHERE days_to_departure BETWEEN {low} AND {high}"
        )[0]

    return {
        "rows": rows,
        "distinct_flights": legs,
        "routes": routes,
        "search_dates": [str(first_search), str(last_search)],
        "flight_dates": [str(first_flight), str(last_flight)],
        "max_days_to_departure": max_lead,
        "null_counts": null_counts,
        "bad_rows": {
            "non_positive_fare": one("SELECT count(*) FROM it WHERE totalFare <= 0")[0],
            "searched_after_departure": one("SELECT count(*) FROM it WHERE days_to_departure < 0")[0],
        },
        "basic_economy_share": one("SELECT avg(isBasicEconomy::INT) FROM it")[0],
        "fares_per_flight_day": one("SELECT avg(n_fares) FROM leg_daily")[0],
        "search_days_per_flight": {
            "median": median_days,
            "p10": p10_days,
            "p90": p90_days,
            "share_seen_2_plus_days": multi_day,
            "share_seen_7_plus_days": week_plus,
        },
        "share_of_flights_whose_price_changed": price_moved,
        "flights_with_valid_fare": total_legs,
        "missing_search_days": missing_search_days,
        "rows_by_days_to_departure": rows_by_lead,
    }
