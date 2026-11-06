"""Turn the raw Expedia CSV into typed Parquet and a daily cheapest-fare series."""

from pathlib import Path

import duckdb

EXCLUDED_AIRLINES = ("NK", "F9")  # Spirit, Frontier


def _connect(temp_dir: Path) -> duckdb.DuckDBPyConnection:
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory = {quote_path(temp_dir)}")
    con.execute("SET preserve_insertion_order = false")
    return con


def quote_path(path: Path | str) -> str:
    escaped = str(path).replace("'", "''")
    return f"'{escaped}'"


def _copy_to_parquet(con: duckdb.DuckDBPyConnection, query: str, out: Path) -> None:
    """Write through a temp name so an interrupted run never leaves a partial file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial")
    con.execute(f"COPY ({query}) TO {quote_path(partial)} (FORMAT parquet, COMPRESSION zstd)")
    partial.replace(out)


def convert_csv(csv_path: Path, out: Path, temp_dir: Path) -> None:
    """Read every column as text and TRY_CAST, so bad values become NULLs we can count.

    A strict read could fail hours into a 31 GB file. run_checks reports the NULLs.
    """
    query = f"""
        SELECT
            legId,
            TRY_CAST(searchDate AS DATE) AS searchDate,
            TRY_CAST(flightDate AS DATE) AS flightDate,
            startingAirport AS origin,
            destinationAirport AS destination,
            fareBasisCode,
            TRY_CAST(isBasicEconomy AS BOOLEAN) AS isBasicEconomy,
            TRY_CAST(isRefundable AS BOOLEAN) AS isRefundable,
            TRY_CAST(isNonStop AS BOOLEAN) AS isNonStop,
            TRY_CAST(baseFare AS DOUBLE) AS baseFare,
            TRY_CAST(totalFare AS DOUBLE) AS totalFare,
            TRY_CAST(seatsRemaining AS INTEGER) AS seatsRemaining,
            TRY_CAST(totalTravelDistance AS INTEGER) AS totalTravelDistance,
            segmentsAirlineCode,
            segmentsCabinCode,
            date_diff('day', TRY_CAST(searchDate AS DATE), TRY_CAST(flightDate AS DATE))
                AS days_to_departure
        FROM read_csv({quote_path(csv_path)}, header = true, all_varchar = true)
    """
    _copy_to_parquet(_connect(temp_dir), query, out)


def build_daily_series(
    itineraries: Path,
    out: Path,
    temp_dir: Path,
    nonstop_only: bool = False,
    exclude_airlines: tuple[str, ...] = EXCLUDED_AIRLINES,
) -> None:
    """One row per route, departure date and search date: the cheapest comparable fare.

    Basic economy and premium cabins are excluded, otherwise a change in which fare
    product happens to be on sale would look like a price change. Spirit and Frontier are
    excluded by default because their fares are not flagged as basic economy but may leave
    out bags. Any itinerary with a segment on an excluded airline is dropped.
    """
    nonstop_filter = "AND isNonStop" if nonstop_only else ""
    airline_filter = ""
    if exclude_airlines:
        codes = "|".join(exclude_airlines)
        airline_filter = f"AND NOT regexp_matches(segmentsAirlineCode, '(^|\\|)({codes})(\\||$)')"
    keys = "origin, destination, flightDate, searchDate, days_to_departure"
    query = f"""
        WITH comparable AS (
            SELECT
                {keys}, legId, totalFare, seatsRemaining, isNonStop, totalTravelDistance,
                split_part(segmentsAirlineCode, '||', 1) AS airline
            FROM read_parquet({quote_path(itineraries)})
            WHERE totalFare > 0
                AND days_to_departure >= 0
                AND isBasicEconomy = false
                AND regexp_full_match(segmentsCabinCode, 'coach(\\|\\|coach)*')
                {nonstop_filter}
                {airline_filter}
        ),
        airline_fares AS (
            SELECT {keys}, airline, min(totalFare) AS fare
            FROM comparable
            GROUP BY {keys}, airline
        ),
        competition AS (
            SELECT
                {keys},
                count(*) AS n_airlines,
                list_sort(list(fare))[2] AS next_airline_fare
            FROM airline_fares
            GROUP BY {keys}
        ),
        daily AS (
            SELECT
                {keys},
                min(totalFare) AS price,
                count(DISTINCT legId) AS n_flights,
                arg_min(airline, totalFare) AS cheapest_airline,
                arg_min(seatsRemaining, totalFare) AS seats_remaining,
                arg_min(isNonStop, totalFare) AS cheapest_nonstop,
                median(totalTravelDistance) AS distance
            FROM comparable
            GROUP BY {keys}
        )
        SELECT * FROM daily JOIN competition USING ({keys})
    """
    _copy_to_parquet(_connect(temp_dir), query, out)
