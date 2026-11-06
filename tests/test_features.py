import numpy as np
import pandas as pd
import pytest

from ticket_prices.features import FEATURES, build_panel


def quote(flight, lead, price, origin="JFK", destination="LAX", **extra):
    flight = pd.Timestamp(flight)
    return {
        "origin": origin,
        "destination": destination,
        "flightDate": flight,
        "searchDate": flight - pd.Timedelta(days=lead),
        "days_to_departure": lead,
        "price": float(price),
        "n_flights": 10,
        "n_airlines": 2,
        "next_airline_fare": price * 1.1,
        "cheapest_airline": "DL",
        "seats_remaining": 5,
        "cheapest_nonstop": True,
        "distance": 2475.0,
        **extra,
    }


def test_trip_history_features_match_hand_calculation():
    prices = {10: 100, 9: 100, 8: 80, 6: 120, 5: 120}  # no quote at lead 7
    panel = build_panel(pd.DataFrame([quote("2022-06-20", lead, p) for lead, p in prices.items()]))
    trip = panel.droplevel(["origin", "destination", "flightDate"])

    assert np.isnan(trip.loc[10, "change_1d"])
    assert trip.loc[8, "change_1d"] == pytest.approx(0.8)
    assert trip.loc[8, "vs_trip_low"] == 1
    assert trip.loc[8, "vs_trip_high"] == pytest.approx(0.8)
    assert trip.loc[6, "vs_trip_low"] == pytest.approx(1.5)
    assert trip.loc[6, "vs_trip_first"] == pytest.approx(1.2)
    # On the missing day the latest known price (80) carries, so lead 6 compares with it.
    assert trip.loc[6, "change_1d"] == pytest.approx(1.5)
    assert trip.loc[5, "change_3d"] == pytest.approx(1.5)
    assert trip.loc[10, "days_since_change"] == 0
    assert trip.loc[9, "days_since_change"] == 1
    assert trip.loc[8, "days_since_change"] == 0
    assert trip.loc[5, "days_since_change"] == 1


def test_same_day_features_use_other_departures_and_past_route_prices():
    rows = []
    for offset, price in enumerate([100, 200, 300]):
        flight = pd.Timestamp("2022-06-20") + pd.Timedelta(days=offset)
        search = pd.Timestamp("2022-06-01")
        rows.append(quote(flight, (flight - search).days, price))
        week_ago = search - pd.Timedelta(days=7)
        rows.append(quote(flight, (flight - week_ago).days, price / 2))
    panel = build_panel(pd.DataFrame(rows))
    row = panel.xs(pd.Timestamp("2022-06-20"), level="flightDate").xs(19, level="days_to_departure")

    assert row["price"].iloc[0] == 100
    # On 06-01 the two neighbours cost 200 and 300, and the route median is 200.
    assert row["vs_neighbors"].iloc[0] == pytest.approx(100 / 250)
    assert row["vs_route_today"].iloc[0] == pytest.approx(0.5)
    assert row["route_change_7d"].iloc[0] == pytest.approx(2)


def test_days_to_holiday_counts_from_departure_date():
    series = pd.DataFrame([quote("2022-07-04", 5, 100), quote("2022-07-07", 5, 100), quote("2022-08-20", 5, 100)])
    panel = build_panel(series)
    days = panel["days_to_holiday"].droplevel(["origin", "destination", "days_to_departure"])
    assert days[pd.Timestamp("2022-07-04")] == 0
    assert days[pd.Timestamp("2022-07-07")] == 3
    assert days[pd.Timestamp("2022-08-20")] == 14  # capped


def test_no_feature_changes_when_only_the_future_changes():
    """The leakage guard: rewriting every quote after a date can't alter earlier features."""
    rng = np.random.default_rng(0)
    rows = []
    for route in ["LAX", "ORD", "BOS"]:
        for flight in pd.date_range("2022-06-01", periods=40):
            for lead in range(1, 31):
                if rng.random() < 0.9:
                    rows.append(quote(flight, lead, rng.uniform(100, 500), destination=route))
    series = pd.DataFrame(rows)
    cutoff = pd.Timestamp("2022-06-15")

    changed = series.copy()
    future = changed["searchDate"] > cutoff
    changed.loc[future, "price"] *= rng.uniform(0.2, 3, future.sum())
    changed.loc[future, "next_airline_fare"] *= 2
    changed.loc[future, "seats_remaining"] = 1

    original = build_panel(series)
    rewritten = build_panel(changed)
    flights = original.index.get_level_values("flightDate")
    leads = original.index.get_level_values("days_to_departure")
    past = (flights - pd.to_timedelta(leads, unit="D")) <= cutoff

    assert past.sum() > 500 and (~past).sum() > 500
    pd.testing.assert_frame_equal(original.loc[past, ["price", *FEATURES]], rewritten.loc[past, ["price", *FEATURES]])
    # Guard the guard: later rows really did change.
    assert not original.loc[~past, "vs_trip_low"].equals(rewritten.loc[~past, "vs_trip_low"])
