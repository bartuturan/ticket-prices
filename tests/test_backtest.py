import numpy as np
import pandas as pd
import pytest

from ticket_prices.backtest import (
    Config,
    best_fixed_lead,
    default_split_date,
    policy_prices,
    split_trips,
    summarize,
    trip_curves,
)
from ticket_prices.curves import booking_curve, plot_booking_curve

CFG = Config(start_lead=5, end_lead=1, min_route_trips=2, bootstrap_samples=200)


def quotes(flight, prices, origin="JFK", destination="LAX"):
    """prices maps days_to_departure to price."""
    flight = pd.Timestamp(flight)
    return [
        {
            "origin": origin,
            "destination": destination,
            "flightDate": flight,
            "searchDate": flight - pd.Timedelta(days=lead),
            "days_to_departure": lead,
            "price": price,
            "n_flights": 1,
        }
        for lead, price in prices.items()
    ]


def series_of(*trips):
    return pd.DataFrame([quote for trip in trips for quote in trip])


def curves_of(*trips):
    return trip_curves(series_of(*trips), CFG)


FULL = {5: 100, 4: 90, 3: 80, 2: 110, 1: 120}


def test_trip_curves_require_both_ends_and_enough_coverage():
    complete = quotes("2022-06-10", FULL)
    no_last_day = quotes("2022-06-11", {5: 100, 4: 90, 3: 80, 2: 110})
    no_first_day = quotes("2022-06-12", {4: 90, 3: 80, 2: 110, 1: 120})
    one_gap = quotes("2022-06-13", {5: 100, 4: 90, 2: 110, 1: 120})
    two_gaps = quotes("2022-06-14", {5: 100, 2: 110, 1: 120})

    curves = curves_of(complete, no_last_day, no_first_day, one_gap, two_gaps)

    kept = curves.index.get_level_values("flightDate").strftime("%m-%d").tolist()
    assert kept == ["06-10", "06-13"]  # coverage of 4/5 passes the 0.8 threshold, 3/5 fails
    assert list(curves.columns) == [5, 4, 3, 2, 1]


def test_split_keeps_all_train_quotes_before_all_test_quotes():
    flights = pd.date_range("2022-06-01", "2022-07-01")
    curves = curves_of(*[quotes(day, FULL) for day in flights])
    split = pd.Timestamp("2022-06-15")

    train, test = split_trips(curves, split, CFG)

    last_train_quote = train.index.get_level_values("flightDate").max() - pd.Timedelta(days=CFG.end_lead)
    first_test_quote = test.index.get_level_values("flightDate").min() - pd.Timedelta(days=CFG.start_lead)
    assert last_train_quote < split <= first_test_quote
    assert last_train_quote == pd.Timestamp("2022-06-14")
    assert first_test_quote == split
    # Trips quoted on both sides of the split belong to neither set.
    assert len(train) + len(test) < len(curves)


def test_missing_quote_slips_to_the_next_day_not_an_older_price():
    train = curves_of(quotes("2022-06-01", FULL), quotes("2022-06-02", FULL))
    # Training says buy at lead 3. The test trip has no quote that day, and the older
    # price at lead 4 is a bargain, which a forward fill would wrongly grant.
    test = curves_of(quotes("2022-07-01", {5: 100, 4: 50, 2: 110, 1: 120}))

    prices, leads = policy_prices(train, test, CFG)

    assert leads["overall"] == 3
    assert prices["best_fixed_lead"].iloc[0] == 110
    assert prices["hindsight"].iloc[0] == 50


def test_rules_are_chosen_from_training_trips_only():
    train = curves_of(quotes("2022-06-01", FULL), quotes("2022-06-02", FULL))
    test_trip = {5: 100, 4: 40, 3: 200, 2: 110, 1: 120}  # test's best day is lead 4
    test = curves_of(quotes("2022-07-01", test_trip))

    prices, leads = policy_prices(train, test, CFG)

    assert leads["overall"] == 3
    assert prices["best_fixed_lead"].iloc[0] == 200


def test_route_rule_uses_each_route_and_falls_back_to_overall():
    jfk_cheap_late = {5: 100, 4: 100, 3: 100, 2: 60, 1: 100}
    bos_cheap_early = {5: 100, 4: 50, 3: 100, 2: 100, 1: 100}
    train = curves_of(
        quotes("2022-06-01", jfk_cheap_late),
        quotes("2022-06-02", jfk_cheap_late),
        quotes("2022-06-03", jfk_cheap_late),
        quotes("2022-06-01", bos_cheap_early, origin="BOS"),
        quotes("2022-06-02", bos_cheap_early, origin="BOS"),
        # One trip is below min_route_trips, so this route uses the overall best.
        quotes("2022-06-01", {5: 100, 4: 100, 3: 10, 2: 100, 1: 100}, origin="SFO"),
    )
    test_prices = {5: 100, 4: 41, 3: 42, 2: 43, 1: 44}
    test = curves_of(
        quotes("2022-07-01", test_prices),
        quotes("2022-07-01", test_prices, origin="BOS"),
        quotes("2022-07-01", test_prices, origin="SFO"),
    )

    prices, leads = policy_prices(train, test, CFG)

    assert leads["by_route"] == {("BOS", "LAX"): 4, ("JFK", "LAX"): 2}
    by_origin = prices["best_lead_by_route"].groupby(level="origin").first()
    assert by_origin["JFK"] == 43
    assert by_origin["BOS"] == 41
    assert by_origin["SFO"] == test_prices[leads["overall"]]


def test_best_fixed_lead_weights_trips_equally_not_by_price():
    # An expensive trip is cheapest at lead 4, two cheap trips at lead 2.
    train = curves_of(
        quotes("2022-06-01", {5: 1000, 4: 900, 3: 1000, 2: 1000, 1: 1000}),
        quotes("2022-06-02", {5: 100, 4: 100, 3: 100, 2: 80, 1: 100}),
        quotes("2022-06-03", {5: 100, 4: 100, 3: 100, 2: 80, 1: 100}),
    )
    assert best_fixed_lead(train, CFG) == 2


def test_summary_dollars_match_hand_calculation():
    train = curves_of(quotes("2022-06-01", FULL), quotes("2022-06-02", FULL))
    test = curves_of(
        quotes("2022-07-01", {5: 100, 4: 90, 3: 70, 2: 110, 1: 120}),  # lead 3 saves 30
        quotes("2022-07-02", {5: 100, 4: 90, 3: 110, 2: 60, 1: 120}),  # lead 3 costs 10, hindsight 60
    )

    prices, _ = policy_prices(train, test, CFG)
    summary = summarize(prices, CFG)
    rule = summary.loc["best_fixed_lead"]

    assert rule["trips"] == 2
    assert rule["mean_price"] == 90
    assert rule["mean_saving"] == 10
    assert rule["mean_saving_pct"] == 10
    assert rule["share_of_possible_saving"] == pytest.approx(10 / 35)
    assert rule["share_trips_cheaper"] == 0.5
    assert rule["share_trips_dearer"] == 0.5
    assert summary.loc["hindsight", "mean_saving"] == 35
    assert summary.loc["buy_first_day", "mean_saving"] == 0


def test_confidence_interval_resamples_departure_dates():
    train = curves_of(quotes("2022-06-01", FULL), quotes("2022-06-02", FULL))
    # Same saving on every date: resampling dates gives no spread at all.
    same = {5: 100, 4: 90, 3: 70, 2: 110, 1: 120}
    test = curves_of(*[quotes(day, same) for day in pd.date_range("2022-07-01", periods=5)])

    summary = summarize(policy_prices(train, test, CFG)[0], CFG)

    assert summary.loc["best_fixed_lead", "saving_ci_low"] == pytest.approx(30)
    assert summary.loc["best_fixed_lead", "saving_ci_high"] == pytest.approx(30)


def test_default_split_is_between_first_and_last_search():
    series = series_of(*[quotes(day, FULL) for day in pd.date_range("2022-04-20", "2022-10-05")])
    split = default_split_date(series)
    assert series["searchDate"].min() < split < series["searchDate"].max()


def test_booking_curve_and_plot(tmp_path):
    curves = curves_of(quotes("2022-06-01", FULL), quotes("2022-06-02", {5: 200, 4: 180, 3: 160, 2: 220, 1: 240}))

    curve = booking_curve(curves, CFG)

    assert curve.loc[5, "median"] == 1
    assert curve.loc[3, "median"] == pytest.approx(0.8)
    assert curve.loc[3, "share_below_first_day"] == 1
    assert np.array_equal(curve["trips"].to_numpy(), [2, 2, 2, 2, 2])

    out = tmp_path / "curve.png"
    plot_booking_curve(curve, 2, CFG, out)
    assert out.stat().st_size > 0
