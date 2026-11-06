import numpy as np
import pandas as pd
import pytest

from ticket_prices.backtest import Config, policy_prices, split_trips, trip_curves
from ticket_prices.evaluate import Fold, choose, delayed_prices, final_test, tune
from ticket_prices.features import FEATURES, NUMERIC, build_panel
from ticket_prices.policy import MODEL_KINDS, Policy, fit_policy, rows_at_lead, simulate, training_rows

CFG = Config(start_lead=4, end_lead=1, bootstrap_samples=100)


class Constant:
    """A stand-in model predicting a fixed ratio, so results can be worked out by hand."""

    def __init__(self, value):
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


def panel_of(trips):
    """trips maps departure date to {lead: price}. Features are filled with a constant."""
    rows = []
    for flight, prices in trips.items():
        for lead, price in prices.items():
            rows.append(("JFK", "LAX", pd.Timestamp(flight), lead, float(price)))
    panel = pd.DataFrame(rows, columns=["origin", "destination", "flightDate", "days_to_departure", "price"])
    for name in FEATURES:
        panel[name] = 0.0
    return panel.set_index(["origin", "destination", "flightDate", "days_to_departure"]).sort_index()


def trip_index(*flights):
    return pd.MultiIndex.from_tuples(
        [("JFK", "LAX", pd.Timestamp(flight)) for flight in flights],
        names=["origin", "destination", "flightDate"],
    )


TRIPS = {f"2022-06-{day:02d}": {4: 100, 3: 80, 2: 90, 1: 120} for day in range(1, 7)}


def test_target_is_what_the_rule_pays_not_the_future_low():
    # A model that always says "cheaper later" makes the rule wait to the last day, so
    # the target at every lead is the last-day price, even though lead 3 was cheaper.
    policy = fit_policy(panel_of(TRIPS), CFG, 0.0, lambda: Constant(0.5), min_rows=1)
    assert policy.targets[2].iloc[0] == pytest.approx(120 / 90)
    assert policy.targets[3].iloc[0] == pytest.approx(120 / 80)
    assert policy.targets[4].iloc[0] == pytest.approx(120 / 100)


def test_target_uses_the_next_days_purchase_when_the_rule_buys():
    policy = fit_policy(panel_of(TRIPS), CFG, 0.0, lambda: Constant(1.5), min_rows=1)
    assert policy.targets[2].iloc[0] == pytest.approx(120 / 90)
    assert policy.targets[3].iloc[0] == pytest.approx(90 / 80)
    assert policy.targets[4].iloc[0] == pytest.approx(80 / 100)


def test_targets_carry_past_days_without_a_quote():
    trips = {**TRIPS, "2022-06-07": {4: 100, 2: 90, 1: 120}}
    policy = fit_policy(panel_of(trips), CFG, 0.0, lambda: Constant(1.5), min_rows=1)
    gappy = trip_index("2022-06-07")[0]
    assert gappy not in policy.targets[3].index
    assert policy.targets[4].loc[gappy] == pytest.approx(90 / 100)


def test_trip_without_a_last_day_quote_is_not_used_for_training():
    trips = {**TRIPS, "2022-06-07": {4: 100, 3: 80, 2: 90}}
    policy = fit_policy(panel_of(trips), CFG, 0.0, lambda: Constant(1.0), min_rows=1)
    assert trip_index("2022-06-07")[0] not in policy.targets[4].index


def test_margin_requires_a_bigger_predicted_saving_before_waiting():
    panel = panel_of({"2022-07-01": {4: 100, 3: 50, 2: 90, 1: 120}})
    trips = trip_index("2022-07-01")
    predicts_4pct_drop = {lead: Constant(0.96) for lead in (2, 3, 4)}

    no_margin = simulate(panel, trips, Policy(0.0, predicts_4pct_drop), CFG)
    with_margin = simulate(panel, trips, Policy(0.05, predicts_4pct_drop), CFG)

    assert no_margin["paid"].iloc[0] == 120 and no_margin["bought_at"].iloc[0] == 1
    assert with_margin["paid"].iloc[0] == 100 and with_margin["bought_at"].iloc[0] == 4


def test_simulate_stops_at_first_buy_skips_gaps_and_forces_the_last_day():
    panel = panel_of(
        {
            "2022-07-01": {4: 100, 3: 80, 2: 90, 1: 120},
            "2022-07-02": {4: 100, 2: 90, 1: 120},
            "2022-07-03": {4: 100, 3: 80, 2: 90, 1: 120},
        }
    )
    trips = trip_index("2022-07-01", "2022-07-02", "2022-07-03")
    models = {4: Constant(0.5), 3: Constant(1.5), 2: Constant(0.5)}

    result = simulate(panel, trips, Policy(0.0, models), CFG)

    assert result["paid"].tolist() == [80, 120, 80]
    assert result["bought_at"].tolist() == [3, 1, 3]


def test_lead_with_too_few_rows_means_buy():
    panel = panel_of({"2022-07-01": {4: 100, 3: 80, 2: 90, 1: 120}})
    policy = fit_policy(panel, CFG, 0.0, lambda: Constant(0.5), min_rows=5)
    assert policy.models == {}
    result = simulate(panel, trip_index("2022-07-01"), policy, CFG)
    assert result["bought_at"].iloc[0] == 4


def test_training_rows_end_before_the_split():
    panel = panel_of(TRIPS)
    rows = training_rows(panel, pd.Timestamp("2022-06-03"), CFG)
    flights = rows.index.get_level_values("flightDate")
    assert flights.max() == pd.Timestamp("2022-06-03")  # last buying day is 06-02


def synthetic_series(seed=0, signal=True):
    """Half the trips drop by 40% near departure, flagged by a visible competitor gap."""
    rng = np.random.default_rng(seed)
    rows = []
    for route in ["LAX", "ORD", "BOS", "SFO"]:
        for flight in pd.date_range("2022-05-20", "2022-09-30"):
            level = rng.uniform(150, 350)
            dropping = rng.random() < 0.5
            for lead in range(1, 21):
                if signal:
                    price = level * (0.6 if dropping and lead <= 5 else 1.0)
                else:
                    price = level * (1 + 0.15 * (rng.random() < 0.5 and lead <= 5))
                price *= rng.lognormal(0, 0.005)
                rows.append(
                    {
                        "origin": "JFK",
                        "destination": route,
                        "flightDate": flight,
                        "searchDate": flight - pd.Timedelta(days=lead),
                        "days_to_departure": lead,
                        "price": price,
                        "n_flights": 10,
                        "n_airlines": 2,
                        "next_airline_fare": level * (1.5 if signal and dropping else 1.05),
                        "cheapest_airline": "DL",
                        "seats_remaining": 5,
                        "cheapest_nonstop": True,
                        "distance": 2000.0,
                    }
                )
    return pd.DataFrame(rows)


LONG = Config(start_lead=20, end_lead=1, bootstrap_samples=100)
SPLIT = pd.Timestamp("2022-07-20")


@pytest.mark.parametrize("kind", list(MODEL_KINDS))
def test_models_learn_a_real_signal_and_beat_the_baselines(kind):
    series = synthetic_series()
    panel = build_panel(series)

    result = final_test(panel, series, SPLIT, LONG, {kind: 0.02}, min_rows=50)
    summary = result["summary"]

    assert summary.loc[f"model_{kind}", "saving_ci_low"] > 0
    assert summary.loc[f"model_{kind}", "mean_saving"] > 0.7 * summary.loc["hindsight", "mean_saving"]
    assert result["purchases"][kind]["bought_at"].min() < LONG.start_lead  # it does wait sometimes


def test_lightgbm_with_a_margin_buys_early_when_there_is_nothing_to_learn():
    series = synthetic_series(signal=False)
    panel = build_panel(series)

    result = final_test(panel, series, SPLIT, LONG, {"lightgbm": 0.05}, min_rows=50)

    purchases = result["purchases"]["lightgbm"]
    assert (purchases["bought_at"] >= 15).mean() > 0.95
    assert result["summary"].loc["model_lightgbm", "mean_saving"] == pytest.approx(0, abs=2)


def test_tune_never_uses_trips_after_the_folds_and_choose_picks_best():
    series = synthetic_series()
    panel = build_panel(series)
    folds = (Fold("2022-06-15", "2022-07-15"), Fold("2022-07-01", "2022-08-01"))

    tuning = tune(panel, series, LONG, folds=folds, margins=(0.02, 0.1), kinds=("linear",), min_rows=50)

    assert len(tuning) == 4
    assert tuning["mean_saving"].max() > 0
    best = choose(tuning)
    per_setting = tuning.groupby("margin")["mean_saving"].mean()
    assert best["margin"] == per_setting.idxmax()


def test_delayed_prices_pay_the_next_quoted_day_and_keep_the_last_day():
    cfg = Config(start_lead=4, end_lead=1)
    index = trip_index("2022-07-01", "2022-07-02", "2022-07-03")
    curves = pd.DataFrame(
        {4: [100, 100, 100], 3: [80, np.nan, 80], 2: [90, 90, 90], 1: [120, 120, 120]}, index=index
    )
    purchases = {"linear": pd.DataFrame({"bought_at": [3, 4, 1]}, index=index)}

    prices = delayed_prices(curves, purchases, cfg)

    assert prices["model_linear"].tolist() == [90, 90, 120]  # the second trip skips the gap
    assert prices["buy_first_day"].tolist() == [80, 90, 80]
    assert prices["hindsight"].tolist() == [80, 90, 80]


def test_final_test_baselines_match_the_backtest():
    series = synthetic_series()
    panel = build_panel(series)
    result = final_test(panel, series, SPLIT, LONG, {"linear": 0.02}, min_rows=50)

    train, test = split_trips(trip_curves(series, LONG), SPLIT, LONG)
    baseline, _ = policy_prices(train, test, LONG)
    pd.testing.assert_series_equal(result["prices"]["buy_first_day"], baseline["buy_first_day"])
    assert len(rows_at_lead(result["train"], 5)) > 0
    assert len(NUMERIC) + 2 == len(FEATURES)
