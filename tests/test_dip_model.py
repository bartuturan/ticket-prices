import numpy as np
import pandas as pd
import pytest

from ticket_prices.dip_model import (
    CATEGORICAL_COLUMNS,
    MODEL_COLUMNS,
    Decisions,
    baseline_table,
    build_hazard,
    decision_scores,
    delayed_prices,
    design_matrix,
    fit_hazard,
    forecast,
    pass_mark,
    predict_hazard,
)
from ticket_prices.dips import DipConfig
from ticket_prices.features import FEATURES

CFG = DipConfig(min_drop_pct=0.05, min_drop_abs=3.0)
LEADS = np.array([5, 4, 3, 2, 1])
NAN = np.nan


def hazard_for(paths, decision_leads, trip=None):
    """Hazard rows for decisions on the given leads of each path (one decision list per path)."""
    prices = np.array(paths, dtype=float)
    trips, leads = [], []
    for index, wanted in enumerate(decision_leads):
        for lead in wanted:
            trips.append(index)
            leads.append(lead)
    return build_hazard(prices, LEADS, np.array(trips), np.array(leads), CFG, end_lead=1)


def first_dip_days(hazard):
    """First catchable dip lead for each decision, or None."""
    n = hazard.row.max() + 1
    days = [None] * n
    for row, k in zip(hazard.row[hazard.y], hazard.k[hazard.y]):
        days[row] = int(k)
    return days


def test_a_dip_that_holds_is_catchable():
    hazard = hazard_for([[100, 100, 90, 90, 90]], [[5]])

    assert first_dip_days(hazard) == [3]
    # Four later days exist. Only the days up to the first dip are at risk.
    assert list(hazard.k) == [4, 3, 2, 1]
    assert list(hazard.at_risk) == [True, True, False, False]


def test_a_dip_that_rebounds_the_next_day_is_not_catchable():
    # Lead 3 is 10% down, but the next quote is back at 100. The later days stay level.
    hazard = hazard_for([[100, 100, 90, 100, 100]], [[5]])

    assert first_dip_days(hazard) == [None]
    assert hazard.at_risk.all()


def test_a_dip_on_the_last_buying_day_counts_at_its_own_price():
    hazard = hazard_for([[100, 100, 100, 100, 90]], [[5]])

    assert first_dip_days(hazard) == [1]


def test_a_small_or_cheap_drop_is_not_a_dip():
    # 4% of 100, and 2 of 40 is 5% but under 3 units.
    hazard = hazard_for([[100, 96, 96, 96, 96], [40, 38, 38, 38, 38]], [[5], [5]])

    assert first_dip_days(hazard) == [None, None]


def test_a_missing_quote_day_cannot_be_the_dip():
    # No quote at lead 4. Lead 3 is the first day with a price, and it holds.
    hazard = hazard_for([[100, NAN, 90, 90, 90]], [[5]])

    assert first_dip_days(hazard) == [3]
    assert hazard.y[hazard.k == 4].sum() == 0


def test_the_delayed_price_skips_a_missing_day():
    delayed = delayed_prices(np.array([[100.0, 80.0, NAN, 70.0, 60.0]]))

    assert list(delayed[0]) == [80.0, 70.0, 70.0, 60.0, 60.0]


def test_a_dip_followed_by_a_missing_day_uses_the_next_quote():
    # Lead 3 is down, lead 2 has no quote, lead 1 is back at 100: the buyer pays 100.
    hazard = hazard_for([[100, 100, 90, NAN, 100]], [[5]])

    assert first_dip_days(hazard) == [None]


def test_each_decision_is_measured_from_its_own_price():
    hazard = hazard_for([[100, 100, 90, 90, 90]], [[5, 4, 3]])

    # From lead 4 the price is still 100 and falls at 3. From lead 3 (90) nothing is lower.
    assert first_dip_days(hazard) == [3, 3, None]


def test_the_label_ignores_everything_before_the_decision_day():
    quiet = [[100, 100, 90, 90, 90]]
    moved = [[250, 40, 90, 90, 90]]  # changed leads 5 and 4, which come before a decision at lead 3
    base = hazard_for(quiet, [[3]])
    changed = hazard_for(moved, [[3]])

    assert list(base.y) == list(changed.y)
    assert list(base.at_risk) == list(changed.at_risk)


def test_a_decision_on_the_last_buying_day_has_no_rows():
    hazard = hazard_for([[100, 100, 90, 90, 90]], [[1]])

    assert len(hazard) == 0


def test_daily_chances_chain_into_if_and_when():
    rng = np.random.default_rng(0)
    horizons = [4, 3, 1]
    row = np.repeat(np.arange(len(horizons)), horizons)
    h = rng.uniform(0.02, 0.4, size=row.size)

    p_any, none, first = forecast(row, h, len(horizons))

    assert p_any == pytest.approx(1 - none)
    for decision in range(len(horizons)):
        assert first[row == decision].sum() + none[decision] == pytest.approx(1)
    # The first day's chance is just its own daily chance, and the second needs the first to be missed.
    assert first[0] == pytest.approx(h[0])
    assert first[1] == pytest.approx((1 - h[0]) * h[1])


def test_the_baseline_table_reproduces_the_frequencies():
    # Decision lead 3, later day 2: four rows at risk, one of them a first dip.
    lead = np.array([3, 3, 3, 3, 3, 3])
    k = np.array([2, 2, 2, 2, 1, 1])
    y = np.array([1, 0, 0, 0, 0, 1], dtype=float)
    at_risk = np.array([True, True, True, True, True, False])

    table = baseline_table(lead, k, y, at_risk, max_lead=3, prior_rows=0.0)

    assert table[3, 2] == pytest.approx(0.25)
    assert table[3, 1] == pytest.approx(1e-6)  # one at-risk row and it was not a dip


def test_scores_match_a_hand_calculation():
    decisions = Decisions(
        trip=np.array([0], dtype=np.int32),
        lead=np.array([3]),
        flight_date=np.array(["2022-06-20"], dtype="datetime64[ns]"),
        features=np.zeros((1, len(FEATURES)), dtype=np.float32),
    )
    from ticket_prices.dip_model import Hazard

    hazard = Hazard(
        row=np.array([0, 0], dtype=np.int32),
        k=np.array([2, 1], dtype=np.int16),
        y=np.array([False, True]),
        at_risk=np.array([True, True]),
    )
    scores = decision_scores(decisions, hazard, np.array([0.5, 0.5]))

    assert scores["p_if"].iloc[0] == pytest.approx(0.75)
    assert scores["if_log_loss"].iloc[0] == pytest.approx(-np.log(0.75))
    assert scores["when_log_loss"].iloc[0] == pytest.approx(2 * np.log(2))
    assert bool(scores["any_dip"].iloc[0])


def test_design_matrix_carries_lead_and_days_ahead():
    from ticket_prices.dip_model import Hazard

    features = np.arange(2 * len(FEATURES), dtype=np.float32).reshape(2, -1)
    decisions = Decisions(
        np.array([0, 1], dtype=np.int32),
        np.array([10, 4]),
        np.array(["2022-06-20", "2022-06-21"], dtype="datetime64[ns]"),
        features,
    )
    hazard = Hazard(
        np.array([0, 0, 1], dtype=np.int32),
        np.array([9, 8, 3], dtype=np.int16),
        np.zeros(3, dtype=bool),
        np.ones(3, dtype=bool),
    )
    matrix = design_matrix(decisions, hazard)

    width = len(FEATURES)
    assert matrix.shape == (3, len(MODEL_COLUMNS))
    assert list(matrix[:, width]) == [10, 10, 4]
    assert list(matrix[:, width + 1]) == [1, 2, 1]
    assert (matrix[2, :width] == features[1]).all()
    assert [MODEL_COLUMNS[i] for i in CATEGORICAL_COLUMNS] == ["flight_weekday", "cheapest_airline"]


def test_the_model_fits_and_predicts_probabilities():
    from ticket_prices.dip_model import Hazard

    rng = np.random.default_rng(1)
    n = 600
    decisions = Decisions(
        np.arange(n, dtype=np.int32),
        np.full(n, 10),
        np.full(n, np.datetime64("2022-06-20")),
        rng.normal(size=(n, len(FEATURES))).astype(np.float32),
    )
    decisions.features[:, FEATURES.index("flight_weekday")] = rng.integers(0, 7, n)
    decisions.features[:, FEATURES.index("cheapest_airline")] = rng.integers(0, 3, n)
    hazard = Hazard(
        np.arange(n, dtype=np.int32),
        np.full(n, 5, dtype=np.int16),
        decisions.features[:, 0] > 0.5,
        np.ones(n, dtype=bool),
    )
    booster = fit_hazard(decisions, hazard, num_leaves=4, rounds=20, min_data_in_leaf=10)
    h = predict_hazard(booster, decisions, hazard)

    assert h.shape == (n,)
    assert ((h > 0) & (h < 1)).all()
    assert h[hazard.y].mean() > h[~hazard.y].mean()


def test_pass_mark_needs_gains_and_calibration():
    summary = pd.DataFrame(
        {"gain_ci_low": [0.01, 0.02]}, index=pd.Index(["if_log_loss", "when_log_loss"], name="metric")
    )
    good = pd.DataFrame({"gap": [0.01, -0.04]})
    bad = pd.DataFrame({"gap": [0.01, -0.08]})

    assert pass_mark(summary, good)["passed"]
    assert not pass_mark(summary, bad)["passed"]
    summary.loc["when_log_loss", "gain_ci_low"] = -0.001
    assert not pass_mark(summary, good)["passed"]
