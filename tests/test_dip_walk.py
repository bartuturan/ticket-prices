import numpy as np
import pandas as pd
import pytest

from ticket_prices.dip_model import Decisions
from ticket_prices.dip_walk import (
    apply_band_shifts,
    apply_recalibration,
    fit_band_shifts,
    fit_recalibration,
    recalibration_rows,
    recent_departure_rows,
    settled_columns,
    variant_names,
)
from ticket_prices.dips import DipConfig
from ticket_prices.features import FEATURES

CFG = DipConfig(min_drop_pct=0.05, min_drop_abs=3.0)
LEADS = np.arange(8, 0, -1)
WEEK = pd.Timestamp("2022-07-18")
NAN = np.nan


def make_decisions(flights, decision_leads):
    """One decision per trip and lead in `decision_leads`."""
    trip, lead, flight = [], [], []
    for index, date in enumerate(flights):
        for value in decision_leads:
            trip.append(index)
            lead.append(value)
            flight.append(date)
    return Decisions(
        np.array(trip, dtype=np.int32),
        np.array(lead),
        np.array(flight, dtype="datetime64[ns]"),
        np.zeros((len(trip), len(FEATURES)), dtype=np.float32),
    )


def kept_pairs(rows):
    keep = rows.keep
    leads = rows.decisions.lead[rows.hazard.row[keep]]
    return sorted(zip(leads.tolist(), rows.hazard.k[keep].tolist()))


def run(paths, flights, decision_leads, window_days=28):
    prices = np.array(paths, dtype=float)
    decisions = make_decisions(flights, decision_leads)
    return recalibration_rows(prices, LEADS, settled_columns(prices), decisions, WEEK, window_days, CFG, 1)


def test_a_days_outcome_is_known_on_the_next_quoted_day():
    settled = settled_columns(np.array([[100.0, NAN, 90.0, 70.0, 60.0]]))

    # Column 1 has no quote, so it settles itself. The last column settles itself too.
    assert list(settled[0]) == [2, 1, 3, 4, 4]


def test_only_outcomes_known_before_the_week_are_kept():
    # Departs 3 days after the week starts. Quotes exist up to lead 4, dated the day before.
    flight = WEEK + pd.Timedelta(days=3)
    path = [100, 100, 100, 100, 100, 100, 100, 100]
    rows = run([path], [flight], [8])

    # Lead 5's label needs the quote at lead 4 (known). Lead 4's needs lead 3, dated on the week.
    assert kept_pairs(rows) == [(8, 5), (8, 6), (8, 7)]


def test_days_after_the_first_dip_are_dropped():
    flight = WEEK + pd.Timedelta(days=3)
    path = [100, 100, 90, 90, 90, 90, 90, 90]  # first catchable dip at lead 6
    rows = run([path], [flight], [8])

    assert kept_pairs(rows) == [(8, 6), (8, 7)]
    assert list(rows.hazard.y[rows.keep]) == [False, True]


def test_finished_trips_are_not_used():
    on_the_week = WEEK
    before = WEEK - pd.Timedelta(days=1)
    path = [100] * 8
    rows = run([path, path], [on_the_week, before], [8])

    assert rows.keep.sum() == 0


def test_the_window_keeps_only_recent_outcomes():
    flight = WEEK + pd.Timedelta(days=3)
    path = [100, 100, 100, 100, 100, 100, 100, 100]

    # Lead 7's label is known three days before the week, lead 6's two days before, lead 5's one.
    narrow = run([path], [flight], [8], window_days=2)
    assert kept_pairs(narrow) == [(8, 5), (8, 6)]


def test_changing_quotes_dated_on_or_after_the_week_changes_nothing():
    rng = np.random.default_rng(0)
    flights = [WEEK + pd.Timedelta(days=d) for d in range(-2, 12)]
    paths = 100 + rng.normal(scale=12, size=(len(flights), len(LEADS)))
    paths[rng.random(paths.shape) < 0.1] = np.nan
    paths[:, -1] = np.where(np.isnan(paths[:, -1]), 100.0, paths[:, -1])

    changed = paths.copy()
    for index, flight in enumerate(flights):
        dates = np.array([flight - pd.Timedelta(days=int(lead)) for lead in LEADS])
        later = dates >= WEEK
        changed[index, later] = rng.normal(loc=100, scale=40, size=later.sum())
        changed[index, later & (rng.random(len(LEADS)) < 0.3)] = np.nan
    changed[:, -1] = np.where(np.isnan(changed[:, -1]), 100.0, changed[:, -1])

    base = run(paths, flights, list(range(8, 1, -1)))
    other = run(changed, flights, list(range(8, 1, -1)))

    assert base.keep.sum() > 0
    assert (base.keep == other.keep).all()
    assert (base.hazard.row[base.keep] == other.hazard.row[other.keep]).all()
    assert (base.hazard.k[base.keep] == other.hazard.k[other.keep]).all()
    assert (base.hazard.y[base.keep] == other.hazard.y[other.keep]).all()

    h = np.full(base.keep.sum(), 0.3)
    assert fit_recalibration(h, base.hazard.y[base.keep], min_events=1) == fit_recalibration(
        h, other.hazard.y[other.keep], min_events=1
    )


def test_the_fit_recovers_a_known_shift_and_scale():
    rng = np.random.default_rng(1)
    z = rng.normal(scale=1.5, size=200_000)
    h = 1 / (1 + np.exp(-z))
    truth = 1 / (1 + np.exp(-(-0.8 + 0.6 * z)))
    y = rng.random(len(z)) < truth

    a, b = fit_recalibration(h, y, scale=True)

    assert a == pytest.approx(-0.8, abs=0.05)
    assert b == pytest.approx(0.6, abs=0.03)


def test_a_shift_alone_matches_the_observed_rate():
    rng = np.random.default_rng(2)
    z = rng.normal(size=50_000)
    h = 1 / (1 + np.exp(-z))
    y = rng.random(len(z)) < 0.2

    a, b = fit_recalibration(h, y, scale=False)

    assert b == 1.0
    assert apply_recalibration(h, a, b).mean() == pytest.approx(y.mean(), abs=1e-4)


def test_too_few_events_leave_the_probabilities_alone():
    h = np.full(1000, 0.4)
    y = np.zeros(1000, dtype=bool)
    y[:5] = True

    assert fit_recalibration(h, y, scale=True) == (0.0, 1.0)
    assert apply_recalibration(h, 0.0, 1.0) == pytest.approx(h)


def test_variant_names():
    assert variant_names((28, 14)) == ["raw", "shift_28", "shift_scale_28", "shift_14", "shift_scale_14"]


def run_recent(paths, flights, decision_leads, lag_days):
    prices = np.array(paths, dtype=float)
    decisions = make_decisions(flights, decision_leads)
    return recent_departure_rows(prices, LEADS, decisions, WEEK, lag_days, CFG, 1)


def test_recent_rows_come_only_from_trips_that_departed_in_the_lag_window():
    day = pd.Timedelta(days=1)
    path = [100, 100, 90, 90, 90, 90, 90, 90]  # first catchable dip at lead 6
    flights = [WEEK - 20 * day, WEEK - 14 * day, WEEK - 13 * day, WEEK, WEEK + day]
    rows = run_recent([path] * 5, flights, [8], lag_days=14)

    kept = pd.to_datetime(rows.decisions.flight_date[rows.hazard.row[rows.keep]])
    # Departing exactly lag days back belongs to the training side, and the open trip isn't used.
    assert set(kept) == {WEEK - 13 * day, WEEK}
    # Each decision keeps its at-risk days only: lead 7, then the dip at lead 6, nothing after.
    assert kept_pairs(rows) == [(8, 6), (8, 6), (8, 7), (8, 7)]


def test_recent_rows_ignore_later_quotes_and_trips_still_open():
    rng = np.random.default_rng(3)
    flights = [WEEK + pd.Timedelta(days=d) for d in range(-16, 10)]
    paths = 100 + rng.normal(scale=12, size=(len(flights), len(LEADS)))
    paths[rng.random(paths.shape) < 0.1] = np.nan
    changed = paths.copy()
    for index, flight in enumerate(flights):
        dates = np.array([flight - pd.Timedelta(days=int(lead)) for lead in LEADS])
        later = dates >= WEEK
        changed[index, later] = rng.normal(loc=100, scale=40, size=later.sum())
        if flight > WEEK:
            changed[index] = rng.normal(loc=100, scale=40, size=len(LEADS))
    for table in (paths, changed):
        table[:, -1] = np.where(np.isnan(table[:, -1]), 100.0, table[:, -1])

    decision_leads = list(range(8, 1, -1))
    base = run_recent(paths, flights, decision_leads, 14)
    other = run_recent(changed, flights, decision_leads, 14)

    assert base.keep.sum() > 0
    assert (base.keep == other.keep).all()
    assert (base.hazard.row[base.keep] == other.hazard.row[other.keep]).all()
    assert (base.hazard.k[base.keep] == other.hazard.k[other.keep]).all()
    assert (base.hazard.y[base.keep] == other.hazard.y[other.keep]).all()

    h = np.full(base.keep.sum(), 0.3)
    one = fit_band_shifts(h, base.hazard.y[base.keep], base.hazard.k[base.keep], min_events=1)
    two = fit_band_shifts(h, other.hazard.y[other.keep], other.hazard.k[other.keep], min_events=1)
    assert (one[0] == two[0]).all() and one[1] == two[1]


def test_band_shifts_recover_each_band_and_fall_back_when_sparse():
    rng = np.random.default_rng(4)
    n = 400_000
    k = rng.integers(1, 22, n)  # nothing in the 22-29 band
    h = rng.uniform(0.05, 0.6, n)
    truth = np.array([-0.8, -0.4, 0.0])
    band = np.digitize(k, [8, 15])
    y = rng.random(n) < 1 / (1 + np.exp(-(np.log(h / (1 - h)) + truth[band])))

    shifts, pooled = fit_band_shifts(h, y, k)

    assert shifts[:3] == pytest.approx(truth, abs=0.05)
    assert shifts[3] == pooled  # no rows, so the all-rows shift

    # A band with a few dozen rows, under the 100-event minimum, also falls back.
    few = fit_band_shifts(np.append(h, np.full(50, 0.5)), np.append(y, np.ones(50, dtype=bool)), np.append(k, np.full(50, 25)))
    assert few[0][3] == few[1]


def test_band_shifts_are_applied_to_the_matching_band():
    h = np.full(4, 0.5)  # logit 0, so the output is the shift itself
    k = np.array([3, 10, 17, 25])
    shifts = np.array([-1.0, 0.0, 1.0, 2.0])

    out = apply_band_shifts(h, k, shifts)

    assert out == pytest.approx(1 / (1 + np.exp(-shifts)))
    # A day outside every band takes the default.
    assert apply_band_shifts(np.array([0.5]), np.array([40]), shifts, default=0.5)[0] == pytest.approx(1 / (1 + np.exp(-0.5)))
