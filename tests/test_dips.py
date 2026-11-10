import numpy as np
import pandas as pd
import pytest

from ticket_prices.dips import (
    DipConfig,
    daily_change_by_lead,
    dip_rates,
    google_cases,
    late_dips,
    plot_daily_change,
    plot_dip_rates,
    ratio_ci,
    wide_curves,
)

CFG = DipConfig(min_drop_pct=0.05, min_drop_abs=3.0, bootstrap_samples=200)
LEADS = [7, 6, 5, 4, 3, 2, 1]


def curves_of(*paths):
    """Each path is a list of 7 prices for leads 7 down to 1, with None for no quote."""
    index = pd.MultiIndex.from_tuples(
        [("AAA", "BBB", pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)) for i in range(len(paths))],
        names=["origin", "destination", "flightDate"],
    )
    return pd.DataFrame([[np.nan if p is None else p for p in path] for path in paths], index=index, columns=LEADS)


def dips_of(*paths, window=(7, 1), min_obs=None):
    return late_dips(curves_of(*paths), window, CFG, min_obs)


def test_dip_that_rebounds_is_flagged_and_marked_transient():
    result = dips_of([100, 100, 100, 80, 100, 100, 100]).iloc[0]

    assert result["dipped"]
    assert result["min_price"] == 80
    assert result["min_lead"] == 4
    assert result["drop"] == 20
    assert result["drop_pct"] == pytest.approx(0.2)
    assert result["rebounded"]


def test_dip_that_holds_does_not_rebound():
    result = dips_of([100, 100, 100, 80, 80, 85, 90]).iloc[0]

    assert result["dipped"]
    assert not result["rebounded"]
    assert not result["dip_at_end"]


def test_dip_on_the_last_day_cannot_be_tested_for_a_rebound():
    result = dips_of([100, 100, 100, 100, 100, 100, 70]).iloc[0]

    assert result["dipped"]
    assert result["dip_at_end"]
    assert not result["rebounded"]
    assert result["last_day_drop"]


def test_rising_prices_are_not_a_dip():
    result = dips_of([100, 100, 105, 110, 120, 130, 150]).iloc[0]

    assert not result["dipped"]
    assert result["min_price"] == 100
    assert not result["last_day_drop"]


def test_small_falls_need_both_thresholds():
    # 4% is under min_drop_pct, 2 is under min_drop_abs even though 10% is over the percentage.
    four_percent = [100, 96, 100, 100, 100, 100, 100]
    two_euro_ten_percent = [20, 18, 20, 20, 20, 20, 20]
    big_enough = [100, 94, 100, 100, 100, 100, 100]

    result = dips_of(four_percent, two_euro_ten_percent, big_enough)

    assert result["dipped"].tolist() == [False, False, True]


def test_the_reference_is_the_first_quote_in_the_window():
    # No quote at lead 7, so the reference is lead 6, and 7 days of quotes aren't required.
    result = dips_of([None, 100, 100, 90, 100, 100, 100]).iloc[0]

    assert result["eligible"]
    assert result["ref_lead"] == 6
    assert result["ref_price"] == 100
    assert result["dipped"]


def test_departures_with_too_few_quotes_are_not_eligible():
    sparse = [100, None, None, None, None, None, 80]

    assert not dips_of(sparse).iloc[0]["eligible"]
    assert dips_of(sparse, min_obs=2).iloc[0]["eligible"]


def test_departures_not_quoted_at_the_window_end_are_not_eligible():
    watched_far_out = [100, 100, 100, 100, 100, None, None]  # never seen in the last two days

    assert not dips_of(watched_far_out).iloc[0]["eligible"]
    assert not dips_of(watched_far_out, min_obs=4).iloc[0]["eligible"]
    assert dips_of(watched_far_out, min_obs=4, window=(7, 3)).iloc[0]["eligible"]


def test_a_missing_day_inside_the_window_is_skipped_when_checking_the_rebound():
    result = dips_of([100, 100, 100, 80, None, 100, 100]).iloc[0]

    assert result["rebounded"]  # the next quote after lead 4 is lead 2, at 100


def test_window_only_looks_at_its_own_leads():
    path = [100, 100, 100, 100, 100, 100, 100]
    early_dip = [100, 70, 100, 100, 100, 100, 100]  # dip at lead 6

    final = dips_of(path, early_dip, window=(4, 1))
    early = dips_of(path, early_dip, window=(7, 5))

    assert final["dipped"].tolist() == [False, False]
    assert early["dipped"].tolist() == [False, True]


def test_dip_rates_per_window_and_placebo_comparison():
    flat = [100] * 7
    late_dip = [100, 100, 100, 100, 100, 80, 100]
    curves = curves_of(flat, flat, late_dip, late_dip)
    groups = curves.index.get_level_values("flightDate")

    rates = dip_rates(curves, {"final": (3, 1), "early": (7, 5)}, CFG, groups)

    assert rates.loc["final", "dip_rate"] == 0.5
    assert rates.loc["final", "departures"] == 4
    assert rates.loc["final", "rebound_share"] == 1
    assert rates.loc["final", "median_dip_pct"] == pytest.approx(0.2)
    assert rates.loc["early", "dip_rate"] == 0
    assert np.isnan(rates.loc["early", "median_dip_pct"])


def test_confidence_interval_resamples_groups_not_rows():
    flags = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)

    # Each group is all hits or all misses, so resampling groups gives real spread ...
    low, high = ratio_ci(flags, ["a", "a", "a", "a", "b", "b", "b", "b"], CFG)
    assert low == 0 and high == 1
    # ... and one share everywhere gives none.
    low, high = ratio_ci(np.ones(8), list("abcdefgh"), CFG)
    assert low == high == 1
    assert all(np.isnan(ratio_ci(flags, ["a"] * 8, CFG)))


def test_daily_change_counts_only_days_quoted_on_both_sides():
    curves = curves_of(
        [100, 100, 100, 100, 100, 100, 100],
        [100, 100, 100, 90, 100, 100, 100],  # falls 10% going from lead 5 to lead 4
        [100, 100, 100, None, 100, 100, 100],  # no quote at lead 4: no pair across it
    )

    change = daily_change_by_lead(curves, CFG)

    assert change.loc[4, "pairs"] == 2  # the third departure has no quote at lead 4
    assert change.loc[3, "pairs"] == 2  # ... nor a pair from 4 to 3
    assert change.loc[5, "pairs"] == 3
    assert change.loc[4, "share_down_big"] == pytest.approx(0.5)
    assert change.loc[3, "share_up_big"] == pytest.approx(0.5)
    assert change.loc[4, "median_log_change"] == pytest.approx(np.log(0.9) / 2)
    assert 7 not in change.index  # lead 7 has no earlier day


def test_wide_curves_takes_the_cheapest_quote_and_orders_leads_from_far_to_near():
    series = pd.DataFrame(
        {
            "origin": "AAA",
            "destination": "BBB",
            "flightDate": pd.Timestamp("2026-01-10"),
            "days_to_departure": [3, 2, 2, 1],
            "price": [100, 90, 80, 95],
        }
    )

    wide = wide_curves(series, 3)

    assert list(wide.columns) == [3, 2, 1]
    assert wide.iloc[0].tolist() == [100, 80, 95]


def test_google_cases_and_intraday_spread():
    def row(snapshot, lead, price, flight="2026-01-10"):
        flight = pd.Timestamp(flight)
        return {
            "origin": "FRA", "destination": "BER", "flightDate": flight,
            "searchDate": pd.Timestamp(snapshot).normalize(), "days_to_departure": lead,
            "snapshot_at": pd.Timestamp(snapshot, tz="UTC"), "price": price, "airline": "X", "stops": 0,
        }

    fares = pd.DataFrame(
        [
            row("2026-01-08T05:00", 2, 100), row("2026-01-08T13:00", 2, 110),
            row("2026-01-09T05:00", 1, 90),
            row("2026-01-08T05:00", 20, 50, flight="2026-01-28"),  # never reaches lead 1
        ]
    )

    paths, spread = google_cases(fares)

    assert len(paths) == 3 and set(paths["flightDate"]) == {pd.Timestamp("2026-01-10")}
    assert spread.loc[2, "departure_days"] == 1
    assert spread.loc[2, "mean_spread_pct"] == pytest.approx(0.1)


def test_plots_are_written(tmp_path):
    curves = curves_of([100] * 7, [100, 100, 100, 100, 100, 80, 100], [100, 90, 100, 100, 100, 100, 100])
    groups = curves.index.get_level_values("flightDate")
    windows = {"final": (3, 1), "early": (7, 5)}
    rates = dip_rates(curves, windows, CFG, groups)

    plot_dip_rates(rates, "final", "Dips", tmp_path / "rates.png")
    plot_daily_change(daily_change_by_lead(curves, CFG), "Changes", tmp_path / "change.png")

    assert (tmp_path / "rates.png").stat().st_size > 0
    assert (tmp_path / "change.png").stat().st_size > 0
