# Ticket Prices

Decides when to buy a plane ticket from historical fare quotes. This first stage prepares
the data and backtests simple buying rules, which any later model has to beat.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
```

## Data

[Flight Prices](https://www.kaggle.com/datasets/dilwong/flightprices) by Dillon Wong (CC BY 4.0):
Expedia one-way quotes for 16 US airports, searched 2022-04-16 to 2022-10-05. It is one
31 GB CSV. The download needs a Kaggle login.

```bash
kaggle datasets download dilwong/flightprices -p data/raw --unzip
```

## Run

Run from the project root. `-I` stops Python from importing code from the current directory.

```bash
.venv/Scripts/python -I -m ticket_prices.cli all
```

`all` runs `convert`, `checks`, `series` and `backtest` in order. The model has three more
steps, run in this order:

```bash
.venv/Scripts/python -I -m ticket_prices.cli features
.venv/Scripts/python -I -m ticket_prices.cli tune
.venv/Scripts/python -I -m ticket_prices.cli final
```

`final` refuses to run twice, because a test rerun after seeing its results stops being a
fair test. `--allow-rerun` overrides that for a deliberate rerun.

Spirit and Frontier are excluded by default. `--all-airlines` keeps them and `--nonstop-only`
keeps nonstop fares. Both build separate files, so pass them to every step. `backtest` also
accepts `--start-lead`, `--end-lead` and `--split-date`.

Results go to `reports/`:

| File | Contents |
|---|---|
| `data_checks.json` | Row counts, lead times, gaps, and whether flights repeat across search days |
| `booking_curve.png`, `booking_curve.csv` | Fare by days before departure, training trips only |
| `booking_curve_by_route.csv` | The same curve for each route |
| `backtest_summary.csv` | Dollar results for each rule on test trips |
| `backtest_setup.json` | Split date, trip counts, and the leads chosen from training |
| `model_tuning.csv` | Saving per model type, margin and tuning fold |
| `model_summary.csv` | Final test: dollar results for the models and baselines |
| `model_summary_one_day_delay.csv` | The same test when each purchase goes through a day late |
| `model_setup.json` | Chosen settings, buying days, and the bad-case price check |
| `model_feature_importance.csv` | Share of LightGBM gain per feature |

## Method

- **Price series:** the cheapest non-basic-economy, all-coach fare per route, departure date
  and search date. Spirit, Frontier and JetBlue have no rows flagged as basic economy, so
  their lowest fares stay in the series even when they exclude bags or other extras.
  Spirit and Frontier are excluded for that reason, while JetBlue is kept.
- **Trip:** a traveller starts shopping 30 days out and must buy by 1 day out. Trips need
  quotes on both of those days and on at least 80% of the days in between.
- **Split:** every training quote comes before the split date and every test quote on or
  after it. Trips that straddle the split are dropped.
- **Rules:** buy on the first day, buy on the last day, best fixed lead overall, and best
  fixed lead per route. A perfect-hindsight buyer sets the upper limit.

## Model

Each day, the model estimates what the traveller will end up paying by waiting, as a ratio
of today's price, and the rule waits only if that beats a safety margin.

- **Features:** only what is known on that day. `tests/test_features.py` rewrites future
  quotes and checks that no earlier feature changes.
- **Training:** one model per day before departure, fitted backward from the last buying day.
  The target is the price the rule really pays from the next day on. Costs passed back to
  earlier days use out-of-fold predictions, so in-sample decisions don't flatter waiting.
- **Models:** LightGBM, plus a NumPy ridge regression as a simple benchmark. scikit-learn
  isn't used because Windows Application Control blocked one of its compiled files here.
- **Tuning:** model type and margin come from two time-ordered folds before the final split.
- **Pass mark:** the 95% confidence interval on saving against buying on the first day stays
  above $0, both for same-day purchases and with a one-day purchase delay.

## Status

The model did not pass. On the final test (9,040 trips, 39 departure dates), LightGBM saved
$6.62 per trip against buying on the first day (95% CI $4.32 to $8.86), but lost $11.29
(CI -$14.50 to -$8.34) with a one-day purchase delay. The result can't tell whether the
dips it buys on could really be booked. That test has now been used, so a changed model
needs new data for a fair test.

## Dips near departure

Does a departure's cheapest fare sometimes fall in the last days before it leaves? The `dips`
step answers that on the Expedia series and on the fares scraped by `germany-flight-fares/`
(the scraper is its own repository, so pull it first for newer data):

```bash
.venv/Scripts/python -I -m ticket_prices.cli dips
```

It needs `series` to have run for Expedia. `--scraped-dir`, `--min-drop-pct` (default 5%) and
`--min-drop-abs` (default 3 currency units) change where it reads and what counts as a dip. It
writes to `reports/dips/`. `dips` is not part of `all`.

- **Dip:** inside a window of days before departure, the first quote is the reference and a
  later quote at least 5% and 3 units lower is a dip. A departure needs most of the window's
  quotes, including the last one.
- **Placebo windows:** the same measure further from departure. Dips are concentrated near
  departure only if the final window's rate is clearly higher than theirs.
- **Rebound:** whether the next quote is back at or above the reference. A dip that doesn't
  rebound could still be caught one day late.

| File | Contents |
|---|---|
| `{expedia,ryanair}_dip_rates.csv/.png` | Dip share, last-day-drop share, dip size and rebound share per window, at 2%, 5% and 10% |
| `{expedia,ryanair}_daily_change.csv/.png` | Share of fares that fell, rose or stayed, from one day to the next, by days to departure |
| `{expedia,ryanair}_departures.csv` | One row per departure in the final window |
| `ryanair_same_flight_dip_rates.csv` | The same rates on quotes that stay on one flight number |
| `ryanair_dip_rates_by_origin.csv` | Final-window dip share by origin airport |
| `google_flights_cases.csv`, `google_flights_intraday_spread.csv` | Price paths of the few Google departures tracked to the end, and how far the three daily runs differ |
| `dips_summary.json` | Headline rates, counts and the snapshot dates used |

**Findings so far:** dips are rarer near departure, not more common. Expedia (2022, 27,427
departures) has a 5% dip in 28% of final-week windows (7-1 days out), against 41-43% two to
four weeks out. Ryanair (Nov 2026, 277 departures) has them in 15% of final-week windows
(6-1 days out), against 48% about six weeks out. Fares step up near departure (at 5 days and 1
day out on Expedia), and a dip that does appear in the final week is often temporary: 38% of
Expedia's and 33% of Ryanair's rebound the next day.

The Ryanair result is preliminary. It covers 6 days of snapshots (Nov 4-9, 2026), so each
window holds only about three departure dates, all in one week of November. Its intervals
resample routes and ignore shocks shared by a day, so they are too narrow. Google Flights has
only 5 departures tracked to the last day, too few for a rate. Rerun the step as snapshots
accumulate.

`pyarrow` can't load here because Windows Application Control blocks its DLL, so `cli.py`
reads parquet through DuckDB when that happens. Four tests in `test_prepare_and_checks.py` fail
for the same reason.

## Dip-timing model

Predicts, on any shopping day, whether a fare you could still buy appears below today's
price before departure (the "if" forecast) and on which day (the "when" forecast). It is a
separate model from the buy-or-wait one: it shares only the Expedia data and features. It needs
`series` and `features` to have run.

```bash
.venv/Scripts/python -I -m ticket_prices.cli dip-tune
.venv/Scripts/python -I -m ticket_prices.cli dip-final
```

`dip-final` refuses to run twice, like `final` (`--allow-rerun` overrides). `--min-drop-pct`
and `--min-drop-abs` set what counts as a dip, and `dip-final` checks that `dip-tune` used the
same values.

- **Decision:** a trip (route and departure date) on one shopping day from 30 to 2 days out,
  with its price that day.
- **Catchable dip:** a later day whose quote is at least 5% and 3 units below the decision's
  price, where the price a buyer pays one quoted day later is also that far below. A dip that
  rebounds the next day doesn't count. The last buying day has no later day, so it keeps its
  own price. A later day with no quote is not a dip.
- **Model:** one LightGBM classifier gives the chance that the first catchable dip lands on
  day `k`, given none landed earlier. Inputs are the 20 features from `features.py` at the
  decision, plus the decision's lead and the days ahead. Training rows run from the day after
  the decision up to and including the first dip. "If" is one minus the chance of no dip on
  any later day, and "when" is the chance each day is the first dip.
- **Baseline:** the training frequency of a first dip for each (decision lead, later day) pair.
- **Tuning:** leaves (7, 15, 31) and rounds (100, 200, 400) on the same two time-ordered folds
  as the buy-or-wait model. The score is the mean of the "if" and "when" log-loss gains over
  the baseline.
- **Test:** the same split date as the buy-or-wait model. Intervals resample departure dates.
- **Pass mark** (set before `dip-final` was run): the 95% interval for the log-loss gain over
  the baseline stays above 0 for both "if" and "when", and in every one of ten equal-sized
  bins the predicted chance of a dip is within 5 points of the observed share.

| File (`reports/dip_model/`) | Contents |
|---|---|
| `tuning.csv` | Log-loss gains per setting and fold |
| `summary.csv` | Final metrics for the model and baseline, with intervals on the gains |
| `calibration.csv`, `calibration.png` | Predicted and observed chance of a dip in ten bins |
| `by_lead.csv` | The "if" forecast at each decision lead |
| `feature_importance.csv` | Share of LightGBM gain per input |
| `setup.json` | Split date, thresholds, chosen setting, counts and the pass-mark check |

**Result: did not pass.** Tuning chose 31 leaves and 100 rounds. On the final test (9,040 trips,
39 departure dates, 251,163 decisions) the model beat the baseline but its probabilities were
far too high:

| Test metric | Model | Baseline | Gain (95% CI) |
|---|---|---|---|
| "If" log loss | 0.596 | 0.622 | 0.027 (0.003 to 0.051) |
| "When" log loss | 1.261 | 1.287 | 0.026 (0.002 to 0.051) |
| "If" AUC | 0.771 | 0.677 | |
| "When" hit in top 5 days | 66.4% | 66.0% | 0.3 points (-0.1 to 0.7) |

Both log-loss gains are above 0, but the calibration condition fails: the model's predicted
chance of a dip is above the observed share in all ten bins, by 3 to 30 points (the baseline's
gaps are 5 to 21). The model ranks decisions well but gets the level wrong, and it doesn't
place the day better than the baseline.

The cause is a lower dip rate in the test period. 44.5% of training decisions saw a catchable
dip, with 42% to 47% in each of May, June and July departures. In the test, which covers
departures from late August to early October, it was 28.6%. The tuning folds sit right after the
training data and showed a gain of 0.14 in log loss, against 0.03 on the test. One summer of
data can't separate a seasonal change from drift, so the model needs more than one season of
quotes to be tested for it.

### Walk-forward recalibration

Can a weekly correction of the probability level fix the over-prediction? `dip-walk` replays 13
Mondays from 2022-06-06 to 2022-08-29 and scores the week after each. **This is development
evidence, not a test:** the replay covers the period the final test used, and it was run once
with settings fixed in advance.

```bash
.venv/Scripts/python -I -m ticket_prices.cli dip-walk
```

It needs `dip-tune` to have run, and writes to `reports/dip_model/walk_forward/` (`summary.csv`,
`by_week.csv`, `calibration.csv`, `calibration.png`, `setup.json`).

Each Monday `W`, using only quotes before it:

1. The model and the baseline are refitted on trips that finished before `W`.
2. A correction `logit(h') = a + b * logit(h)` is fitted by logistic regression on trips still
   being shopped for. It uses the days whose outcome became known in the last 28 days (also tried
   with 14), predicted by the model just fitted. A day's outcome is known on the next quoted day,
   because the catchable test needs that day's price. The correction is either a shift (`b = 1`)
   or a shift and scale, and the baseline gets the same treatment.
3. Decisions made in `[W, W+7)` are scored on their full outcome.

`tests/test_dip_walk.py` rewrites every quote dated on or after `W` and checks that the rows and
labels used for recalibration don't change.

**Result: the success criteria were not met.** They were set in advance: pooled over the 13 weeks,
the predicted chance in every one of ten bins within 5 points of the observed share, and the
interval for the log-loss gain over the equally corrected baseline above 0. On 540,319 scored
decisions:

| Variant | "If" log loss (model / baseline) | "If" gain (95% CI) | Largest calibration gap |
|---|---|---|---|
| Weekly refit only | 0.494 / 0.623 | 0.129 (0.113 to 0.143) | 9.6 points |
| Shift, 28 days (headline) | 0.500 / 0.616 | 0.116 (0.103 to 0.127) | 16.1 points |
| Shift and scale, 28 days | 0.495 / 0.617 | 0.122 (0.108 to 0.135) | 14.6 points |
| Shift and scale, 14 days | 0.492 / 0.615 | 0.123 (0.110 to 0.135) | 13.5 points |

The "when" gains match these (0.12 to 0.13), and the model's AUC is 0.83 against 0.68. The weekly
refit alone narrows the calibration gap from the final test's 30 points, but not to 5, and
recalibration makes the pooled result no better:

- **The model's edge fades in August.** Its "if" log-loss gain over the baseline is 0.15 to 0.20
  in each week from June 6 to August 1, then 0.085, 0.017, 0.004 and 0.014 in the last four weeks.
  The observed dip rate fell from 52% to about 28% over the same weeks, while the raw model kept
  predicting about 40%.
- **Recalibration helps late and hurts early.** In the last four weeks the corrected model beats
  the raw one (log loss 0.516 to 0.538 against 0.529 to 0.593), but in June and July it is worse,
  for example 0.593 against 0.489 in the week of June 27. Fitted shifts were positive (up to +0.64)
  until July and only turned negative in August (down to -0.44).
- **Why it lags.** Trips still being shopped for have only their far-lead outcomes known, so the
  window mostly measures far-lead hazards. Near-lead hazards fell first. At the July 25 refit
  the window rows were calibrated (predicted 0.067 against 0.069 observed at 20 to 30 days ahead),
  while in the scored week the model over-predicted rows within 10 days (0.030 against 0.016).
  By August 22 the drift was visible in the window too, and the correction turned negative. A single
  shift is also too coarse, because the drift differs by days ahead.
- **Window length makes no difference.** The 14-day and 28-day versions are within 0.001 to 0.002.

### Lagged recalibration

The first replay corrected from trips still being shopped for, which show only far-lead outcomes.
`dip-walk-lag` corrects from trips that have already departed, which show outcomes at every lead.
Each Monday `W` the model and baseline are fitted on trips that departed on or before `W - lag`,
and the correction is fitted on trips that departed in `(W - lag, W]`, which the model never saw.
It is one shift for each band of days to departure (1-7, 8-14, 15-21, 22-29), and a band with
under 100 dips or non-dips takes the all-rows shift. It replays 9 Mondays from 2022-07-04 to
2022-08-29 (401,674 scored decisions) and writes to `reports/dip_model/walk_lagged/`. **This is
development evidence, a second look at weeks the earlier replay and the final test already used.**

```bash
.venv/Scripts/python -I -m ticket_prices.cli dip-walk-lag
```

**Result: the success criteria were not met.** They were set in advance for the headline variant
(`lag28_band`): every one of ten calibration bins within 5 points, a log-loss gain over the equally
corrected baseline with an interval above 0, and a gain over the plain weekly refit with an
interval above 0. It met the second only:

| Variant | "If" log loss | Gain over baseline (95% CI) | Gain over weekly refit (95% CI) | Largest calibration gap |
|---|---|---|---|---|
| `refit` (trained to `W`) | 0.498 | 0.119 (0.098 to 0.137) | | 13.6 points |
| `lag28_raw` | 0.526 | 0.105 (0.086 to 0.122) | -0.028 (-0.031 to -0.024) | 15.0 points |
| `lag28_shift` | 0.524 | 0.086 (0.068 to 0.102) | -0.026 (-0.033 to -0.019) | 15.9 points |
| `lag28_band` (headline) | 0.513 | 0.094 (0.077 to 0.109) | -0.015 (-0.021 to -0.008) | 16.2 points |
| `lag14_band` | 0.499 | 0.100 (0.085 to 0.113) | -0.001 (-0.009 to +0.008) | 14.2 points |

- **The bands see the near-lead drift.** The 1-7 day shift is negative in every week and grows
  from -0.15 to -0.88 (28-day lag), while the 15-29 day shifts were positive in July (up to
  +0.67). A single shift averages these out, which is why `lag28_shift` is worse than `lag28_band`.
- **The lag costs more than the correction recovers.** Training 28 days behind makes the model
  0.028 worse than the weekly refit, and the band shifts win back about half of that. With a
  14-day lag the correction recovers all of the cost, tying the refit ("if" gain -0.001),
  with a small gain on the "when" forecast (0.009, CI 0.001 to 0.017).
- **It helps in August and hurts in July, as before.** `lag14_band` beats the refit in each of
  the last four weeks by 0.011 to 0.061 in log loss, and is worse in July (0.483 against 0.459 in the
  week of July 4).
- **Calibration still fails by 14 to 16 points.** The dip rate was falling by departure date, and
  the scored trips depart 1 to 30 days after `W`, while the correction comes from trips that
  departed up to 28 days before it, so it trails the drift by about a month. In the week of
  August 1, 23% of decisions saw a dip while every variant predicted 38% to 39%.

## Tests

```bash
.venv/Scripts/python -m pytest
```
