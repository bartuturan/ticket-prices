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

## Tests

```bash
.venv/Scripts/python -m pytest
```
