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

## Tests

```bash
.venv/Scripts/python -m pytest
```
