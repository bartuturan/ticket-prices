"""Run the pipeline: python -I -m ticket_prices.cli STEP. See README.md for the steps."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from ticket_prices.backtest import (
    Config,
    default_split_date,
    policy_prices,
    split_trips,
    summarize,
    trip_curves,
)
from ticket_prices.checks import run_checks
from ticket_prices.curves import booking_curve, booking_curve_by_route, plot_booking_curve
from ticket_prices.evaluate import bad_case_check, choose, feature_importance, final_test, tune
from ticket_prices.features import build_panel
from ticket_prices.prepare import EXCLUDED_AIRLINES, build_daily_series, convert_csv

CSV = Path("data/raw/itineraries.csv")
ITINERARIES = Path("data/processed/itineraries.parquet")
TEMP = Path("data/tmp")
REPORTS = Path("reports")


def series_path(args: argparse.Namespace, prefix: str = "daily_cheapest") -> Path:
    parts = [prefix]
    if args.all_airlines:
        parts.append("all_airlines")
    if args.nonstop_only:
        parts.append("nonstop")
    return Path("data/processed") / f"{'_'.join(parts)}.parquet"


def load_series(path: Path) -> pd.DataFrame:
    series = pd.read_parquet(path)
    for column in ("flightDate", "searchDate"):
        series[column] = pd.to_datetime(series[column])
    return series


def checks_step() -> None:
    report = run_checks(ITINERARIES)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "data_checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def backtest_step(args: argparse.Namespace) -> None:
    cfg = replace(Config(), start_lead=args.start_lead, end_lead=args.end_lead)
    series = load_series(series_path(args))
    split = pd.Timestamp(args.split_date) if args.split_date else default_split_date(series)

    curves = trip_curves(series, cfg)
    train, test = split_trips(curves, split, cfg)
    if train.empty or test.empty:
        raise SystemExit(
            f"Not enough trips around {split.date()}: {len(train)} train, {len(test)} test. "
            "Try a different --split-date or a smaller --start-lead."
        )

    # Curves use training trips only, so what they show can't influence the test result.
    REPORTS.mkdir(exist_ok=True)
    curve = booking_curve(train, cfg)
    curve.to_csv(REPORTS / "booking_curve.csv")
    booking_curve_by_route(train, cfg).to_csv(REPORTS / "booking_curve_by_route.csv")
    plot_booking_curve(curve, len(train), cfg, REPORTS / "booking_curve.png")

    prices, leads = policy_prices(train, test, cfg)
    summary = summarize(prices, cfg)
    summary.to_csv(REPORTS / "backtest_summary.csv")
    (REPORTS / "backtest_setup.json").write_text(
        json.dumps(
            {
                "split_date": str(split.date()),
                "start_lead": cfg.start_lead,
                "end_lead": cfg.end_lead,
                "nonstop_only": args.nonstop_only,
                "all_airlines": args.all_airlines,
                "usable_trips": len(curves),
                "train_trips": len(train),
                "test_trips": len(test),
                "test_departure_dates": prices.index.get_level_values("flightDate").nunique(),
                "best_fixed_lead": leads["overall"],
                "best_lead_by_route": {f"{o}-{d}": lead for (o, d), lead in leads["by_route"].items()},
            },
            indent=2,
        )
    )
    print(f"Split {split.date()}: {len(train):,} train trips, {len(test):,} test trips")
    print(f"Best fixed lead from training: {leads['overall']} days")
    print(summary.round(3).to_string())


def features_step(args: argparse.Namespace) -> None:
    panel = build_panel(load_series(series_path(args)))
    panel.to_parquet(series_path(args, "panel"))
    print(f"{len(panel):,} rows")
    print(panel.describe().T[["count", "mean", "min", "50%", "max"]].round(3).to_string())


def tune_step(args: argparse.Namespace) -> None:
    cfg = replace(Config(), start_lead=args.start_lead, end_lead=args.end_lead)
    series = load_series(series_path(args))
    panel = pd.read_parquet(series_path(args, "panel"))
    tuning = tune(panel, series, cfg, progress=lambda message: print(message, flush=True))
    REPORTS.mkdir(exist_ok=True)
    tuning.to_csv(REPORTS / "model_tuning.csv", index=False)
    print(tuning.groupby(["kind", "margin"])["mean_saving"].agg(["mean", "min"]).round(2).to_string())
    print("Chosen:", choose(tuning).to_dict())


def final_step(args: argparse.Namespace) -> None:
    summary_path = REPORTS / "model_summary.csv"
    if summary_path.exists() and not args.allow_rerun:
        raise SystemExit(
            f"{summary_path} exists. The final test is meant to run once: rerunning it after "
            "seeing results makes it a tuning set. Use --allow-rerun only for a deliberate rerun."
        )
    cfg = replace(Config(), start_lead=args.start_lead, end_lead=args.end_lead)
    series = load_series(series_path(args))
    panel = pd.read_parquet(series_path(args, "panel"))
    split = pd.Timestamp(args.split_date) if args.split_date else default_split_date(series)

    tuning = pd.read_csv(REPORTS / "model_tuning.csv")
    primary = choose(tuning)
    margins = {
        kind: float(choose(tuning[tuning["kind"] == kind])["margin"]) for kind in tuning["kind"].unique()
    }
    result = final_test(panel, series, split, cfg, margins)
    result["summary"].to_csv(summary_path)
    result["delayed_summary"].to_csv(REPORTS / "model_summary_one_day_delay.csv")

    kind = primary["kind"]
    policy = result["policies"][kind]
    paid = result["purchases"][kind]["paid"]
    setup = {
        "split_date": str(split.date()),
        "primary_model": kind,
        "margins": margins,
        "train_rows": result["train_rows"],
        "test_trips": len(result["trips"]),
        "test_departure_dates": result["trips"].get_level_values("flightDate").nunique(),
        "mean_bought_at": {k: float(v["bought_at"].mean()) for k, v in result["purchases"].items()},
        "share_bought_first_day": {
            k: float((v["bought_at"] == cfg.start_lead).mean()) for k, v in result["purchases"].items()
        },
        "bad_case_check": bad_case_check(result["train"], result["test_panel"], policy, paid, cfg),
    }
    (REPORTS / "model_setup.json").write_text(json.dumps(setup, indent=2))
    if "lightgbm" in result["policies"]:
        feature_importance(result["policies"]["lightgbm"]).to_csv(REPORTS / "model_feature_importance.csv")
    print(json.dumps(setup, indent=2))
    print(result["summary"].round(3).to_string())
    print("\nOne-day purchase delay:")
    print(result["delayed_summary"].round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step", choices=["convert", "checks", "series", "backtest", "features", "tune", "final", "all"]
    )
    parser.add_argument("--start-lead", type=int, default=Config.start_lead)
    parser.add_argument("--end-lead", type=int, default=Config.end_lead)
    parser.add_argument("--split-date", help="YYYY-MM-DD; default is 60%% through the search dates")
    parser.add_argument("--nonstop-only", action="store_true")
    parser.add_argument("--all-airlines", action="store_true", help="keep Spirit and Frontier")
    parser.add_argument("--allow-rerun", action="store_true", help="rerun the final test on purpose")
    args = parser.parse_args()

    if args.step in ("convert", "all"):
        convert_csv(CSV, ITINERARIES, TEMP)
    if args.step in ("checks", "all"):
        checks_step()
    if args.step in ("series", "all"):
        excluded = () if args.all_airlines else EXCLUDED_AIRLINES
        build_daily_series(ITINERARIES, series_path(args), TEMP, args.nonstop_only, excluded)
    if args.step in ("backtest", "all"):
        backtest_step(args)
    if args.step == "features":
        features_step(args)
    if args.step == "tune":
        tune_step(args)
    if args.step == "final":
        final_step(args)


if __name__ == "__main__":
    main()
