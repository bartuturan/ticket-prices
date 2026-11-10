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
from ticket_prices.dips import (
    TRIP_KEYS,
    DipConfig,
    analyse_dataset,
    dip_rates,
    google_cases,
    late_dips,
    wide_curves,
    write_summary,
)
from ticket_prices.dip_model import (
    build_decisions,
    choose_setting,
    final_dip_test,
    plot_calibration,
    tune_dip_model,
)
from ticket_prices.dip_walk import (
    DAY_BANDS,
    HEADLINE,
    HEADLINE_LAG,
    LAG_START,
    WALK_END,
    WALK_START,
    WINDOWS,
    summarize_lagged,
    summarize_walk,
    walk_forward,
    walk_forward_lagged,
)
from ticket_prices.evaluate import bad_case_check, choose, feature_importance, final_test, tune
from ticket_prices.features import build_panel
from ticket_prices.prepare import EXCLUDED_AIRLINES, build_daily_series, convert_csv
from ticket_prices.scraped import daily_series, load_scraped_fares, missing_last_quote, same_flight_only

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


def read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError:  # pyarrow can't load where Windows Application Control blocks its DLL
        import duckdb

        return duckdb.connect().execute("SELECT * FROM read_parquet(?)", [str(path)]).df()


def load_series(path: Path) -> pd.DataFrame:
    series = read_parquet(path)
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


EXPEDIA_WINDOWS = {"7-1 days out": (7, 1), "14-8 days out": (14, 8), "21-15 days out": (21, 15), "30-24 days out": (30, 24)}
RYANAIR_WINDOWS = {"6-1 days out": (6, 1), "16-11 days out": (16, 11), "31-26 days out": (31, 26), "45-40 days out": (45, 40)}
RYANAIR_MIN_QUOTES = 4  # of the 6 days in a window: the scraper has only about a week of snapshots


def dips_step(args: argparse.Namespace) -> None:
    """Do fares dip near departure? Expedia 2022 and the scraped Germany fares, side by side."""
    out_dir = REPORTS / "dips"
    cfg = DipConfig(min_drop_pct=args.min_drop_pct, min_drop_abs=args.min_drop_abs)
    summary: dict = {"min_drop_pct": cfg.min_drop_pct, "min_drop_abs": cfg.min_drop_abs}

    # Expedia: trips quoted on every day from 30 days out to 1 day out, so all windows share trips.
    expedia = trip_curves(load_series(series_path(args)), Config(start_lead=30, end_lead=1))
    summary["expedia"] = analyse_dataset(
        "expedia", expedia, EXPEDIA_WINDOWS, cfg, expedia.index.get_level_values("flightDate"), out_dir
    )

    # Ryanair: groups are routes, because only a few departure dates have a full window.
    fares = load_scraped_fares(Path(args.scraped_dir), "ryanair", skip_legacy=True)
    series = daily_series(fares)
    curves = wide_curves(series, int(series["days_to_departure"].max()))
    route = curves.index.get_level_values("origin") + "-" + curves.index.get_level_values("destination")
    summary["ryanair"] = analyse_dataset(
        "ryanair", curves, RYANAIR_WINDOWS, cfg, route, out_dir, min_obs=RYANAIR_MIN_QUOTES
    )
    summary["ryanair"]["snapshot_dates"] = [str(d.date()) for d in sorted(series["searchDate"].unique())]
    summary["ryanair"]["missing_at_lead_1"] = missing_last_quote(series)

    # The same check on quotes that stay on one flight number.
    same = same_flight_only(series)
    same_curves = wide_curves(same, int(same["days_to_departure"].max()))
    same_route = same_curves.index.get_level_values("origin") + "-" + same_curves.index.get_level_values("destination")
    same_rates = dip_rates(same_curves, RYANAIR_WINDOWS, cfg, same_route, RYANAIR_MIN_QUOTES)
    same_rates.to_csv(out_dir / "ryanair_same_flight_dip_rates.csv")
    summary["ryanair"]["same_flight_dip_rate_by_window"] = {
        w: round(float(r), 4) for w, r in same_rates["dip_rate"].items()
    }

    final = late_dips(curves, RYANAIR_WINDOWS["6-1 days out"], cfg, RYANAIR_MIN_QUOTES)
    final = final[final["eligible"]].reset_index()
    by_origin = final.groupby("origin").agg(departures=("dipped", "size"), dip_rate=("dipped", "mean"))
    by_origin.to_csv(out_dir / "ryanair_dip_rates_by_origin.csv")

    google = load_scraped_fares(Path(args.scraped_dir), "google_flights")
    paths, spread = google_cases(google)
    paths.to_csv(out_dir / "google_flights_cases.csv", index=False)
    spread.to_csv(out_dir / "google_flights_intraday_spread.csv")
    summary["google_flights"] = {"departures_tracked_to_lead_1": int(paths[TRIP_KEYS].drop_duplicates().shape[0])}

    write_summary(summary, out_dir / "dips_summary.json")
    print(json.dumps(summary, indent=2, default=str))


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


DIP_MODEL = REPORTS / "dip_model"


def dip_model_inputs(args: argparse.Namespace):
    cfg = replace(Config(), start_lead=args.start_lead, end_lead=args.end_lead)
    dip_cfg = DipConfig(min_drop_pct=args.min_drop_pct, min_drop_abs=args.min_drop_abs)
    curves = trip_curves(load_series(series_path(args)), cfg)
    decisions = build_decisions(read_parquet(series_path(args, "panel")), curves, cfg)
    return cfg, dip_cfg, curves, decisions


def dip_tune_step(args: argparse.Namespace) -> None:
    cfg, dip_cfg, curves, decisions = dip_model_inputs(args)
    tuning = tune_dip_model(curves, decisions, cfg, dip_cfg, progress=lambda message: print(message, flush=True))
    DIP_MODEL.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(DIP_MODEL / "tuning.csv", index=False)
    print(tuning.groupby(["num_leaves", "rounds"])[["if_gain", "when_gain", "score"]].mean().round(4).to_string())
    print("Chosen:", choose_setting(tuning).to_dict())


def dip_final_step(args: argparse.Namespace) -> None:
    summary_path = DIP_MODEL / "summary.csv"
    if summary_path.exists() and not args.allow_rerun:
        raise SystemExit(
            f"{summary_path} exists. The final test is meant to run once: rerunning it after "
            "seeing results makes it a tuning set. Use --allow-rerun only for a deliberate rerun."
        )
    cfg, dip_cfg, curves, decisions = dip_model_inputs(args)
    tuning = pd.read_csv(DIP_MODEL / "tuning.csv")
    if not ((tuning["min_drop_pct"] == dip_cfg.min_drop_pct) & (tuning["min_drop_abs"] == dip_cfg.min_drop_abs)).all():
        raise SystemExit("tuning.csv used other dip thresholds than --min-drop-pct and --min-drop-abs. Run dip-tune again.")
    chosen = choose_setting(tuning)
    split = pd.Timestamp(args.split_date) if args.split_date else default_split_date(load_series(series_path(args)))

    result = final_dip_test(curves, decisions, split, cfg, dip_cfg, int(chosen["num_leaves"]), int(chosen["rounds"]))
    result["summary"].to_csv(summary_path)
    result["calibration"].to_csv(DIP_MODEL / "calibration.csv")
    result["by_lead"].to_csv(DIP_MODEL / "by_lead.csv")
    result["importance"].to_csv(DIP_MODEL / "feature_importance.csv")
    plot_calibration(
        result["calibration"].loc["model"], result["calibration"].loc["baseline"], DIP_MODEL / "calibration.png"
    )
    setup = {
        "split_date": str(split.date()),
        "start_lead": cfg.start_lead,
        "end_lead": cfg.end_lead,
        "min_drop_pct": dip_cfg.min_drop_pct,
        "min_drop_abs": dip_cfg.min_drop_abs,
        "num_leaves": int(chosen["num_leaves"]),
        "rounds": int(chosen["rounds"]),
        "tuning_score": float(chosen["mean"]),
        **result["counts"],
        "pass_mark": result["pass_mark"],
    }
    (DIP_MODEL / "setup.json").write_text(json.dumps(setup, indent=2))
    print(json.dumps(setup, indent=2))
    print(result["summary"].round(4).to_string())
    print(result["calibration"].round(3).to_string())


def dip_walk_step(args: argparse.Namespace) -> None:
    """Replay the Expedia period week by week, refitting and recalibrating. Not a final test."""
    cfg, dip_cfg, curves, decisions = dip_model_inputs(args)
    tuning = pd.read_csv(DIP_MODEL / "tuning.csv")
    if not ((tuning["min_drop_pct"] == dip_cfg.min_drop_pct) & (tuning["min_drop_abs"] == dip_cfg.min_drop_abs)).all():
        raise SystemExit("tuning.csv used other dip thresholds than --min-drop-pct and --min-drop-abs. Run dip-tune again.")
    chosen = choose_setting(tuning)
    weeks = pd.date_range(WALK_START, WALK_END, freq="7D")

    result = walk_forward(
        curves, decisions, weeks, cfg, dip_cfg, int(chosen["num_leaves"]), int(chosen["rounds"]),
        progress=lambda message: print(message, flush=True),
    )
    summary = summarize_walk(result["scores"], dip_cfg)

    out = DIP_MODEL / "walk_forward"
    out.mkdir(parents=True, exist_ok=True)
    result["weekly"].to_csv(out / "by_week.csv", index=False)
    summary["summary"].to_csv(out / "summary.csv")
    summary["calibration"].to_csv(out / "calibration.csv")
    calibration = summary["calibration"]
    plot_calibration(
        calibration.loc[HEADLINE],
        calibration.loc["raw"],
        out / "calibration.png",
        labels=(f"Recalibrated ({HEADLINE})", "Raw model"),
        title="Calibration over the walk-forward weeks, ten equal-sized bins",
    )
    setup = {
        "weeks": [str(week.date()) for week in weeks],
        "windows": list(WINDOWS),
        "headline_variant": HEADLINE,
        "num_leaves": int(chosen["num_leaves"]),
        "rounds": int(chosen["rounds"]),
        "min_drop_pct": dip_cfg.min_drop_pct,
        "min_drop_abs": dip_cfg.min_drop_abs,
        "scored_decisions": int(len(result["scores"]["raw"]["model"])),
        "evidence": "development: the replay covers the period the final dip-model test used",
        "pass_mark": summary["pass_mark"],
    }
    (out / "setup.json").write_text(json.dumps(setup, indent=2))
    print(json.dumps(setup, indent=2))
    print(summary["summary"].round(4).to_string())
    weekly = result["weekly"]
    for variant in ("raw", HEADLINE):
        part = weekly[weekly["variant"] == variant].assign(week=lambda frame: frame["week"].dt.date.astype(str))
        print(f"\n{variant}")
        print(
            part[["week", "observed", "model_predicted", "baseline_predicted", "model_a", "model_b", "recalibration_rows"]]
            .round(3)
            .to_string(index=False)
        )


def dip_walk_lag_step(args: argparse.Namespace) -> None:
    """Replay with a lagged refit corrected on recently departed trips, by days to departure. Not a final test."""
    cfg, dip_cfg, curves, decisions = dip_model_inputs(args)
    tuning = pd.read_csv(DIP_MODEL / "tuning.csv")
    if not ((tuning["min_drop_pct"] == dip_cfg.min_drop_pct) & (tuning["min_drop_abs"] == dip_cfg.min_drop_abs)).all():
        raise SystemExit("tuning.csv used other dip thresholds than --min-drop-pct and --min-drop-abs. Run dip-tune again.")
    chosen = choose_setting(tuning)
    weeks = pd.date_range(LAG_START, WALK_END, freq="7D")

    result = walk_forward_lagged(
        curves, decisions, weeks, cfg, dip_cfg, int(chosen["num_leaves"]), int(chosen["rounds"]),
        progress=lambda message: print(message, flush=True),
    )
    summary = summarize_lagged(result["scores"], dip_cfg)

    out = DIP_MODEL / "walk_lagged"
    out.mkdir(parents=True, exist_ok=True)
    result["weekly"].to_csv(out / "by_week.csv", index=False)
    summary["summary"].to_csv(out / "summary.csv")
    summary["vs_refit"].to_csv(out / "vs_refit.csv")
    summary["calibration"].to_csv(out / "calibration.csv")
    calibration = summary["calibration"]
    plot_calibration(
        calibration.loc[HEADLINE_LAG],
        calibration.loc["refit"],
        out / "calibration.png",
        labels=(f"Lagged, by band ({HEADLINE_LAG})", "Weekly refit"),
        title="Calibration over the lagged walk-forward weeks, ten equal-sized bins",
    )
    setup = {
        "weeks": [str(week.date()) for week in weeks],
        "variants": ["refit", "lag28_raw", "lag28_shift", "lag28_band", "lag14_band"],
        "bands": [list(band) for band in DAY_BANDS],
        "headline_variant": HEADLINE_LAG,
        "num_leaves": int(chosen["num_leaves"]),
        "rounds": int(chosen["rounds"]),
        "min_drop_pct": dip_cfg.min_drop_pct,
        "min_drop_abs": dip_cfg.min_drop_abs,
        "scored_decisions": int(len(result["scores"]["refit"]["model"])),
        "evidence": "development: a second look at weeks the earlier replay and the final test already used",
        "pass_mark": summary["pass_mark"],
    }
    (out / "setup.json").write_text(json.dumps(setup, indent=2))
    print(json.dumps(setup, indent=2))
    print(summary["summary"].round(4).to_string())
    print("\nAgainst the plain weekly refit (gain = refit loss minus this variant's loss)")
    print(summary["vs_refit"].round(4).to_string())
    weekly = result["weekly"]
    shift_columns = [f"model_shift_{low}_{high}" for low, high in DAY_BANDS]
    part = weekly[weekly["variant"] == HEADLINE_LAG].assign(week=lambda frame: frame["week"].dt.date.astype(str))
    print(f"\n{HEADLINE_LAG}")
    print(part[["week", "observed", "model_predicted", *shift_columns, "recalibration_rows"]].round(3).to_string(index=False))
    for variant in ("refit", "lag28_raw"):
        part = weekly[weekly["variant"] == variant].assign(week=lambda frame: frame["week"].dt.date.astype(str))
        print(f"\n{variant}")
        print(part[["week", "observed", "model_predicted", "model_if_log_loss", "baseline_if_log_loss"]].round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        choices=[
            "convert", "checks", "series", "backtest", "features", "tune", "final", "dips",
            "dip-tune", "dip-final", "dip-walk", "dip-walk-lag", "all",
        ],
    )
    parser.add_argument("--scraped-dir", default="germany-flight-fares/data", help="dips: scraper output folder")
    parser.add_argument("--min-drop-pct", type=float, default=DipConfig.min_drop_pct, help="dips, dip-tune, dip-final: smallest fall that counts")
    parser.add_argument("--min-drop-abs", type=float, default=DipConfig.min_drop_abs, help="dips, dip-tune, dip-final: smallest fall in currency units")
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
    if args.step == "dips":
        dips_step(args)
    if args.step == "dip-tune":
        dip_tune_step(args)
    if args.step == "dip-final":
        dip_final_step(args)
    if args.step == "dip-walk":
        dip_walk_step(args)
    if args.step == "dip-walk-lag":
        dip_walk_lag_step(args)


if __name__ == "__main__":
    main()
