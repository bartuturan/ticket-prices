"""Tune on time-ordered folds inside the training period, then run the single final test."""

from dataclasses import dataclass

import pandas as pd

from ticket_prices.backtest import Config, policy_prices, purchase_prices, split_trips, summarize, trip_curves
from ticket_prices.features import FEATURES, LEAD
from ticket_prices.policy import MODEL_KINDS, fit_bad_case, fit_policy, rows_at_lead, simulate, training_rows

MARGINS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.25)


@dataclass(frozen=True)
class Fold:
    """Train on trips finished before `split`, test on departures in [first_flight, end_flight)."""

    split: str
    end_flight: str


# Every quote in these folds comes before the 2022-07-28 final-test split.
TUNING_FOLDS = (Fold("2022-05-30", "2022-07-15"), Fold("2022-06-15", "2022-07-29"))


def _fold_trips(curves: pd.DataFrame, fold: Fold, cfg: Config) -> pd.MultiIndex:
    flights = curves.index.get_level_values("flightDate")
    first = pd.Timestamp(fold.split) + pd.Timedelta(days=cfg.start_lead)
    return curves.index[(flights >= first) & (flights < pd.Timestamp(fold.end_flight))]


def tune(
    panel: pd.DataFrame,
    series: pd.DataFrame,
    cfg: Config,
    folds: tuple[Fold, ...] = TUNING_FOLDS,
    margins: tuple[float, ...] = MARGINS,
    kinds: tuple[str, ...] = tuple(MODEL_KINDS),
    n_folds: int = 3,
    min_rows: int = 200,
    progress=lambda message: None,
) -> pd.DataFrame:
    """Mean saving against buying on the first day, per model kind, margin and fold."""
    curves = trip_curves(series, cfg)
    rows = []
    for fold in folds:
        trips = _fold_trips(curves, fold, cfg)
        train = training_rows(panel, pd.Timestamp(fold.split), cfg)
        test_panel = panel[panel.index.droplevel(LEAD).isin(trips)]
        first_day = curves.loc[trips, cfg.start_lead]
        for kind in kinds:
            for margin in margins:
                policy = fit_policy(train, cfg, margin, MODEL_KINDS[kind], n_folds, min_rows)
                result = simulate(test_panel, trips, policy, cfg)
                saving = first_day - result["paid"]
                rows.append(
                    {
                        "fold": fold.split,
                        "kind": kind,
                        "margin": margin,
                        "trips": len(trips),
                        "mean_saving": saving.mean(),
                        "mean_saving_pct": 100 * saving.mean() / first_day.mean(),
                        "share_bought_first_day": (result["bought_at"] == cfg.start_lead).mean(),
                        "mean_bought_at": result["bought_at"].mean(),
                    }
                )
                progress(f"{fold.split} {kind} margin={margin}: saving {saving.mean():.2f}")
    return pd.DataFrame(rows)


def choose(tuning: pd.DataFrame) -> pd.Series:
    """Pick the setting with the best mean saving across folds, weighting folds equally."""
    scores = tuning.groupby(["kind", "margin"])["mean_saving"].agg(["mean", "min"]).reset_index()
    return scores.sort_values(["mean", "margin"], ascending=[False, False]).iloc[0]


def delayed_prices(test_curves: pd.DataFrame, purchases: dict, cfg: Config) -> pd.DataFrame:
    """What each rule pays if the purchase goes through one quoted day after the decision.

    A saving that needs the exact quote on the day could come from one-day flickers in the
    data, which a real buyer may not be able to catch. Every rule gets the same delay. A
    purchase on the last buying day has no later day, so it stays at that day's price.
    """
    next_day = purchase_prices(test_curves).shift(-1, axis=1).ffill(axis=1)
    prices = {
        "buy_first_day": next_day[cfg.start_lead],
        "hindsight": next_day.min(axis=1),
    }
    rows = next_day.index.get_indexer(test_curves.index)
    for kind, purchase in purchases.items():
        columns = next_day.columns.get_indexer(purchase["bought_at"].reindex(test_curves.index).astype(int))
        prices[f"model_{kind}"] = pd.Series(next_day.to_numpy()[rows, columns], index=test_curves.index)
    return pd.DataFrame(prices)[["buy_first_day", *[f"model_{kind}" for kind in purchases], "hindsight"]]


def final_test(
    panel: pd.DataFrame,
    series: pd.DataFrame,
    split: pd.Timestamp,
    cfg: Config,
    margins: dict[str, float],
    n_folds: int = 3,
    min_rows: int = 200,
) -> dict:
    """Fit on everything before the split and score test trips against the baselines."""
    curves = trip_curves(series, cfg)
    train_curves, test_curves = split_trips(curves, split, cfg)
    prices, leads = policy_prices(train_curves, test_curves, cfg)

    train = training_rows(panel, split, cfg)
    trips = test_curves.index
    test_panel = panel[panel.index.droplevel(LEAD).isin(trips)]

    policies, purchases = {}, {}
    for kind, margin in margins.items():
        policies[kind] = fit_policy(train, cfg, margin, MODEL_KINDS[kind], n_folds, min_rows)
        purchases[kind] = simulate(test_panel, trips, policies[kind], cfg)
        prices[f"model_{kind}"] = purchases[kind]["paid"]

    return {
        "prices": prices,
        "summary": summarize(prices, cfg),
        "delayed_summary": summarize(delayed_prices(test_curves, purchases, cfg), cfg),
        "leads": leads,
        "policies": policies,
        "purchases": purchases,
        "train": train,
        "test_panel": test_panel,
        "trips": trips,
        "train_rows": len(train),
    }


def bad_case_check(train: pd.DataFrame, test_panel: pd.DataFrame, policy, paid: pd.Series, cfg: Config) -> dict:
    """On the first shopping day, how often did the final price stay within the bad case?"""
    models = fit_bad_case(train, policy)
    if cfg.start_lead not in models:
        return {"share_within_bad_case": None, "mean_bad_case_ratio": None}
    rows = rows_at_lead(test_panel, cfg.start_lead)
    ratio = models[cfg.start_lead].predict(rows[FEATURES])
    limit = pd.Series(ratio * rows["price"].to_numpy(), index=rows.index)
    return {
        "share_within_bad_case": float((paid.reindex(rows.index) <= limit).mean()),
        "mean_bad_case_ratio": float(ratio.mean()),
    }


def feature_importance(policy) -> pd.DataFrame:
    """Total gain per feature across a LightGBM policy's models, as a share."""
    totals = pd.Series(0.0, index=FEATURES)
    for model in policy.models.values():
        totals += pd.Series(model.booster_.feature_importance("gain"), index=model.booster_.feature_name())
    return (totals / totals.sum()).sort_values(ascending=False).rename("share_of_gain").to_frame()
