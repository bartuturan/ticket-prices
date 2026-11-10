"""Dip-timing model: will a fare I could still buy appear below today's price, and on which day?

A decision is a trip (route and departure date) on one shopping day, its lead. A later day is
a catchable dip for it when that day's quote is below the decision's price by at least
`DipConfig.min_drop_pct` and `min_drop_abs`, and so is the price a buyer pays one quoted day
later. That second test drops dips that vanish before anyone could act on them. The first such
day is the one to place.

The model is a discrete-time hazard. One LightGBM classifier says how likely a catchable dip
lands on day `k`, given that none landed on an earlier day, from what is known at the decision
lead. Chaining those daily chances gives both forecasts:

- if:   1 - prod(1 - h_k) over the remaining days
- when: h_k * prod(1 - h_j) over the days j before k, and the leftover chance of no dip

The baseline is the training frequency of a dip for each (decision lead, later day) pair.
"""

from dataclasses import dataclass

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ticket_prices.backtest import TRIP_KEYS, Config, split_trips
from ticket_prices.curves import GRID, SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY
from ticket_prices.dips import DipConfig, ratio_ci
from ticket_prices.evaluate import TUNING_FOLDS, Fold, _fold_trips
from ticket_prices.features import CATEGORICAL, FEATURES, LEAD
from ticket_prices.policy import GBM_PARAMS

MODEL_COLUMNS = [*FEATURES, "lead", "horizon"]
CATEGORICAL_COLUMNS = [MODEL_COLUMNS.index(name) for name in CATEGORICAL]
PROB_CLIP = 1e-6
TOP_DAYS = 5  # the "when" forecast hits if the true day is among its 5 most likely days
MIN_HORIZON_FOR_TOP = 10  # with fewer days left, 5 days cover half the horizon and the hit is trivial
CALIBRATION_BINS = 10
CALIBRATION_TOLERANCE = 0.05
GRID_LEAVES = (7, 15, 31)
GRID_ROUNDS = (100, 200, 400)
BASELINE_PRIOR_ROWS = 1.0


@dataclass
class Decisions:
    """One entry per trip and shopping day with a quote, with what is known on that day."""

    trip: np.ndarray  # row of the trip in the curves
    lead: np.ndarray
    flight_date: np.ndarray
    features: np.ndarray  # float32, one column per name in FEATURES

    def __len__(self) -> int:
        return len(self.trip)

    def take(self, mask: np.ndarray) -> "Decisions":
        return Decisions(self.trip[mask], self.lead[mask], self.flight_date[mask], self.features[mask])


@dataclass
class Hazard:
    """One row per decision and later day, in time order, grouped by decision.

    `y` marks the day of the first catchable dip. `at_risk` marks days up to and including
    it, which is where the hazard model learns: after a dip has landed, the question is over.
    """

    row: np.ndarray  # position of the decision
    k: np.ndarray  # the later day, as days to departure
    y: np.ndarray
    at_risk: np.ndarray

    def __len__(self) -> int:
        return len(self.row)


def delayed_prices(prices: np.ndarray) -> np.ndarray:
    """The price paid when a purchase decided on each day goes through on the next quoted day.

    Matches `evaluate.delayed_prices`: the last buying day has no later day, so it keeps its
    own price. Columns run from the earliest day to the last buying day.
    """
    filled = pd.DataFrame(prices).bfill(axis=1).to_numpy()
    delayed = np.empty_like(filled)
    delayed[:, :-1] = filled[:, 1:]
    delayed[:, -1] = filled[:, -1]
    return delayed


def _is_drop(reference: np.ndarray, price: np.ndarray, cfg: DipConfig) -> np.ndarray:
    drop = reference - price
    return (drop / reference >= cfg.min_drop_pct) & (drop >= cfg.min_drop_abs)


def _expand(lead: np.ndarray, end_lead: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every decision, the later days from the day after it to the last buying day."""
    horizon = np.maximum(lead - end_lead, 0)
    row = np.repeat(np.arange(len(lead), dtype=np.int32), horizon)
    start = np.cumsum(horizon) - horizon
    position = np.arange(row.size) - start[row]
    return row, (lead[row] - 1 - position).astype(np.int16), start


def build_hazard(
    prices: np.ndarray, leads: np.ndarray, trip: np.ndarray, lead: np.ndarray, cfg: DipConfig, end_lead: int
) -> Hazard:
    """Label every later day of every decision.

    `prices` has one row per trip and one column per entry of `leads`, which run in steps of
    one from the earliest day to the last buying day. `trip` and `lead` say, for each decision,
    which row and which day. The decision's price is the trip's quote on its own day.
    """
    row, k, start = _expand(lead, end_lead)
    if row.size == 0:
        empty = np.zeros(0, dtype=bool)
        return Hazard(row, k, empty, empty.copy())

    top = int(leads[0])
    reference = prices[trip, top - lead][row]
    trips = trip[row]
    column = (top - k).astype(np.intp)
    delayed = delayed_prices(prices)
    catchable = _is_drop(reference, prices[trips, column], cfg) & _is_drop(reference, delayed[trips, column], cfg)

    seen = np.cumsum(catchable)
    base = np.where(start > 0, seen[np.maximum(start - 1, 0)], 0)
    within = seen - base[row]  # dips so far for this decision, counting this day
    return Hazard(row, k, catchable & (within == 1), (within - catchable.astype(int)) == 0)


def _encode(column: pd.Series, name: str) -> np.ndarray:
    if name == "cheapest_airline":
        return pd.Categorical(column.astype(str)).codes.astype(np.float32)
    return column.astype(float).to_numpy(dtype=np.float32)


def build_decisions(panel: pd.DataFrame, curves: pd.DataFrame, cfg: Config) -> Decisions:
    """Decisions for every trip in `curves`, from its panel rows between the two buying days.

    Categories are coded once here, so every split of the decisions shares the same codes.
    """
    frame = panel.reset_index() if panel.index.names[0] is not None else panel
    frame = frame[frame[LEAD].between(cfg.end_lead + 1, cfg.start_lead)]
    flight = pd.to_datetime(frame["flightDate"]).astype(curves.index.get_level_values("flightDate").dtype)
    keys = pd.MultiIndex.from_arrays(
        [frame["origin"].to_numpy(), frame["destination"].to_numpy(), flight.to_numpy()], names=TRIP_KEYS
    )
    trip = curves.index.get_indexer(keys)

    top = int(curves.columns[0])
    quoted = np.zeros(len(frame), dtype=bool)
    known = trip >= 0
    quoted[known] = ~np.isnan(curves.to_numpy(dtype=float)[trip[known], top - frame[LEAD].to_numpy()[known]])
    frame, trip = frame[quoted], trip[quoted]

    matrix = np.empty((len(frame), len(FEATURES)), dtype=np.float32)
    for position, name in enumerate(FEATURES):
        matrix[:, position] = _encode(frame[name], name)
    return Decisions(
        trip.astype(np.int32),
        frame[LEAD].to_numpy(),
        pd.to_datetime(frame["flightDate"]).to_numpy(),
        matrix,
    )


def design_matrix(decisions: Decisions, hazard: Hazard, select=slice(None)) -> np.ndarray:
    """Model inputs for the chosen hazard rows: features at the decision, its lead, and the days ahead."""
    row = hazard.row[select]
    width = len(FEATURES)
    matrix = np.empty((len(row), len(MODEL_COLUMNS)), dtype=np.float32)
    matrix[:, :width] = decisions.features[row]
    lead = decisions.lead[row]
    matrix[:, width] = lead
    matrix[:, width + 1] = lead - hazard.k[select]
    return matrix


def fit_hazard(decisions: Decisions, hazard: Hazard, num_leaves: int = 15, rounds: int = 200, **overrides):
    risk = hazard.at_risk
    data = lgb.Dataset(
        design_matrix(decisions, hazard, risk),
        label=hazard.y[risk].astype(np.float32),
        categorical_feature=CATEGORICAL_COLUMNS,
        feature_name=MODEL_COLUMNS,
        free_raw_data=False,
    )
    params = {**GBM_PARAMS, "objective": "binary", "num_leaves": num_leaves, **overrides}
    return lgb.train(params, data, num_boost_round=rounds)


def predict_hazard(booster, decisions: Decisions, hazard: Hazard, num_iteration=None, chunk: int = 1_000_000) -> np.ndarray:
    out = np.empty(len(hazard))
    for start in range(0, len(out), chunk):
        part = slice(start, start + chunk)
        out[part] = booster.predict(design_matrix(decisions, hazard, part), num_iteration=num_iteration)
    return np.clip(out, PROB_CLIP, 1 - PROB_CLIP)


def baseline_table(
    lead: np.ndarray, k: np.ndarray, y: np.ndarray, at_risk: np.ndarray, max_lead: int, prior_rows: float = BASELINE_PRIOR_ROWS
) -> np.ndarray:
    """Share of at-risk rows that were a first dip, for each (decision lead, later day).

    A pseudo-count of `prior_rows` rows at the overall rate keeps empty or all-zero cells
    from forecasting certainty.
    """
    size = max_lead + 1
    flat = lead[at_risk].astype(int) * size + k[at_risk].astype(int)
    hits = np.bincount(flat, weights=y[at_risk], minlength=size * size)
    count = np.bincount(flat, minlength=size * size)
    overall = hits.sum() / max(count.sum(), 1)
    table = (hits + prior_rows * overall) / np.maximum(count + prior_rows, 1e-12)
    return np.clip(table.reshape(size, size), PROB_CLIP, 1 - PROB_CLIP)


def baseline_predict(table: np.ndarray, decisions: Decisions, hazard: Hazard) -> np.ndarray:
    return table[decisions.lead[hazard.row].astype(int), hazard.k.astype(int)]


def forecast(row: np.ndarray, h: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From daily chances to the two answers.

    Returns, per decision, the chance of any dip and the chance of none, and per hazard row
    the chance that the first dip lands on that day. Rows must be grouped by decision.
    """
    stay = np.log1p(-h)
    none = np.exp(np.bincount(row, weights=stay, minlength=n))
    seen = np.cumsum(stay)
    start = np.searchsorted(row, np.arange(n))
    base = np.where(start > 0, seen[np.maximum(start - 1, 0)], 0.0)
    first = h * np.exp(seen - stay - base[row])
    return 1 - none, none, first


def decision_scores(decisions: Decisions, hazard: Hazard, h: np.ndarray) -> pd.DataFrame:
    """Per-decision losses and hits for one set of daily chances."""
    n = len(decisions)
    row = hazard.row
    p_any, _, first = forecast(row, h, n)
    got = np.bincount(row, weights=hazard.y, minlength=n) > 0
    p = np.clip(p_any, PROB_CLIP, 1 - PROB_CLIP)

    # The first-dip day's likelihood is the product of daily terms over the at-risk days.
    daily = np.where(hazard.y, np.log(h), np.log1p(-h))
    when_loss = -np.bincount(row, weights=np.where(hazard.at_risk, daily, 0.0), minlength=n)

    dip_first = np.bincount(row, weights=np.where(hazard.y, first, 0.0), minlength=n)
    ahead = np.bincount(row, weights=(first > dip_first[row]).astype(float), minlength=n)
    horizon = np.bincount(row, minlength=n)
    return pd.DataFrame(
        {
            "lead": decisions.lead,
            "flight_date": decisions.flight_date,
            "any_dip": got,
            "p_if": p,
            "if_log_loss": -(got * np.log(p) + (~got) * np.log1p(-p)),
            "if_brier": (p - got) ** 2,
            "when_log_loss": when_loss,
            "top_eligible": got & (horizon >= MIN_HORIZON_FOR_TOP),
            "top_hit": got & (ahead < TOP_DAYS),
        }
    )


def auc(score: np.ndarray, label: np.ndarray) -> float:
    ranks = pd.Series(score).rank().to_numpy()
    positives = int(label.sum())
    negatives = len(label) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def summarize_scores(model: pd.DataFrame, baseline: pd.DataFrame, cfg: DipConfig) -> pd.DataFrame:
    """Model against baseline. Intervals resample departure dates, not decisions, because
    decisions for one departure share its price path and the same search days."""
    dates = model["flight_date"].to_numpy()
    rows = []

    def compare(name, model_values, baseline_values, groups, lower_is_better):
        gain = (baseline_values - model_values) if lower_is_better else (model_values - baseline_values)
        low, high = ratio_ci(gain, groups, cfg)
        rows.append(
            {
                "metric": name,
                "model": float(np.mean(model_values)),
                "baseline": float(np.mean(baseline_values)),
                "gain": float(np.mean(gain)),
                "gain_ci_low": low,
                "gain_ci_high": high,
            }
        )

    for name in ("if_log_loss", "if_brier", "when_log_loss"):
        compare(name, model[name].to_numpy(), baseline[name].to_numpy(), dates, True)

    keep = model["top_eligible"].to_numpy()
    compare(
        f"when_top{TOP_DAYS}_hit",
        model["top_hit"].to_numpy()[keep].astype(float),
        baseline["top_hit"].to_numpy()[keep].astype(float),
        dates[keep],
        False,
    )
    rows.append(
        {
            "metric": "if_auc",
            "model": auc(model["p_if"].to_numpy(), model["any_dip"].to_numpy()),
            "baseline": auc(baseline["p_if"].to_numpy(), baseline["any_dip"].to_numpy()),
            "gain": np.nan,
            "gain_ci_low": np.nan,
            "gain_ci_high": np.nan,
        }
    )
    return pd.DataFrame(rows).set_index("metric")


def calibration_table(scores: pd.DataFrame, bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    """Predicted against observed chance of a dip, in bins of equal numbers of decisions."""
    bucket = pd.qcut(scores["p_if"], bins, labels=False, duplicates="drop")
    table = (
        pd.DataFrame({"bin": bucket, "predicted": scores["p_if"], "observed": scores["any_dip"].astype(float)})
        .groupby("bin")
        .agg(decisions=("observed", "size"), predicted=("predicted", "mean"), observed=("observed", "mean"))
    )
    table["gap"] = table["predicted"] - table["observed"]
    return table


def by_lead_table(model: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """The 'if' forecast at each decision lead, to show where the model adds anything."""
    frame = pd.DataFrame(
        {
            "lead": model["lead"],
            "observed": model["any_dip"].astype(float),
            "model_predicted": model["p_if"],
            "baseline_predicted": baseline["p_if"],
            "model_log_loss": model["if_log_loss"],
            "baseline_log_loss": baseline["if_log_loss"],
            "model_brier": model["if_brier"],
            "baseline_brier": baseline["if_brier"],
        }
    )
    table = frame.groupby("lead").mean()
    table.insert(0, "decisions", frame.groupby("lead").size())
    table["log_loss_gain"] = table["baseline_log_loss"] - table["model_log_loss"]
    return table.sort_index(ascending=False)


def feature_importance(booster) -> pd.DataFrame:
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    return (gain / gain.sum()).sort_values(ascending=False).rename("share_of_gain").to_frame()


def plot_calibration(
    model: pd.DataFrame,
    baseline: pd.DataFrame,
    out,
    labels: tuple[str, str] = ("Model", "Baseline"),
    title: str = "Calibration on test decisions, ten equal-sized bins",
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    top = max(model[["predicted", "observed"]].to_numpy().max(), baseline[["predicted", "observed"]].to_numpy().max())
    ax.plot([0, top], [0, top], color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 4)))
    ax.plot(baseline["predicted"], baseline["observed"], "o-", color=TEXT_SECONDARY, linewidth=1.5, markersize=5, label=labels[1])
    ax.plot(model["predicted"], model["observed"], "o-", color=SERIES, linewidth=2, markersize=6, label=labels[0])
    ax.set_xlabel("Predicted chance of a dip before departure", color=TEXT_PRIMARY)
    ax.set_ylabel("Share that had one", color=TEXT_PRIMARY)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _subsets(curves: pd.DataFrame, decisions: Decisions, train_trips: pd.MultiIndex, test_trips: pd.MultiIndex):
    train = decisions.take(np.isin(decisions.trip, curves.index.get_indexer(train_trips)))
    test = decisions.take(np.isin(decisions.trip, curves.index.get_indexer(test_trips)))
    return train, test


def _fit_and_score_inputs(curves, train, test, cfg: Config, dip_cfg: DipConfig):
    prices = curves.to_numpy(dtype=float)
    leads = np.asarray(curves.columns)
    train_hazard = build_hazard(prices, leads, train.trip, train.lead, dip_cfg, cfg.end_lead)
    test_hazard = build_hazard(prices, leads, test.trip, test.lead, dip_cfg, cfg.end_lead)
    table = baseline_table(
        train.lead[train_hazard.row], train_hazard.k, train_hazard.y, train_hazard.at_risk, cfg.start_lead
    )
    baseline = decision_scores(test, test_hazard, baseline_predict(table, test, test_hazard))
    return train_hazard, test_hazard, baseline


def tune_dip_model(
    curves: pd.DataFrame,
    decisions: Decisions,
    cfg: Config,
    dip_cfg: DipConfig,
    folds: tuple[Fold, ...] = TUNING_FOLDS,
    leaves: tuple[int, ...] = GRID_LEAVES,
    rounds: tuple[int, ...] = GRID_ROUNDS,
    progress=lambda message: None,
) -> pd.DataFrame:
    """Gain over the baseline, in log loss per decision, for each setting and time-ordered fold.

    A fold trains on trips that finished before its split and tests on departures that were
    first shopped for after it. One fit per leaf count covers every round count.
    """
    rows = []
    for fold in folds:
        split = pd.Timestamp(fold.split)
        train_trips = curves.index[curves.index.get_level_values("flightDate") < split + pd.Timedelta(days=cfg.end_lead)]
        train, test = _subsets(curves, decisions, train_trips, _fold_trips(curves, fold, cfg))
        train_hazard, test_hazard, baseline = _fit_and_score_inputs(curves, train, test, cfg, dip_cfg)
        progress(f"{fold.split}: {len(train):,} train and {len(test):,} test decisions, {len(train_hazard):,} hazard rows")
        for num_leaves in leaves:
            booster = fit_hazard(train, train_hazard, num_leaves, max(rounds))
            for num_rounds in rounds:
                scores = decision_scores(test, test_hazard, predict_hazard(booster, test, test_hazard, num_rounds))
                if_gain = float((baseline["if_log_loss"] - scores["if_log_loss"]).mean())
                when_gain = float((baseline["when_log_loss"] - scores["when_log_loss"]).mean())
                rows.append(
                    {
                        "fold": fold.split,
                        "num_leaves": num_leaves,
                        "rounds": num_rounds,
                        "train_decisions": len(train),
                        "test_decisions": len(test),
                        "if_gain": if_gain,
                        "when_gain": when_gain,
                        "score": (if_gain + when_gain) / 2,
                        "min_drop_pct": dip_cfg.min_drop_pct,
                        "min_drop_abs": dip_cfg.min_drop_abs,
                    }
                )
                progress(f"{fold.split} leaves={num_leaves} rounds={num_rounds}: if {if_gain:+.4f}, when {when_gain:+.4f}")
    return pd.DataFrame(rows)


def choose_setting(tuning: pd.DataFrame) -> pd.Series:
    """Best mean score across folds. A tie goes to the smaller model."""
    scores = tuning.groupby(["num_leaves", "rounds"])["score"].agg(["mean", "min"]).reset_index()
    return scores.sort_values(["mean", "num_leaves", "rounds"], ascending=[False, True, True]).iloc[0]


def pass_mark(summary: pd.DataFrame, calibration: pd.DataFrame) -> dict:
    """The test fixed before the final run: better than the baseline on both questions, and calibrated."""
    if_low = float(summary.loc["if_log_loss", "gain_ci_low"])
    when_low = float(summary.loc["when_log_loss", "gain_ci_low"])
    worst_gap = float(calibration["gap"].abs().max())
    return {
        "if_log_loss_gain_ci_low": if_low,
        "when_log_loss_gain_ci_low": when_low,
        "max_calibration_gap": worst_gap,
        "calibration_tolerance": CALIBRATION_TOLERANCE,
        "passed": bool(if_low > 0 and when_low > 0 and worst_gap <= CALIBRATION_TOLERANCE),
    }


def final_dip_test(
    curves: pd.DataFrame,
    decisions: Decisions,
    split: pd.Timestamp,
    cfg: Config,
    dip_cfg: DipConfig,
    num_leaves: int,
    rounds: int,
) -> dict:
    """Fit on everything before the split and score the departures first shopped for after it."""
    train_curves, test_curves = split_trips(curves, split, cfg)
    train, test = _subsets(curves, decisions, train_curves.index, test_curves.index)
    train_hazard, test_hazard, baseline = _fit_and_score_inputs(curves, train, test, cfg, dip_cfg)

    booster = fit_hazard(train, train_hazard, num_leaves, rounds)
    model = decision_scores(test, test_hazard, predict_hazard(booster, test, test_hazard))
    summary = summarize_scores(model, baseline, dip_cfg)
    calibration = pd.concat(
        {"model": calibration_table(model), "baseline": calibration_table(baseline)}, names=["forecast"]
    )
    return {
        "summary": summary,
        "calibration": calibration,
        "by_lead": by_lead_table(model, baseline),
        "importance": feature_importance(booster),
        "pass_mark": pass_mark(summary, calibration.loc["model"]),
        "counts": {
            "train_trips": len(train_curves),
            "test_trips": len(test_curves),
            "test_departure_dates": int(test_curves.index.get_level_values("flightDate").nunique()),
            "train_decisions": len(train),
            "test_decisions": len(test),
            "train_hazard_rows": len(train_hazard),
            "test_hazard_rows": len(test_hazard),
            "test_dip_rate": float(model["any_dip"].mean()),
        },
    }
