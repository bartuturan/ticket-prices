"""Walk-forward replay of the dip-timing model with a weekly recalibration of its probabilities.

Every Monday `W` the model is refitted on trips that have finished, and its daily dip chances
are rescaled using trips still being shopped for, whose recent outcomes are already known.
The week `[W, W+7)` is then scored. Nothing dated on or after `W` is used before scoring.

A day's outcome is known on the next quoted day, because the catchable test needs that day's
price (the price a buyer pays one quoted day later).
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ticket_prices.backtest import Config
from ticket_prices.dip_model import (
    PROB_CLIP,
    Decisions,
    Hazard,
    baseline_predict,
    baseline_table,
    build_hazard,
    calibration_table,
    decision_scores,
    design_matrix,
    fit_hazard,
    pass_mark,
    predict_hazard,
    summarize_scores,
)
from ticket_prices.dips import DipConfig

WALK_START = "2022-06-06"
WALK_END = "2022-08-29"
WINDOWS = (28, 14)
MIN_EVENTS = 100  # dips and non-dips needed in the window before the correction is fitted
MAX_NEWTON = 50
HEADLINE = "shift_28"


def variant_names(windows: tuple[int, ...] = WINDOWS) -> list[str]:
    return ["raw", *[f"{kind}_{window}" for window in windows for kind in ("shift", "shift_scale")]]


def settled_columns(prices: np.ndarray) -> np.ndarray:
    """For every trip and day, the column of the day on which that day's label becomes known.

    A quoted day is settled by the next quoted day after it, or by itself on the last buying day.
    A day with no quote is settled that day. Columns run from the earliest day to the last buying
    day, so the lead of a settled column is `leads[column]`.
    """
    columns = np.arange(prices.shape[1], dtype=float)
    quoted = ~np.isnan(prices)
    index = np.where(quoted, columns[None, :], np.nan)
    filled = pd.DataFrame(index).bfill(axis=1).to_numpy()
    settled = np.empty_like(filled)
    settled[:, :-1] = filled[:, 1:]
    settled[:, -1] = columns[-1]
    own = np.broadcast_to(columns[None, :], prices.shape)
    settled = np.where(quoted & ~np.isnan(settled), settled, own)
    return settled.astype(int)  # a column index, so the lead is `leads[settled]`


def _days(values: np.ndarray) -> np.ndarray:
    return values.astype("datetime64[D]").astype(np.int64)


def _group_cummax(row: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Running maximum that restarts at each decision. `row` is sorted ascending."""
    low = values.min()
    width = values.max() - low + 1
    shifted = values - low + row.astype(np.int64) * width
    return np.maximum.accumulate(shifted) - row.astype(np.int64) * width + low


@dataclass
class RecalibrationRows:
    """At-risk days on trips still being shopped for, whose outcome is known and recent."""

    decisions: Decisions
    hazard: Hazard
    keep: np.ndarray  # mask over hazard rows


def recalibration_rows(
    prices: np.ndarray,
    leads: np.ndarray,
    settled: np.ndarray,
    decisions: Decisions,
    week: pd.Timestamp,
    window_days: int,
    dip_cfg: DipConfig,
    end_lead: int,
) -> RecalibrationRows:
    """Hazard rows usable for recalibration on `week`.

    Open trips depart after `week`, and their decisions were made before it. A row is kept if
    the outcomes of its day and of every earlier day of that decision were known before `week`,
    and the latest of them within the last `window_days`. The labels of such rows depend only on
    quotes before `week`, so building them from the full price table leaks nothing.
    """
    week_day = int(_days(np.array([np.datetime64(week)]))[0])
    flight = _days(decisions.flight_date)
    open_mask = (flight > week_day) & (flight - decisions.lead < week_day)
    opened = decisions.take(open_mask)
    hazard = build_hazard(prices, leads, opened.trip, opened.lead, dip_cfg, end_lead)
    if len(hazard) == 0:
        return RecalibrationRows(opened, hazard, np.zeros(0, dtype=bool))

    top = int(leads[0])
    columns = (top - hazard.k).astype(np.intp)
    trips = opened.trip[hazard.row]
    known_lead = leads[settled[trips, columns]]
    known = _group_cummax(hazard.row, flight[open_mask][hazard.row] - known_lead)
    keep = hazard.at_risk & (known < week_day) & (known >= week_day - window_days)
    return RecalibrationRows(opened, hazard, keep)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def fit_recalibration(h: np.ndarray, y: np.ndarray, scale: bool = False, min_events: int = MIN_EVENTS) -> tuple[float, float]:
    """Fit logit(h') = a + b * logit(h) by logistic regression, with b = 1 unless `scale`.

    With too few dips or non-dips in the window the correction stays at a = 0, b = 1.
    """
    y = np.asarray(y, dtype=float)
    events = y.sum()
    if events < min_events or len(y) - events < min_events:
        return 0.0, 1.0
    z = _logit(h)
    design = np.column_stack([np.ones_like(z), z]) if scale else np.ones((len(z), 1))
    offset = np.zeros_like(z) if scale else z
    beta = np.array([0.0, 1.0]) if scale else np.zeros(1)
    for _ in range(MAX_NEWTON):
        p = _sigmoid(design @ beta + offset)
        gradient = design.T @ (y - p)
        hessian = design.T @ (design * (p * (1 - p))[:, None]) + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    if not np.isfinite(beta).all():
        return 0.0, 1.0
    return (float(beta[0]), float(beta[1])) if scale else (float(beta[0]), 1.0)


def apply_recalibration(h: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip(_sigmoid(a + b * _logit(h)), PROB_CLIP, 1 - PROB_CLIP)


def walk_forward(
    curves: pd.DataFrame,
    decisions: Decisions,
    weeks: pd.DatetimeIndex,
    cfg: Config,
    dip_cfg: DipConfig,
    num_leaves: int,
    rounds: int,
    windows: tuple[int, ...] = WINDOWS,
    progress=lambda message: None,
) -> dict:
    """Replay the weeks in order. Returns per-decision scores for every variant, and a weekly table."""
    prices = curves.to_numpy(dtype=float)
    leads = np.asarray(curves.columns)
    settled = settled_columns(prices)
    flight = _days(decisions.flight_date)
    made = flight - decisions.lead  # the search day of each decision

    scores: dict[str, dict[str, list]] = {name: {"model": [], "baseline": []} for name in variant_names(windows)}
    weekly = []
    for week in weeks:
        started = time.time()
        day = int(_days(np.array([np.datetime64(week)]))[0])

        train = decisions.take(flight < day + cfg.end_lead)  # trips that finished before the week
        train_hazard = build_hazard(prices, leads, train.trip, train.lead, dip_cfg, cfg.end_lead)
        table = baseline_table(
            train.lead[train_hazard.row], train_hazard.k, train_hazard.y, train_hazard.at_risk, cfg.start_lead
        )
        booster = fit_hazard(train, train_hazard, num_leaves, rounds)

        scored = decisions.take((made >= day) & (made < day + 7))
        scored_hazard = build_hazard(prices, leads, scored.trip, scored.lead, dip_cfg, cfg.end_lead)
        score_h = {
            "model": predict_hazard(booster, scored, scored_hazard),
            "baseline": baseline_predict(table, scored, scored_hazard),
        }

        recal = {
            window: recalibration_rows(prices, leads, settled, decisions, week, window, dip_cfg, cfg.end_lead)
            for window in windows
        }
        recal_h = {}
        for window, rows in recal.items():
            keep = rows.keep
            if not keep.any():
                recal_h[window] = {"model": np.zeros(0), "baseline": np.zeros(0)}
                continue
            kept = Hazard(rows.hazard.row[keep], rows.hazard.k[keep], rows.hazard.y[keep], rows.hazard.at_risk[keep])
            recal_h[window] = {
                "model": np.clip(booster.predict(design_matrix(rows.decisions, rows.hazard, keep)), PROB_CLIP, 1 - PROB_CLIP),
                "baseline": baseline_predict(table, rows.decisions, kept),
            }

        for name in variant_names(windows):
            record = {"week": week, "variant": name, "train_trips": int(len(np.unique(train.trip)))}
            for source in ("model", "baseline"):
                a, b, used, events = 0.0, 1.0, 0, 0
                h = score_h[source]
                if name != "raw":
                    kind, window = name.rsplit("_", 1)[0], int(name.rsplit("_", 1)[1])
                    keep = recal[window].keep
                    y = recal[window].hazard.y[keep]
                    a, b = fit_recalibration(recal_h[window][source], y, scale=kind == "shift_scale")
                    h = apply_recalibration(h, a, b)
                    used, events = int(keep.sum()), int(y.sum())
                result = decision_scores(scored, scored_hazard, h)
                result["week"] = week
                scores[name][source].append(result)
                record.update(
                    {
                        f"{source}_a": a,
                        f"{source}_b": b,
                        f"{source}_predicted": float(result["p_if"].mean()),
                        f"{source}_if_log_loss": float(result["if_log_loss"].mean()),
                        f"{source}_when_log_loss": float(result["when_log_loss"].mean()),
                        "recalibration_rows": used,
                        "recalibration_dips": events,
                    }
                )
            record["decisions"] = len(scored)
            record["observed"] = float(result["any_dip"].mean())
            weekly.append(record)
        progress(f"{week.date()}: {len(train):,} train, {len(scored):,} scored decisions, {time.time() - started:.0f}s")

    pooled = {
        name: {source: pd.concat(frames, ignore_index=True) for source, frames in sources.items()}
        for name, sources in scores.items()
    }
    return {"scores": pooled, "weekly": pd.DataFrame(weekly)}


def summarize_walk(scores: dict, dip_cfg: DipConfig) -> dict:
    """Pooled metrics and calibration for every variant, and the success check on the headline one."""
    summary = pd.concat(
        {name: summarize_scores(parts["model"], parts["baseline"], dip_cfg) for name, parts in scores.items()},
        names=["variant"],
    )
    calibration = pd.concat(
        {name: calibration_table(parts["model"]) for name, parts in scores.items()}, names=["variant"]
    )
    return {
        "summary": summary,
        "calibration": calibration,
        "pass_mark": pass_mark(summary.loc[HEADLINE], calibration.loc[HEADLINE]),
    }


# --- Lagged refit, recalibrated on recently departed trips -------------------------------------

LAG_START = "2022-07-04"
DAY_BANDS = ((1, 7), (8, 14), (15, 21), (22, 29))  # days to departure of the later day
LAG_SPECS = {  # variant -> (lag in days, correction)
    "refit": (0, "raw"),
    "lag28_raw": (28, "raw"),
    "lag28_shift": (28, "shift"),
    "lag28_band": (28, "band"),
    "lag14_band": (14, "band"),
}
HEADLINE_LAG = "lag28_band"


def recent_departure_rows(
    prices: np.ndarray,
    leads: np.ndarray,
    decisions: Decisions,
    week: pd.Timestamp,
    lag_days: int,
    dip_cfg: DipConfig,
    end_lead: int,
) -> RecalibrationRows:
    """At-risk days of every decision on trips that departed in `(week - lag_days, week]`.

    Their last quote is dated before `week`, so every outcome is known, and a model trained on
    trips that departed on or before `week - lag_days` has never seen them.
    """
    day = int(_days(np.array([np.datetime64(week)]))[0])
    flight = _days(decisions.flight_date)
    recent = decisions.take((flight > day - lag_days) & (flight <= day))
    hazard = build_hazard(prices, leads, recent.trip, recent.lead, dip_cfg, end_lead)
    return RecalibrationRows(recent, hazard, hazard.at_risk)


def fit_band_shifts(
    h: np.ndarray, y: np.ndarray, k: np.ndarray, bands=DAY_BANDS, min_events: int = MIN_EVENTS
) -> tuple[np.ndarray, float]:
    """One shift per band of days to departure, and the shift fitted on all rows.

    A band with too few dips or non-dips takes the all-rows shift.
    """
    y = np.asarray(y, dtype=float)
    pooled, _ = fit_recalibration(h, y, scale=False, min_events=min_events)
    shifts = np.full(len(bands), pooled)
    for position, (low, high) in enumerate(bands):
        inside = (k >= low) & (k <= high)
        events = y[inside].sum()
        if events >= min_events and inside.sum() - events >= min_events:
            shifts[position], _ = fit_recalibration(h[inside], y[inside], scale=False, min_events=min_events)
    return shifts, pooled


def apply_band_shifts(h: np.ndarray, k: np.ndarray, shifts: np.ndarray, bands=DAY_BANDS, default: float = 0.0) -> np.ndarray:
    shift = np.full(len(h), float(default))
    for position, (low, high) in enumerate(bands):
        shift[(k >= low) & (k <= high)] = shifts[position]
    return np.clip(_sigmoid(_logit(h) + shift), PROB_CLIP, 1 - PROB_CLIP)


def _fit_cutoff(prices, leads, decisions, flight, day, cfg, dip_cfg, num_leaves, rounds):
    """Model and baseline fitted on trips that departed on or before `day`."""
    train = decisions.take(flight <= day)
    hazard = build_hazard(prices, leads, train.trip, train.lead, dip_cfg, cfg.end_lead)
    table = baseline_table(train.lead[hazard.row], hazard.k, hazard.y, hazard.at_risk, cfg.start_lead)
    return fit_hazard(train, hazard, num_leaves, rounds), table, int(len(np.unique(train.trip)))


def _recent_predictions(booster, table, rows: RecalibrationRows) -> dict:
    keep = rows.keep
    if not keep.any():
        empty = np.zeros(0)
        return {"model": empty, "baseline": empty, "k": empty, "y": empty}
    kept = Hazard(rows.hazard.row[keep], rows.hazard.k[keep], rows.hazard.y[keep], rows.hazard.at_risk[keep])
    return {
        "model": np.clip(booster.predict(design_matrix(rows.decisions, rows.hazard, keep)), PROB_CLIP, 1 - PROB_CLIP),
        "baseline": baseline_predict(table, rows.decisions, kept),
        "k": kept.k,
        "y": kept.y,
    }


def walk_forward_lagged(
    curves: pd.DataFrame,
    decisions: Decisions,
    weeks: pd.DatetimeIndex,
    cfg: Config,
    dip_cfg: DipConfig,
    num_leaves: int,
    rounds: int,
    progress=lambda message: None,
) -> dict:
    """Replay the weeks with a model trained `lag` days behind and corrected from recent departures."""
    prices = curves.to_numpy(dtype=float)
    leads = np.asarray(curves.columns)
    flight = _days(decisions.flight_date)
    made = flight - decisions.lead
    lags = sorted({lag for lag, _ in LAG_SPECS.values()})
    correcting = sorted({lag for lag, mode in LAG_SPECS.values() if mode != "raw"})

    scores: dict[str, dict[str, list]] = {name: {"model": [], "baseline": []} for name in LAG_SPECS}
    weekly = []
    for week in weeks:
        started = time.time()
        day = int(_days(np.array([np.datetime64(week)]))[0])
        fits = {
            lag: _fit_cutoff(prices, leads, decisions, flight, day - lag, cfg, dip_cfg, num_leaves, rounds) for lag in lags
        }
        recent = {
            lag: _recent_predictions(
                fits[lag][0], fits[lag][1],
                recent_departure_rows(prices, leads, decisions, week, lag, dip_cfg, cfg.end_lead),
            )
            for lag in correcting
        }

        scored = decisions.take((made >= day) & (made < day + 7))
        scored_hazard = build_hazard(prices, leads, scored.trip, scored.lead, dip_cfg, cfg.end_lead)
        score_h = {
            lag: {
                "model": predict_hazard(fits[lag][0], scored, scored_hazard),
                "baseline": baseline_predict(fits[lag][1], scored, scored_hazard),
            }
            for lag in lags
        }

        for name, (lag, mode) in LAG_SPECS.items():
            record = {"week": week, "variant": name, "train_trips": fits[lag][2], "decisions": len(scored)}
            for source in ("model", "baseline"):
                h = score_h[lag][source]
                if mode == "shift":
                    shift, _ = fit_recalibration(recent[lag][source], recent[lag]["y"], scale=False)
                    h = apply_recalibration(h, shift, 1.0)
                    record[f"{source}_shift_all"] = shift
                elif mode == "band":
                    shifts, pooled = fit_band_shifts(recent[lag][source], recent[lag]["y"], recent[lag]["k"])
                    h = apply_band_shifts(h, scored_hazard.k, shifts, default=pooled)
                    record[f"{source}_shift_all"] = pooled
                    for (low, high), value in zip(DAY_BANDS, shifts):
                        record[f"{source}_shift_{low}_{high}"] = float(value)
                result = decision_scores(scored, scored_hazard, h)
                result["week"] = week
                scores[name][source].append(result)
                record.update(
                    {
                        f"{source}_predicted": float(result["p_if"].mean()),
                        f"{source}_if_log_loss": float(result["if_log_loss"].mean()),
                        f"{source}_when_log_loss": float(result["when_log_loss"].mean()),
                    }
                )
            record["observed"] = float(result["any_dip"].mean())
            record["recalibration_rows"] = int(len(recent[lag]["y"])) if mode != "raw" else 0
            record["recalibration_dips"] = int(np.sum(recent[lag]["y"])) if mode != "raw" else 0
            weekly.append(record)
        progress(f"{week.date()}: {len(scored):,} scored decisions, {time.time() - started:.0f}s")

    pooled_scores = {
        name: {source: pd.concat(frames, ignore_index=True) for source, frames in sources.items()}
        for name, sources in scores.items()
    }
    return {"scores": pooled_scores, "weekly": pd.DataFrame(weekly)}


def summarize_lagged(scores: dict, dip_cfg: DipConfig) -> dict:
    """Pooled metrics against the equally treated baseline and against the plain weekly refit."""
    summary = pd.concat(
        {name: summarize_scores(parts["model"], parts["baseline"], dip_cfg) for name, parts in scores.items()},
        names=["variant"],
    )
    reference = scores["refit"]["model"]
    # Every variant scores the same decisions in the same order, so the rows pair up.
    vs_refit = pd.concat(
        {
            name: summarize_scores(parts["model"], reference, dip_cfg)
            for name, parts in scores.items()
            if name != "refit"
        },
        names=["variant"],
    )
    calibration = pd.concat(
        {name: calibration_table(parts["model"]) for name, parts in scores.items()}, names=["variant"]
    )
    check = pass_mark(summary.loc[HEADLINE_LAG], calibration.loc[HEADLINE_LAG])
    beats_refit = float(vs_refit.loc[(HEADLINE_LAG, "if_log_loss"), "gain_ci_low"])
    check["vs_refit_if_log_loss_gain_ci_low"] = beats_refit
    check["passed"] = bool(check["passed"] and beats_refit > 0)
    return {"summary": summary, "vs_refit": vs_refit, "calibration": calibration, "pass_mark": check}
