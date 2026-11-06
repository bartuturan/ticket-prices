"""Learn when to wait by working backward from the last buying day.

For each day before departure, a model estimates what the traveller will end up paying
by waiting, as a ratio of today's price. The target is the price the rule really pays from
the next day on, not the future low, which nobody can time.
"""

from dataclasses import dataclass, field
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

from ticket_prices.backtest import Config
from ticket_prices.features import CATEGORICAL, FEATURES, LEAD, NUMERIC

TARGET_CLIP = (0.25, 4.0)  # a few extreme ratios shouldn't dominate a mean-squared fit
GBM_PARAMS = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=15,
    min_data_in_leaf=100,
    bagging_fraction=0.8,
    bagging_freq=1,
    feature_fraction=0.8,
    lambda_l2=1.0,
    num_threads=4,
    seed=0,
    verbosity=-1,
)
GBM_ROUNDS = 100


class GradientBoosting:
    def __init__(self, **overrides):
        self.params = {**GBM_PARAMS, **overrides}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        data = lgb.Dataset(X, label=np.asarray(y), categorical_feature=CATEGORICAL, free_raw_data=False)
        self.booster_ = lgb.train(self.params, data, num_boost_round=GBM_ROUNDS)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster_.predict(X)


class RidgeLinear:
    """Ridge regression with median imputation, scaling and one-hot categories."""

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        numeric = X[NUMERIC].astype(float).fillna(self.medians_)
        scaled = ((numeric - self.means_) / self.scales_).to_numpy()
        dummies = [
            (X[name].astype(object).to_numpy()[:, None] == np.array(levels, dtype=object)[None, :]).astype(float)
            for name, levels in self.levels_.items()
        ]
        return np.hstack([scaled, *dummies])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        numeric = X[NUMERIC].astype(float)
        self.medians_ = numeric.median().fillna(0.0)
        filled = numeric.fillna(self.medians_)
        self.means_ = filled.mean()
        self.scales_ = filled.std().replace(0.0, 1.0).fillna(1.0)
        self.levels_ = {name: list(X[name].astype(object).unique()) for name in CATEGORICAL}

        design = self._design(X)
        target = np.asarray(y, dtype=float)
        self.intercept_ = target.mean()
        self.center_ = design.mean(axis=0)
        centered = design - self.center_
        gram = centered.T @ centered + self.alpha * np.eye(centered.shape[1])
        self.coef_ = np.linalg.solve(gram, centered.T @ (target - self.intercept_))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept_ + (self._design(X) - self.center_) @ self.coef_


MODEL_KINDS: dict[str, Callable] = {"lightgbm": GradientBoosting, "linear": RidgeLinear}


@dataclass
class Policy:
    margin: float
    models: dict = field(default_factory=dict)
    targets: dict = field(default_factory=dict)


def rows_at_lead(panel: pd.DataFrame, lead: int) -> pd.DataFrame:
    return panel[panel.index.get_level_values(LEAD) == lead].droplevel(LEAD)


def _date_blocks(rows: pd.DataFrame, n_folds: int) -> np.ndarray:
    """Assign each row to a block of consecutive departure dates."""
    dates = rows.index.get_level_values("flightDate")
    unique = np.sort(dates.unique())
    block_of_date = {}
    for block, members in enumerate(np.array_split(unique, min(n_folds, len(unique)))):
        for date in members:
            block_of_date[date] = block
    return np.array([block_of_date[date] for date in dates.to_numpy()])


def _out_of_fold(rows: pd.DataFrame, target: pd.Series, make_model: Callable, n_folds: int) -> np.ndarray:
    """Predict each row from a model that never saw its departure dates.

    Without this, the costs passed back to earlier days would come from in-sample
    decisions and make waiting look better than it is.
    """
    blocks = _date_blocks(rows, n_folds)
    predictions = np.full(len(rows), np.nan)
    if blocks.max() == 0:
        return make_model().fit(rows[FEATURES], target).predict(rows[FEATURES])
    for block in np.unique(blocks):
        held_out = blocks == block
        model = make_model().fit(rows.loc[~held_out, FEATURES], target[~held_out])
        predictions[held_out] = model.predict(rows.loc[held_out, FEATURES])
    return predictions


def fit_policy(
    panel: pd.DataFrame,
    cfg: Config,
    margin: float,
    make_model: Callable,
    n_folds: int = 3,
    min_rows: int = 200,
) -> Policy:
    """Fit one model per lead, from the day before the last buying day up to the first.

    Every row in `panel` must have its whole future known in the training period. A trip
    with no quote on the last buying day has no known outcome and isn't used. Leads with
    fewer than `min_rows` usable rows get no model, which means buy.
    """
    policy = Policy(margin=margin)
    paid = rows_at_lead(panel, cfg.end_lead)["price"]

    for lead in range(cfg.end_lead + 1, cfg.start_lead + 1):
        rows = rows_at_lead(panel, lead)
        rows = rows[rows.index.isin(paid.index)]
        target = (paid.reindex(rows.index) / rows["price"]).clip(*TARGET_CLIP)
        policy.targets[lead] = target

        if len(rows) < min_rows:
            wait = np.zeros(len(rows), dtype=bool)
        else:
            predicted = _out_of_fold(rows, target, make_model, n_folds)
            wait = predicted < 1 - margin
            policy.models[lead] = make_model().fit(rows[FEATURES], target)

        # Trips with no quote today carry their later cost unchanged.
        paid = paid.copy()
        paid.iloc[paid.index.get_indexer(rows.index[~wait])] = rows["price"].to_numpy()[~wait]

    return policy


def simulate(panel: pd.DataFrame, trips: pd.MultiIndex, policy: Policy, cfg: Config) -> pd.DataFrame:
    """Walk each trip forward day by day. A day with no quote passes without a decision.

    Returns the price paid and the lead it was bought at. Each trip must have a quote on
    the last buying day, where buying is forced.
    """
    paid = pd.Series(np.nan, index=trips, dtype=float)
    bought_at = pd.Series(np.nan, index=trips, dtype=float)
    open_trips = trips

    for lead in range(cfg.start_lead, cfg.end_lead, -1):
        rows = rows_at_lead(panel, lead)
        rows = rows[rows.index.isin(open_trips)]
        if rows.empty:
            continue
        model = policy.models.get(lead)
        if model is None:
            buy = np.ones(len(rows), dtype=bool)
        else:
            buy = model.predict(rows[FEATURES]) >= 1 - policy.margin
        positions = trips.get_indexer(rows.index[buy])
        paid.iloc[positions] = rows["price"].to_numpy()[buy]
        bought_at.iloc[positions] = lead
        open_trips = open_trips.difference(rows.index[buy])

    last_day = rows_at_lead(panel, cfg.end_lead)["price"]
    positions = trips.get_indexer(open_trips)
    paid.iloc[positions] = last_day.reindex(open_trips).to_numpy()
    bought_at.iloc[positions] = cfg.end_lead
    return pd.DataFrame({"paid": paid, "bought_at": bought_at})


def fit_bad_case(panel: pd.DataFrame, policy: Policy, quantile: float = 0.9) -> dict:
    """Per-lead models of a bad-case outcome of waiting, as a ratio of today's price."""
    models = {}
    for lead, target in policy.targets.items():
        if lead in policy.models:
            rows = rows_at_lead(panel, lead).reindex(target.index)
            models[lead] = GradientBoosting(objective="quantile", alpha=quantile).fit(rows[FEATURES], target)
    return models


def training_rows(panel: pd.DataFrame, split: pd.Timestamp, cfg: Config) -> pd.DataFrame:
    """Rows of trips whose final buying day falls before the split."""
    flights = panel.index.get_level_values("flightDate")
    return panel[flights < split + pd.Timedelta(days=cfg.end_lead)]
