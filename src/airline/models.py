"""Five models, each trained on one CV fold and scored on validation, test and the original survey.

Every model gets the same call: fit(train rows, train labels, valid rows, valid labels, extra frames)
and returns probabilities for valid and each extra frame. Early stopping watches the validation fold
(standard for Kaggle OOF stacking); the CV score is therefore slightly optimistic, equally for all
tree models, and the gap is measured against the public leaderboard in reports/leaderboard.json.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer

from . import features as F
from .data import CATEGORICAL, FEATURES, NUMERIC, RATINGS

Frames = dict[str, pd.DataFrame]


@dataclass
class FoldResult:
    valid: np.ndarray
    extra: dict[str, np.ndarray]
    best_iteration: int | None


def _value_view(tr: pd.DataFrame, y_tr: pd.Series, va: pd.DataFrame, extra: Frames, pool: pd.DataFrame, seed: int):
    """Raw columns + value counts + target encoding of every value and every pair of columns."""
    ref = F.base(pool)
    keys = [F.value_keys(d) for d in (tr, va, *extra.values())]
    encs = F.target_encode(keys[0], y_tr, *keys[1:], seed=seed)
    names = list(keys[0].columns)
    out = []
    for d, e in zip((tr, va, *extra.values()), encs):
        frame = pd.concat([F.align_categories(d[FEATURES], ref), F.counts(d, pool)], axis=1)
        out.append(F.with_encoding(frame, e, names))
    return out[0], out[1], dict(zip(extra, out[2:]))


def logreg(tr, y_tr, va, y_va, extra: Frames, pool, seed=0) -> FoldResult:
    """Linear baseline: standardised numbers and ratings, one-hot text columns."""
    num = NUMERIC + RATINGS
    pre = make_column_transformer(
        (make_pipeline(StandardScaler()), num),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    )
    model = make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))
    fill = lambda d: d[FEATURES].fillna({"Arrival Delay in Minutes": 0})  # noqa: E731
    model.fit(fill(tr), y_tr)
    p = lambda d: model.predict_proba(fill(d))[:, 1]  # noqa: E731
    return FoldResult(p(va), {k: p(v) for k, v in extra.items()}, None)


LGB_PARAMS = dict(
    n_estimators=10000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.5, verbose=-1,
)


def _lgb(tr, y_tr, va, y_va, extra: Frames, seed: int) -> FoldResult:
    model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
    model.fit(tr, y_tr, eval_set=[(va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
    p = lambda d: model.predict_proba(d)[:, 1]  # noqa: E731
    return FoldResult(p(va), {k: p(v) for k, v in extra.items()}, model.best_iteration_)


def lgbm_raw(tr, y_tr, va, y_va, extra: Frames, pool, seed=0) -> FoldResult:
    """LightGBM on the 21 columns as given."""
    ref = F.base(pool)
    al = lambda d: F.align_categories(d[FEATURES], ref)  # noqa: E731
    return _lgb(al(tr), y_tr, al(va), y_va, {k: al(v) for k, v in extra.items()}, seed)


def lgbm_values(tr, y_tr, va, y_va, extra: Frames, pool, seed=0) -> FoldResult:
    """LightGBM on raw columns + counts + target-encoded values and value pairs."""
    a, b, ex = _value_view(tr, y_tr, va, extra, pool, seed)
    return _lgb(a, y_tr, b, y_va, ex, seed)


def xgb_values(tr, y_tr, va, y_va, extra: Frames, pool, seed=0) -> FoldResult:
    """XGBoost (GPU) on the same value view as lgbm_values."""
    a, b, ex = _value_view(tr, y_tr, va, extra, pool, seed)
    model = xgb.XGBClassifier(
        n_estimators=10000, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.5,
        min_child_weight=5, tree_method="hist", device="cuda", enable_categorical=True,
        eval_metric="auc", early_stopping_rounds=200, random_state=seed,
    )
    model.fit(a, y_tr, eval_set=[(b, y_va)], verbose=False)
    p = lambda d: model.predict_proba(d)[:, 1]  # noqa: E731
    return FoldResult(p(b), {k: p(v) for k, v in ex.items()}, int(model.best_iteration))


def catboost_values(tr, y_tr, va, y_va, extra: Frames, pool, seed=0) -> FoldResult:
    """CatBoost (GPU) with every column also given as a categorical string, plus raw numbers."""
    def view(d):
        s = d[FEATURES].astype(str).add_suffix("_v")
        return pd.concat([s, d[NUMERIC]], axis=1)
    cats = [f"{c}_v" for c in FEATURES]
    model = CatBoostClassifier(
        iterations=8000, learning_rate=0.08, depth=7, task_type="GPU", eval_metric="AUC",
        od_type="Iter", od_wait=300, verbose=0, cat_features=cats, random_seed=seed,
    )
    model.fit(view(tr), y_tr, eval_set=(view(va), y_va), use_best_model=True)
    p = lambda d: model.predict_proba(view(d))[:, 1]  # noqa: E731
    return FoldResult(p(va), {k: p(v) for k, v in extra.items()}, int(model.get_best_iteration()))


MODELS: dict[str, Callable[..., FoldResult]] = {
    "logreg": logreg,
    "lgbm_raw": lgbm_raw,
    "lgbm_values": lgbm_values,
    "xgb_values": xgb_values,
    "catboost_values": catboost_values,
}
