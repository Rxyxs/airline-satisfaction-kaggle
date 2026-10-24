"""Feature views. Every encoder that sees labels is fitted inside a CV fold only."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

from .data import CATEGORICAL, FEATURES

PAIRS = list(itertools.combinations(FEATURES, 2))


def base(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns; the four text columns as pandas categories with a fixed level order."""
    out = df[FEATURES].copy()
    for c in CATEGORICAL:
        out[c] = pd.Categorical(out[c], categories=sorted(df[c].dropna().unique()))
    return out


def align_categories(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CATEGORICAL:
        out[c] = pd.Categorical(out[c].astype(str), categories=ref[c].cat.categories)
    return out


def value_keys(df: pd.DataFrame, pairs: bool = True) -> pd.DataFrame:
    """Each column (and optionally each pair of columns) as a string key: every value is a level."""
    s = df[FEATURES].astype(str)
    keys = {c: s[c] for c in FEATURES}
    if pairs:
        for a, b in PAIRS:
            keys[f"{a}|{b}"] = s[a] + "_" + s[b]
    return pd.DataFrame(keys, index=df.index)


def target_encode(
    keys_fit: pd.DataFrame, y_fit: pd.Series, *others: pd.DataFrame, seed: int = 0
) -> list[np.ndarray]:
    """Cross-fitted encoding for the fitting rows, plain encoding for every other frame."""
    te = TargetEncoder(target_type="binary", cv=5, random_state=seed)
    fit = te.fit_transform(keys_fit, y_fit)
    return [fit] + [te.transform(o) for o in others]


def with_encoding(frame: pd.DataFrame, enc: np.ndarray, names: list[str]) -> pd.DataFrame:
    te = pd.DataFrame(enc, columns=[f"te_{n}" for n in names], index=frame.index)
    return pd.concat([frame, te], axis=1)


def counts(df: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Frequency of each value over train+test (label-free, so safe outside the folds)."""
    out = pd.DataFrame(index=df.index)
    for c in FEATURES:
        out[f"cnt_{c}"] = df[c].map(pool[c].value_counts()).fillna(0).to_numpy()
    return out
