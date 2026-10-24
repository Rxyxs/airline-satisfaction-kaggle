import numpy as np
import pandas as pd

from airline import features as F
from airline.data import CATEGORICAL, FEATURES, NUMERIC, RATINGS


def frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = {c: rng.integers(0, 6, n) for c in RATINGS}
    d |= {c: rng.integers(0, 80, n).astype(float) for c in NUMERIC}
    d["Gender"] = rng.choice(["Male", "Female"], n)
    d["Customer Type"] = rng.choice(["Loyal Customer", "disloyal Customer"], n)
    d["Type of Travel"] = rng.choice(["Business travel", "Personal Travel"], n)
    d["Class"] = rng.choice(["Business", "Eco", "Eco Plus"], n)
    return pd.DataFrame(d)[FEATURES]


def test_value_keys_has_every_column_and_pair():
    k = F.value_keys(frame(10))
    assert k.shape[1] == len(FEATURES) + len(FEATURES) * (len(FEATURES) - 1) // 2
    assert k.shape[1] == 231


def test_align_categories_keeps_reference_levels():
    ref = F.base(frame())
    other = frame(5).assign(Class="Eco")
    out = F.align_categories(other, ref)
    for c in CATEGORICAL:
        assert list(out[c].cat.categories) == list(ref[c].cat.categories)


def test_target_encoding_of_fitting_rows_is_cross_fitted():
    """A unique value per row must not leak its own label into its encoding."""
    n = 500
    keys = pd.DataFrame({"k": [str(i) for i in range(n)]})
    y = pd.Series(np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int))
    fit, = F.target_encode(keys, y)
    # every key is unseen in the other folds, so it falls back to a prior near the mean
    assert np.all(np.abs(fit[:, 0] - 0.5) < 0.1)


def test_counts_use_pool_frequencies():
    pool = pd.DataFrame({c: [1, 1, 2] for c in FEATURES})
    d = pd.DataFrame({c: [1, 2, 3] for c in FEATURES})
    out = F.counts(d, pool)
    assert out[f"cnt_{FEATURES[0]}"].tolist() == [2, 1, 0]
