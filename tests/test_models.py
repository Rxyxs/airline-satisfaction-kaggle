import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from airline import models as M
from airline.blend import rank_mean
from test_features import frame


def labelled(n, seed):
    X = frame(n, seed)
    y = pd.Series(((X["Online boarding"] >= 3) ^ (np.random.default_rng(seed).random(n) < 0.1)).astype(int))
    return X, y


def test_cpu_models_learn_a_planted_rule():
    X, y = labelled(1500, 1)
    V, yv = labelled(500, 2)
    E, _ = labelled(50, 3)
    pool = pd.concat([X, V, E])
    for fit in (M.logreg, M.lgbm_raw, M.lgbm_values):
        r = fit(X, y, V, yv, {"extra": E}, pool)
        assert roc_auc_score(yv, r.valid) > 0.85, fit.__name__
        assert r.extra["extra"].shape == (50,)


def test_rank_mean_is_order_preserving():
    m = np.array([[0.1, 10], [0.5, 20], [0.9, 30]])
    assert np.all(np.diff(rank_mean(m)) > 0)


def test_build_view_adds_the_requested_groups():
    X, y = labelled(300, 4)
    V, _ = labelled(50, 5)
    a, b, ex = M.build_view(("counts", "te_single", "digits"), X, y, V, {"e": V}, pd.concat([X, V]), 0)
    assert a.shape[1] == b.shape[1] == ex["e"].shape[1] == 21 + 21 + 21 + 20
    assert not any(c.startswith("te_") and "|" in c for c in a.columns)
