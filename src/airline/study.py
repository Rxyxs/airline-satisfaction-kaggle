"""What the original survey says about the competition data.

1. Train on the original survey only, score the competition train set.
2. Append the original survey to every training fold (CV still scored on competition rows only).
3. Compare distributions the generator changed (delays, the dropped rating).

    python -m airline.study
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import features as F
from .data import FEATURES, load_competition, load_original
from .models import _lgb
from .run import REPORTS, SEED, folds


def _stack_original() -> np.ndarray:
    """The stacked model's prediction for every original-survey row (from airline.blend)."""
    from .blend import TREES, _load, _logits
    from sklearn.linear_model import LogisticRegression

    _, y, _ = load_competition()
    Q = _load(TREES)
    stack = LogisticRegression(C=1.0, max_iter=1000).fit(_logits(Q["oof"]), y)
    return stack.predict_proba(_logits(Q["original"]))[:, 1]


def main() -> None:
    X, y, T = load_competition()
    Xo, yo = load_original()
    ref = F.base(pd.concat([X, T]))
    A = F.align_categories(X[FEATURES], ref)
    O = F.align_categories(Xo[FEATURES], ref)
    out: dict = {}

    # 1. original only -> competition rows. Validation = a held-out 20% of the original survey.
    rng = np.random.default_rng(SEED)
    hold = rng.random(len(O)) < 0.2
    r = _lgb(O[~hold], yo[~hold], O[hold], yo[hold], {"competition": A}, SEED)
    out["original_only"] = {
        "auc_on_original_holdout": roc_auc_score(yo[hold], r.valid),
        "auc_on_competition_train": roc_auc_score(y, r.extra["competition"]),
        "best_iteration": r.best_iteration,
    }
    print(out["original_only"], flush=True)

    # 2. competition folds with and without the original rows appended
    oof_plain, oof_aug = np.zeros(len(y)), np.zeros(len(y))
    for tr, va in folds(y):
        p = _lgb(A.iloc[tr], y.iloc[tr], A.iloc[va], y.iloc[va], {}, SEED)
        oof_plain[va] = p.valid
        Xa = pd.concat([A.iloc[tr], O], ignore_index=True)
        ya = pd.concat([y.iloc[tr], yo], ignore_index=True)
        q = _lgb(Xa, ya, A.iloc[va], y.iloc[va], {}, SEED)
        oof_aug[va] = q.valid
        print(roc_auc_score(y.iloc[va], p.valid), roc_auc_score(y.iloc[va], q.valid), flush=True)
    out["append_original"] = {
        "cv_auc_competition_only": roc_auc_score(y, oof_plain),
        "cv_auc_with_original_rows": roc_auc_score(y, oof_aug),
    }

    # 3. what the generator changed
    dep = "Departure Delay in Minutes"
    out["distribution"] = {
        "rows_competition_train": len(X),
        "rows_original": len(Xo),
        "share_satisfied_competition": float(y.mean()),
        "share_satisfied_original": float(yo.mean()),
        "share_zero_departure_delay_competition": float((X[dep] == 0).mean()),
        "share_zero_departure_delay_original": float((Xo[dep] == 0).mean()),
        "mean_departure_delay_competition": float(X[dep].mean()),
        "mean_departure_delay_original": float(Xo[dep].mean()),
        "dropped_column": "Inflight service",
        "distinct_values_competition": {c: int(X[c].nunique()) for c in FEATURES},
        "distinct_values_original": {c: int(Xo[c].nunique()) for c in FEATURES},
    }
    # 4. label-noise equivalent: flip a share of the original labels at random and find the share
    # at which the stacked model scores on the real survey what it scores on the competition.
    blend = json.loads((REPORTS / "blend.json").read_text())["scores"]["stack"]
    orig_pred = _stack_original()
    curve = []
    for rate in np.round(np.arange(0.0, 0.081, 0.005), 3):
        aucs = []
        for s in range(5):
            flip = np.random.default_rng(s).random(len(yo)) < rate
            aucs.append(roc_auc_score(np.where(flip, 1 - yo, yo), orig_pred))
        curve.append({"flip_rate": float(rate), "auc": float(np.mean(aucs))})
    target = blend["cv_auc"]
    match = min(curve, key=lambda c: abs(c["auc"] - target))
    out["label_noise_equivalent"] = {
        "stack_auc_competition_cv": target,
        "stack_auc_original": blend["original_auc"],
        "flip_rate_matching_competition_auc": match["flip_rate"],
        "curve": curve,
    }
    print(out["label_noise_equivalent"]["flip_rate_matching_competition_auc"])
    (REPORTS / "original_study.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["append_original"], indent=2))


if __name__ == "__main__":
    main()
