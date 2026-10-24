"""Combine the models' out-of-fold predictions and write the submission files.

Two blends, both chosen before any leaderboard score is seen:
- rank_mean: average rank of the four tree models, no fitted weights.
- stack: logistic regression on the models' logits. Its CV score is itself cross-validated over
  the OOF rows, so the weights are never scored on the rows that fitted them.
- stack_extended: the same stack over the four tree models plus RealMLP and lgbm_all.

    python -m airline.blend
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from .data import ROOT, load_competition, load_original
from .run import PREDS, REPORTS, folds

SUBMISSIONS = ROOT / "submissions"
TREES = ["lgbm_raw", "lgbm_values", "xgb_values", "catboost_values"]
EXTENDED = TREES + ["realmlp_values", "lgbm_all"]


def _load(names):
    d = {n: np.load(PREDS / f"{n}.npz") for n in names}
    return {part: np.column_stack([d[n][part] for n in names]) for part in ("oof", "test", "original")}


def rank_mean(m: np.ndarray) -> np.ndarray:
    return np.mean([rankdata(c) / len(c) for c in m.T], axis=0)


def _logits(m: np.ndarray) -> np.ndarray:
    return logit(np.clip(m, 1e-6, 1 - 1e-6))


def _stack(Q: dict, y) -> tuple[dict, LogisticRegression]:
    stack = LogisticRegression(C=1.0, max_iter=1000)
    cv = [(tr, va) for tr, va in folds(y)]
    oof = cross_val_predict(stack, _logits(Q["oof"]), y, cv=cv, method="predict_proba")[:, 1]
    stack.fit(_logits(Q["oof"]), y)
    out = {"oof": oof}
    out |= {p: stack.predict_proba(_logits(Q[p]))[:, 1] for p in ("test", "original")}
    return out, stack


def main() -> None:
    _, y, T = load_competition()
    _, yo = load_original()
    names = [n for n in ["logreg"] + EXTENDED if (PREDS / f"{n}.npz").exists()]
    trees = [n for n in TREES if n in names]
    extended = [n for n in EXTENDED if n in names]
    P = _load(names)
    Q = _load(trees)
    report = {}

    preds = {n: {p: P[p][:, i] for p in P} for i, n in enumerate(names)}
    preds["rank_mean"] = {p: rank_mean(Q[p]) for p in Q}
    preds["stack"], stack = _stack(Q, y)
    report["stack_weights"] = dict(zip(trees, stack.coef_[0].round(4).tolist()))
    if len(extended) > len(trees):
        preds["stack_extended"], stack = _stack(_load(extended), y)
        report["stack_extended_weights"] = dict(zip(extended, stack.coef_[0].round(4).tolist()))

    report["scores"] = {
        n: {"cv_auc": roc_auc_score(y, p["oof"]), "original_auc": roc_auc_score(yo, p["original"])}
        for n, p in preds.items()
    }
    report["oof_correlation"] = pd.DataFrame(P["oof"], columns=names).corr(method="spearman").round(4).to_dict()
    (REPORTS / "blend.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["scores"], indent=2))

    SUBMISSIONS.mkdir(exist_ok=True)
    for n, p in preds.items():
        pd.DataFrame({"id": T["id"], "satisfaction": p["test"]}).to_csv(SUBMISSIONS / f"{n}.csv", index=False)


if __name__ == "__main__":
    main()
