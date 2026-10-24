"""5-fold CV for every model: out-of-fold, test and original-survey predictions.

    python -m airline.run                 # all models (skips ones already on disk)
    python -m airline.run lgbm_raw --force

Each fold model also scores the 129,880-row original survey. That number says how much of what a
model learned is about passengers (it transfers to the real survey) rather than about the generator
that synthesised the competition rows (it does not).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .data import ROOT, load_competition, load_original
from .models import MODELS

PREDS = ROOT / "data" / "preds"
REPORTS = ROOT / "reports"
N_FOLDS = 5
SEED = 0


def folds(y: pd.Series):
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def run_model(name: str, X, y, T, Xo, yo) -> dict:
    fit = MODELS[name]
    pool = pd.concat([X, T], ignore_index=True)
    oof = np.zeros(len(y))
    test = np.zeros(len(T))
    orig = np.zeros(len(Xo))
    fold_auc, fold_orig_auc, iters = [], [], []
    t0 = time.time()
    for k, (tr, va) in enumerate(folds(y)):
        r = fit(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], {"test": T, "original": Xo}, pool, seed=SEED)
        oof[va] = r.valid
        test += r.extra["test"] / N_FOLDS
        orig += r.extra["original"] / N_FOLDS
        fold_auc.append(roc_auc_score(y.iloc[va], r.valid))
        fold_orig_auc.append(roc_auc_score(yo, r.extra["original"]))
        iters.append(r.best_iteration)
        print(f"{name} fold {k}: cv {fold_auc[-1]:.5f}  original {fold_orig_auc[-1]:.5f}  "
              f"iter {r.best_iteration}  {time.time() - t0:.0f}s", flush=True)
    PREDS.mkdir(parents=True, exist_ok=True)
    np.savez(PREDS / f"{name}.npz", oof=oof, test=test, original=orig)
    return {
        "cv_auc": roc_auc_score(y, oof),
        "fold_auc": fold_auc,
        "fold_auc_std": float(np.std(fold_auc)),
        "original_auc_mean_of_folds": float(np.mean(fold_orig_auc)),
        "original_auc_fold_average": roc_auc_score(yo, orig),
        "best_iterations": iters,
        "minutes": round((time.time() - t0) / 60, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=list(MODELS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    X, y, T = load_competition()
    Xo, yo = load_original()
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "cv.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    for name in args.models:
        if name in report and (PREDS / f"{name}.npz").exists() and not args.force:
            print(f"{name}: cached, cv {report[name]['cv_auc']:.5f}")
            continue
        report[name] = run_model(name, X, y, T.drop(columns="id"), Xo, yo)
        path.write_text(json.dumps(report, indent=2))
        print(f"{name}: cv {report[name]['cv_auc']:.5f}  original {report[name]['original_auc_fold_average']:.5f}")


if __name__ == "__main__":
    main()
