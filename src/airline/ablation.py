"""Which feature tricks are about passengers and which are about the generator?

Each feature group is added on its own to the raw-column LightGBM, on the same five folds.
Every fold model is scored twice: on its competition validation fold (CV) and on the real
129,880-passenger survey. A group that lifts CV but not the real-survey AUC is exploiting how the
synthetic rows were made; a group that lifts both carries passenger signal.

    python -m airline.ablation            # resumable: finished configurations are skipped
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import load_competition, load_original
from .models import _lgb, build_view
from .run import REPORTS, SEED, folds

CONFIGS = {
    "raw": (),
    "+counts": ("counts",),
    "+digits": ("digits",),
    "+te_single": ("te_single",),
    "+te_single+te_pairs": ("te_single", "te_pairs"),
    "+survey": ("survey",),
    "values (counts+te)": ("counts", "te_single", "te_pairs"),
    "all": ("counts", "te_single", "te_pairs", "digits", "survey"),
}


# Same folds, different LightGBM seeds: how much the two AUCs move with no change in features.
SEED_CHECK = {"raw": (), "+te_single": ("te_single",), "+digits": ("digits",)}
CHECK_SEEDS = (1, 2, 3)


def evaluate(groups, X, y, T, Xo, yo, model_seed: int = SEED) -> dict:
    pool = pd.concat([X, T], ignore_index=True)
    oof, orig = np.zeros(len(y)), np.zeros(len(yo))
    fold_orig, t0 = [], time.time()
    for tr, va in folds(y):
        a, b, ex = build_view(groups, X.iloc[tr], y.iloc[tr], X.iloc[va], {"original": Xo}, pool, SEED)
        r = _lgb(a, y.iloc[tr], b, y.iloc[va], ex, model_seed)
        oof[va] = r.valid
        orig += r.extra["original"] / len(folds(y))
        fold_orig.append(roc_auc_score(yo, r.extra["original"]))
    return {
        "groups": list(groups),
        "cv_auc": roc_auc_score(y, oof),
        "original_auc_fold_average": roc_auc_score(yo, orig),
        "original_auc_mean_of_folds": float(np.mean(fold_orig)),
        "minutes": round((time.time() - t0) / 60, 1),
    }


def main() -> None:
    X, y, T = load_competition()
    X, T = X.drop(columns="id"), T.drop(columns="id")
    Xo, yo = load_original()
    path = REPORTS / "ablation.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    for name, groups in CONFIGS.items():
        if name in report:
            continue
        report[name] = evaluate(groups, X, y, T, Xo, yo)
        r = report[name]
        print(f"{name:22s} cv {r['cv_auc']:.5f}  real {r['original_auc_fold_average']:.5f}  {r['minutes']} min", flush=True)
        path.write_text(json.dumps(report, indent=2))
    base = report["raw"]
    for name in CONFIGS:
        r = report[name]
        r["delta_cv"] = r["cv_auc"] - base["cv_auc"]
        r["delta_original"] = r["original_auc_fold_average"] - base["original_auc_fold_average"]
    path.write_text(json.dumps(report, indent=2))

    check = report.setdefault("seed_check", {})
    for name, groups in SEED_CHECK.items():
        for s in CHECK_SEEDS:
            key = f"{name} seed {s}"
            if key not in check:
                check[key] = evaluate(groups, X, y, T, Xo, yo, model_seed=s)
                print(f"{key:22s} cv {check[key]['cv_auc']:.5f}  real {check[key]['original_auc_fold_average']:.5f}", flush=True)
                path.write_text(json.dumps(report, indent=2))
    for name in SEED_CHECK:
        runs = [report[name]] + [check[f"{name} seed {s}"] for s in CHECK_SEEDS]
        check[f"{name} summary"] = {
            "cv_mean": float(np.mean([r["cv_auc"] for r in runs])),
            "cv_sd": float(np.std([r["cv_auc"] for r in runs], ddof=1)),
            "original_mean": float(np.mean([r["original_auc_fold_average"] for r in runs])),
            "original_sd": float(np.std([r["original_auc_fold_average"] for r in runs], ddof=1)),
        }
    path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
