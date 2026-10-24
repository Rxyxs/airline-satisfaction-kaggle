"""Figures for the README and the project page, drawn from reports/*.json and the raw data.

    python -m airline.figures
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import ROOT, load_competition, load_original  # noqa: E402
from .run import REPORTS  # noqa: E402

IMG = ROOT / "docs" / "img"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["lgbm_raw", "lgbm_values", "xgb_values", "catboost_values", "realmlp_values", "rank_mean", "stack",
         "stack_extended"]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def models() -> None:
    blend = json.loads((REPORTS / "blend.json").read_text())["scores"]
    lb = {r["name"]: float(r["public_score"]) for r in json.loads((REPORTS / "leaderboard.json").read_text())}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    ypos = np.arange(len(ORDER))[::-1]
    ax = axes[0]
    ax.scatter([blend[n]["cv_auc"] for n in ORDER], ypos, s=60, color=BLUE, label="5-fold CV", zorder=3,
               edgecolor="white", linewidth=1.5)
    ax.scatter([lb[n] for n in ORDER], ypos, s=60, color=ORANGE, marker="D", label="Kaggle public LB",
               zorder=3, edgecolor="white", linewidth=1.5)
    ax.set_yticks(ypos, ORDER)
    ax.set_title("Competition data (synthetic)", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("ROC AUC", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    ax = axes[1]
    ax.scatter([blend[n]["original_auc"] for n in ORDER], ypos, s=60, color=AQUA, zorder=3,
               edgecolor="white", linewidth=1.5)
    ax.set_title("Same models scored on the real survey (129,880 rows)", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("ROC AUC", color=MUTED, fontsize=9)
    for a in axes:
        _style(a)
    fig.tight_layout()
    fig.savefig(IMG / "models.png", dpi=150)
    plt.close(fig)


def noise() -> None:
    st = json.loads((REPORTS / "original_study.json").read_text())["label_noise_equivalent"]
    c = pd.DataFrame(st["curve"])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(c.flip_rate * 100, c.auc, color=BLUE, linewidth=2, marker="o", markersize=5)
    ax.axhline(st["stack_auc_competition_cv"], color=ORANGE, linewidth=1.5, linestyle="--")
    ax.text(c.flip_rate.max() * 100, st["stack_auc_competition_cv"] + 0.0015,
            f"competition CV {st['stack_auc_competition_cv']:.4f}", ha="right", color=MUTED, fontsize=9)
    r = st["flip_rate_matching_competition_auc"]
    ax.annotate(f"{r:.1%} of labels flipped", xy=(r * 100, c.set_index("flip_rate").auc[r]),
                xytext=(r * 100 + 1.2, st["stack_auc_original"] - 0.004), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlabel("share of real-survey labels flipped at random (%)", color=MUTED, fontsize=9)
    ax.set_ylabel("stack AUC on the real survey", color=MUTED, fontsize=9)
    ax.set_title("Random label noise needed to turn 0.994 into the competition's 0.961", loc="left",
                 fontsize=11, color=INK)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(IMG / "noise.png", dpi=150)
    plt.close(fig)


def shift() -> dict:
    X, y, _ = load_competition()
    Xo, yo = load_original()
    bins = [-1, 0, 15, 60, 10_000]
    labels = ["0 min", "1-15", "16-60", "> 60"]
    col = "Departure Delay in Minutes"
    share = pd.DataFrame({
        "competition": pd.cut(X[col], bins, labels=labels).value_counts(normalize=True),
        "original": pd.cut(Xo[col], bins, labels=labels).value_counts(normalize=True),
    }).loc[labels]
    sat = pd.DataFrame({
        "competition": y.groupby(pd.cut(X[col], bins, labels=labels), observed=True).mean(),
        "original": yo.groupby(pd.cut(Xo[col], bins, labels=labels), observed=True).mean(),
    }).loc[labels]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(labels))
    for ax, df, title in ((axes[0], share, "Share of passengers by departure delay"),
                          (axes[1], sat, "Share satisfied, by departure delay")):
        ax.bar(x - 0.2, df["competition"], 0.38, color=BLUE, label="competition (synthetic)")
        ax.bar(x + 0.2, df["original"], 0.38, color=AQUA, label="original survey")
        ax.set_xticks(x, labels)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        _style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "shift.png", dpi=150)
    plt.close(fig)
    return {"delay_share": share.round(4).to_dict(), "satisfied_by_delay": sat.round(4).to_dict()}


def ablation() -> None:
    rep = json.loads((REPORTS / "ablation.json").read_text())
    names = [n for n in rep if n not in ("raw", "seed_check")]
    sd = rep["seed_check"]["raw summary"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    ypos = np.arange(len(names))[::-1]
    for ax, key, sdk, color, title in (
        (axes[0], "delta_cv", "cv_sd", BLUE, "Gain on the competition (5-fold CV)"),
        (axes[1], "delta_original", "original_sd", AQUA, "Gain on the real survey"),
    ):
        vals = [rep[n][key] * 1e4 for n in names]
        band = 2 * sd[sdk] * 1e4
        ax.axvspan(-band, band, color=GRID, alpha=0.8, linewidth=0)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.hlines(ypos, 0, vals, color=color, linewidth=2)
        ax.scatter(vals, ypos, s=60, color=color, zorder=3, edgecolor="white", linewidth=1.5)
        for v, yv in zip(vals, ypos):
            label = "0.0" if abs(v) < 0.05 else f"{v:+.1f}"
            ax.text(v + (0.6 if v >= 0 else -0.6), yv, label, va="center", ha="left" if v >= 0 else "right",
                    fontsize=8.5, color=INK)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("AUC change vs raw LightGBM (x 10^-4); grey band = 2 SD across seeds", color=MUTED, fontsize=8.5)
        _style(ax)
    axes[0].set_yticks(ypos, names)
    lo = min(rep[n]["delta_original"] for n in names) * 1e4 - 1.5
    axes[1].set_xlim(lo, max(rep[n]["delta_original"] for n in names) * 1e4 + 1.5)
    axes[0].set_xlim(-1.5, max(rep[n]["delta_cv"] for n in names) * 1e4 + 3)
    fig.tight_layout()
    fig.savefig(IMG / "ablation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    models()
    noise()
    ablation()
    (REPORTS / "shift.json").write_text(json.dumps(shift(), indent=2))


if __name__ == "__main__":
    main()
